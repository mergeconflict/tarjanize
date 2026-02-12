//! Exports a modified `SymbolGraph` reflecting applied split operations.
//!
//! While `apply_splits` operates at the `TargetGraph` level (costs and
//! dependency edges), `export_symbol_graph` operates at the `SymbolGraph`
//! level — actually moving symbols between module trees to produce a new
//! `SymbolGraph` that can be serialized and used as input to subsequent
//! pipeline stages (e.g., `tarjanize condense`).

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use tarjanize_schemas::{
    Module, Package, Symbol, SymbolGraph, Target, TargetTimings,
    sum_event_times,
};

use crate::split::{SplitOperation, compute_downset};
use crate::target_graph::condense_target;

/// Exports a modified `SymbolGraph` that reflects split operations.
///
/// For each `SplitOperation`:
/// 1. Condenses the source target into its SCC DAG
/// 2. Expands the selected SCCs to a full downset (transitive closure)
/// 3. Collects symbol paths from the downset SCCs
/// 4. Creates a new package+target for the new crate
/// 5. Moves symbols from the source module tree to the new target
/// 6. Adds a dependency from the residual to the new crate
/// 7. Copies external dependencies to the new target
///
/// Returns a new `SymbolGraph` with the splits applied. The original
/// is not modified.
pub fn export_symbol_graph(
    original: &SymbolGraph,
    operations: &[SplitOperation],
) -> SymbolGraph {
    let mut graph = original.clone();

    for op in operations {
        // Condense the source target to obtain its SCC DAG. Skip
        // this operation if condensation fails (bad target name).
        let Some(intra) = condense_target(&graph, &op.source_target) else {
            continue;
        };

        // Build dependency adjacency list for downset computation.
        // IntraTargetGraph edges are (from, to) where `to` depends
        // on `from`, so for the downset we reverse: deps[to] includes
        // from.
        let n_sccs = intra.nodes.len();
        let mut deps_adj: Vec<Vec<usize>> = vec![Vec::new(); n_sccs];
        for &(from_id, to_id) in &intra.edges {
            deps_adj[to_id].push(from_id);
        }

        // Expand selected SCCs to the full downset.
        let downset = compute_downset(&deps_adj, &op.selected_sccs);

        // Collect all symbol paths that belong to downset SCCs.
        let moving_symbols: HashSet<String> = intra
            .nodes
            .iter()
            .filter(|n| downset.contains(&n.id))
            .flat_map(|n| n.symbols.iter().cloned())
            .collect();

        // Parse the source target ID into (package, target_key).
        let Some((pkg_name, target_key)) = op.source_target.split_once('/')
        else {
            continue;
        };

        // Build the target prefix for stripping symbol paths.
        let target_prefix = format!("[{}]::", op.source_target);

        // Parse each symbol path into (module_segments, symbol_name)
        // and collect them for moving.
        let parsed_paths: Vec<(Vec<String>, String)> = moving_symbols
            .iter()
            .filter_map(|path| parse_symbol_path(path, &target_prefix))
            .collect();

        // Create the new target's module tree by extracting symbols
        // from the source.
        let mut new_root = Module::default();
        let mut moved_cost: f64 = 0.0;

        // Look up the source target mutably to remove symbols.
        let source_target = graph
            .packages
            .get_mut(pkg_name)
            .and_then(|p| p.targets.get_mut(target_key));
        let Some(source_target) = source_target else {
            continue;
        };

        for (mod_segments, sym_name) in &parsed_paths {
            // Navigate to the symbol in the source module tree and
            // remove it.
            if let Some(symbol) = remove_symbol_from_module(
                &mut source_target.root,
                mod_segments,
                sym_name,
            ) {
                moved_cost += sum_event_times(&symbol.event_times_ms);

                // Insert into the new target's module tree, creating
                // intermediate modules as needed.
                insert_symbol_into_module(
                    &mut new_root,
                    mod_segments,
                    sym_name.clone(),
                    symbol,
                );
            }
        }

        // Rewrite dependencies within the moved symbols: references
        // to other moved symbols should use the new target prefix
        // instead of the old one.
        let new_target_prefix = format!("[{}/lib]::", op.new_crate_name);
        rewrite_dependencies(
            &mut new_root,
            &moving_symbols,
            &target_prefix,
            &new_target_prefix,
        );

        // Also rewrite dependencies in the residual source target:
        // references to moved symbols should point to the new target.
        rewrite_dependencies(
            &mut source_target.root,
            &moving_symbols,
            &target_prefix,
            &new_target_prefix,
        );

        // Adjust the residual's wall_time downward by the moved cost.
        let moved_dur = Duration::from_secs_f64(moved_cost / 1000.0);
        source_target.timings.wall_time =
            source_target.timings.wall_time.saturating_sub(moved_dur);

        // Copy external dependencies from the source target.
        let source_deps = source_target.dependencies.clone();

        // Add the residual as a dependency of the new crate (the
        // residual depends on the new crate, meaning the new crate
        // must build first).
        let new_target_id = format!("{}/lib", op.new_crate_name);
        source_target.dependencies.insert(new_target_id.clone());

        // Create the new package and target.
        let new_target = Target {
            timings: TargetTimings {
                wall_time: Duration::from_secs_f64(moved_cost / 1000.0),
                event_times_ms: HashMap::new(),
            },
            dependencies: source_deps,
            root: new_root,
        };
        let mut new_targets = HashMap::new();
        new_targets.insert("lib".to_string(), new_target);
        let new_package = Package {
            targets: new_targets,
        };
        graph
            .packages
            .insert(op.new_crate_name.clone(), new_package);
    }

    graph
}

/// Parses a symbol path like `[pkg/target]::mod::submod::SymName` into
/// `(vec!["mod", "submod"], "SymName")`.
///
/// Returns `None` if the path doesn't start with the expected prefix or
/// has no symbol name after stripping the prefix.
pub(crate) fn parse_symbol_path(
    path: &str,
    target_prefix: &str,
) -> Option<(Vec<String>, String)> {
    let remainder = path.strip_prefix(target_prefix)?;
    let segments: Vec<&str> = remainder.split("::").collect();
    if segments.is_empty() {
        return None;
    }
    let sym_name = segments.last()?.to_string();
    let mod_segments: Vec<String> = segments[..segments.len() - 1]
        .iter()
        .map(ToString::to_string)
        .collect();
    Some((mod_segments, sym_name))
}

/// Removes a symbol from a module tree by navigating the module path.
///
/// Returns the removed `Symbol` if found, or `None` if the module path
/// or symbol name doesn't exist. This is a destructive operation on the
/// source tree — the symbol is gone after removal.
pub(crate) fn remove_symbol_from_module(
    root: &mut Module,
    mod_segments: &[String],
    sym_name: &str,
) -> Option<Symbol> {
    let mut current = root;
    for seg in mod_segments {
        current = current.submodules.get_mut(seg)?;
    }
    current.symbols.remove(sym_name)
}

/// Inserts a symbol into a module tree, creating intermediate submodules
/// as needed.
///
/// This mirrors the module path structure from the source target so the
/// new target's module tree matches the original layout of the moved
/// symbols.
pub(crate) fn insert_symbol_into_module(
    root: &mut Module,
    mod_segments: &[String],
    sym_name: String,
    symbol: Symbol,
) {
    let mut current = root;
    for seg in mod_segments {
        current = current.submodules.entry(seg.clone()).or_default();
    }
    current.symbols.insert(sym_name, symbol);
}

/// Rewrites dependency paths in all symbols within a module tree.
///
/// For each symbol, any dependency that references a moved symbol (one
/// in `moving_symbols`) has its target prefix replaced from
/// `old_prefix` to `new_prefix`. This ensures that cross-crate
/// references point to the correct target after the split.
fn rewrite_dependencies(
    module: &mut Module,
    moving_symbols: &HashSet<String>,
    old_prefix: &str,
    new_prefix: &str,
) {
    for symbol in module.symbols.values_mut() {
        let rewritten: HashSet<String> = symbol
            .dependencies
            .iter()
            .map(|dep| {
                if moving_symbols.contains(dep) {
                    // Replace the old target prefix with the new one.
                    if let Some(suffix) = dep.strip_prefix(old_prefix) {
                        format!("{new_prefix}{suffix}")
                    } else {
                        dep.clone()
                    }
                } else {
                    dep.clone()
                }
            })
            .collect();
        symbol.dependencies = rewritten;
    }
    for submod in module.submodules.values_mut() {
        rewrite_dependencies(submod, moving_symbols, old_prefix, new_prefix);
    }
}

#[cfg(test)]
mod tests {
    use tarjanize_schemas::{SymbolKind, Visibility};

    use super::*;
    use crate::target_graph;

    /// Builds a test `SymbolGraph` with a single package "test-pkg"
    /// and a single target "lib".
    ///
    /// The target has two submodules ("alpha" and "beta"), each with
    /// one symbol. `alpha::Foo` depends on `beta::Bar`, but not vice
    /// versa, producing two distinct SCCs so we can split one out.
    fn test_symbol_graph() -> SymbolGraph {
        let prefix = "[test-pkg/lib]::";

        // beta::Bar — no dependencies, costs 5ms.
        let bar = Symbol {
            file: "beta.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), 5.0)]),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Struct".to_string(),
                visibility: Visibility::Public,
            },
        };

        // alpha::Foo — depends on beta::Bar, costs 10ms.
        let foo = Symbol {
            file: "alpha.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), 10.0)]),
            dependencies: HashSet::from([format!("{prefix}beta::Bar")]),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        };

        let alpha_mod = Module {
            symbols: HashMap::from([("Foo".to_string(), foo)]),
            submodules: HashMap::new(),
        };

        let beta_mod = Module {
            symbols: HashMap::from([("Bar".to_string(), bar)]),
            submodules: HashMap::new(),
        };

        let root = Module {
            symbols: HashMap::new(),
            submodules: HashMap::from([
                ("alpha".to_string(), alpha_mod),
                ("beta".to_string(), beta_mod),
            ]),
        };

        let target = Target {
            timings: TargetTimings {
                wall_time: Duration::from_secs_f64(0.020),
                event_times_ms: HashMap::from([(
                    "metadata_decode_foo".to_string(),
                    3.0,
                )]),
            },
            dependencies: HashSet::from(["other-dep/lib".to_string()]),
            root,
        };

        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("test-pkg".to_string(), Package { targets });
        SymbolGraph { packages }
    }

    #[test]
    fn export_with_no_splits_returns_original() {
        let sg = test_symbol_graph();
        let result = export_symbol_graph(&sg, &[]);
        // No splits means the result should be structurally identical.
        assert_eq!(result.packages.len(), sg.packages.len());
        assert!(result.packages.contains_key("test-pkg"));
        let target = &result.packages["test-pkg"].targets["lib"];
        // All symbols should still be present.
        assert!(target.root.submodules.contains_key("alpha"));
        assert!(target.root.submodules.contains_key("beta"));
    }

    #[test]
    fn export_with_split_adds_new_package() {
        let sg = test_symbol_graph();

        // Condense to find the SCC containing beta::Bar (the leaf).
        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-split".to_string(),
            selected_sccs: vec![bar_scc.id],
        }];

        let result = export_symbol_graph(&sg, &ops);

        // A new package should exist for the split crate.
        assert!(
            result.packages.contains_key("test-pkg-split"),
            "new package must be created"
        );
        assert_eq!(
            result.packages.len(),
            2,
            "should have original + new package"
        );
        // The new package should have a "lib" target.
        assert!(
            result.packages["test-pkg-split"]
                .targets
                .contains_key("lib")
        );
    }

    #[test]
    fn export_split_removes_symbols_from_source() {
        let sg = test_symbol_graph();

        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-split".to_string(),
            selected_sccs: vec![bar_scc.id],
        }];

        let result = export_symbol_graph(&sg, &ops);

        // Bar should be gone from the source target.
        let source = &result.packages["test-pkg"].targets["lib"];
        let beta_has_bar = source
            .root
            .submodules
            .get("beta")
            .is_some_and(|m| m.symbols.contains_key("Bar"));
        assert!(!beta_has_bar, "Bar must be removed from source target");

        // Bar should appear in the new target.
        let new_target = &result.packages["test-pkg-split"].targets["lib"];
        let new_has_bar = new_target
            .root
            .submodules
            .get("beta")
            .is_some_and(|m| m.symbols.contains_key("Bar"));
        assert!(new_has_bar, "Bar must appear in new target");

        // Foo should remain in the source.
        let alpha_has_foo = source
            .root
            .submodules
            .get("alpha")
            .is_some_and(|m| m.symbols.contains_key("Foo"));
        assert!(alpha_has_foo, "Foo must remain in source target");
    }

    #[test]
    fn export_split_adds_dependency_from_source_to_new() {
        let sg = test_symbol_graph();

        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-split".to_string(),
            selected_sccs: vec![bar_scc.id],
        }];

        let result = export_symbol_graph(&sg, &ops);

        // The residual (source) should now depend on the new crate.
        let source = &result.packages["test-pkg"].targets["lib"];
        assert!(
            source.dependencies.contains("test-pkg-split/lib"),
            "residual must depend on new crate"
        );
    }

    #[test]
    fn export_split_copies_external_deps_to_new_target() {
        let sg = test_symbol_graph();

        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-split".to_string(),
            selected_sccs: vec![bar_scc.id],
        }];

        let result = export_symbol_graph(&sg, &ops);

        // The new target should inherit the source's external
        // dependencies.
        let new_target = &result.packages["test-pkg-split"].targets["lib"];
        assert!(
            new_target.dependencies.contains("other-dep/lib"),
            "new target must inherit external deps"
        );
    }

    #[test]
    fn export_split_adjusts_wall_time() {
        let sg = test_symbol_graph();

        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-split".to_string(),
            selected_sccs: vec![bar_scc.id],
        }];

        let result = export_symbol_graph(&sg, &ops);

        // Bar has 5ms of attributed cost. The new target should have
        // wall_time equal to the moved cost.
        let new_target = &result.packages["test-pkg-split"].targets["lib"];
        assert_eq!(
            new_target.timings.wall_time,
            Duration::from_secs_f64(0.005),
            "new target wall_time should be moved cost (5ms)"
        );

        // The residual should have reduced wall_time: 20 - 5 = 15ms.
        let source = &result.packages["test-pkg"].targets["lib"];
        assert_eq!(
            source.timings.wall_time,
            Duration::from_secs_f64(0.015),
            "residual wall_time should be original minus moved"
        );
    }

    #[test]
    fn export_split_rewrites_dependencies() {
        let sg = test_symbol_graph();

        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-split".to_string(),
            selected_sccs: vec![bar_scc.id],
        }];

        let result = export_symbol_graph(&sg, &ops);

        // alpha::Foo originally depended on [test-pkg/lib]::beta::Bar.
        // After the split, Bar is in test-pkg-split, so Foo's
        // dependency should be rewritten to
        // [test-pkg-split/lib]::beta::Bar.
        let foo = &result.packages["test-pkg"].targets["lib"]
            .root
            .submodules["alpha"]
            .symbols["Foo"];
        let expected_dep = "[test-pkg-split/lib]::beta::Bar".to_string();
        assert!(
            foo.dependencies.contains(&expected_dep),
            "Foo's dependency on Bar should be rewritten to \
             new target prefix. Got: {:?}",
            foo.dependencies
        );
    }

    #[test]
    fn export_downset_expansion_moves_transitive_deps() {
        // Selecting alpha::Foo (which depends on beta::Bar) should
        // also move beta::Bar via downset expansion.
        let sg = test_symbol_graph();

        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let foo_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Foo")))
            .expect("Foo SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-all".to_string(),
            selected_sccs: vec![foo_scc.id],
        }];

        let result = export_symbol_graph(&sg, &ops);

        // Both Foo and Bar should be in the new target (Bar is a
        // transitive dependency).
        let new_target = &result.packages["test-pkg-all"].targets["lib"];
        let has_foo = new_target
            .root
            .submodules
            .get("alpha")
            .is_some_and(|m| m.symbols.contains_key("Foo"));
        let has_bar = new_target
            .root
            .submodules
            .get("beta")
            .is_some_and(|m| m.symbols.contains_key("Bar"));
        assert!(has_foo, "Foo should be in new target");
        assert!(has_bar, "Bar should be in new target (downset expansion)");

        // Source should be empty of both.
        let source = &result.packages["test-pkg"].targets["lib"];
        let src_has_foo = source
            .root
            .submodules
            .get("alpha")
            .is_some_and(|m| m.symbols.contains_key("Foo"));
        let src_has_bar = source
            .root
            .submodules
            .get("beta")
            .is_some_and(|m| m.symbols.contains_key("Bar"));
        assert!(!src_has_foo, "Foo should be removed from source");
        assert!(!src_has_bar, "Bar should be removed from source");
    }

    #[test]
    fn export_invalid_target_is_skipped() {
        let sg = test_symbol_graph();

        let ops = vec![SplitOperation {
            source_target: "nonexistent/lib".to_string(),
            new_crate_name: "new-pkg".to_string(),
            selected_sccs: vec![0],
        }];

        let result = export_symbol_graph(&sg, &ops);

        // Should return the original graph unchanged.
        assert_eq!(result.packages.len(), 1);
        assert!(result.packages.contains_key("test-pkg"));
        assert!(!result.packages.contains_key("new-pkg"));
    }

    #[test]
    fn parse_symbol_path_basic() {
        let prefix = "[test-pkg/lib]::";
        let path = "[test-pkg/lib]::alpha::Foo";
        let (mods, name) = parse_symbol_path(path, prefix).unwrap();
        assert_eq!(mods, vec!["alpha"]);
        assert_eq!(name, "Foo");
    }

    #[test]
    fn parse_symbol_path_root_level() {
        let prefix = "[test-pkg/lib]::";
        let path = "[test-pkg/lib]::RootSym";
        let (mods, name) = parse_symbol_path(path, prefix).unwrap();
        assert!(mods.is_empty());
        assert_eq!(name, "RootSym");
    }

    #[test]
    fn parse_symbol_path_nested() {
        let prefix = "[test-pkg/lib]::";
        let path = "[test-pkg/lib]::a::b::c::Deep";
        let (mods, name) = parse_symbol_path(path, prefix).unwrap();
        assert_eq!(mods, vec!["a", "b", "c"]);
        assert_eq!(name, "Deep");
    }

    #[test]
    fn parse_symbol_path_wrong_prefix() {
        let prefix = "[other-pkg/lib]::";
        let path = "[test-pkg/lib]::alpha::Foo";
        assert!(parse_symbol_path(path, prefix).is_none());
    }

    // =================================================================
    // Tests derived from design doc behavioral specifications
    // =================================================================

    /// Design doc: "The output is a complete symbol graph with the
    /// split targets, ready for further visualization or analysis."
    ///
    /// Verify the exported `SymbolGraph` can be fed through
    /// `build_target_graph()` + `compute_schedule()` to produce a
    /// valid schedule. This is the "round-trip" property.
    #[test]
    fn export_round_trip_produces_valid_schedule() {
        let sg = test_symbol_graph();

        // Apply a split: move beta::Bar to a new crate.
        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-split".to_string(),
            selected_sccs: vec![bar_scc.id],
        }];

        let exported = export_symbol_graph(&sg, &ops);

        // Round-trip: build a target graph and schedule from the
        // exported symbol graph. This should not panic and should
        // produce a valid schedule.
        let tg = crate::build_target_graph(&exported, None);
        let schedule = crate::schedule::compute_schedule(&tg);

        // The exported graph has 2 packages, so the schedule should
        // have at least 2 targets.
        assert!(
            schedule.targets.len() >= 2,
            "schedule should have at least 2 targets, got {}",
            schedule.targets.len()
        );

        // The critical path should be non-empty (at least one target).
        assert!(
            !schedule.critical_path.is_empty(),
            "critical path should not be empty"
        );

        // Summary statistics should be positive.
        assert!(
            !schedule.summary.critical_path.is_zero(),
            "critical path time should be positive"
        );
        assert!(
            !schedule.summary.total_cost.is_zero(),
            "total cost should be positive"
        );
    }

    /// Design doc: "Split targets appear as new targets in the output;
    /// their symbols, dependencies, and cost predictors are fully
    /// resolved."
    ///
    /// Verify that total symbol count is preserved: no symbols are
    /// lost or duplicated during export.
    #[test]
    fn export_preserves_total_symbol_count() {
        let sg = test_symbol_graph();

        // Count total symbols in the original.
        let original_count = count_symbols_recursive(&sg);

        // Apply a split.
        let intra = target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "test-pkg/lib".to_string(),
            new_crate_name: "test-pkg-split".to_string(),
            selected_sccs: vec![bar_scc.id],
        }];

        let exported = export_symbol_graph(&sg, &ops);

        // Count total symbols in the exported graph.
        let exported_count = count_symbols_recursive(&exported);

        assert_eq!(
            original_count, exported_count,
            "total symbol count should be preserved: original={original_count}, exported={exported_count}"
        );
    }

    /// Helper: recursively count all symbols across all targets.
    fn count_symbols_recursive(sg: &SymbolGraph) -> usize {
        fn count_in_module(module: &Module) -> usize {
            module.symbols.len()
                + module
                    .submodules
                    .values()
                    .map(count_in_module)
                    .sum::<usize>()
        }

        sg.packages
            .values()
            .flat_map(|pkg| pkg.targets.values())
            .map(|target| count_in_module(&target.root))
            .sum()
    }

    /// Design doc: exports should handle multiple splits to different
    /// targets correctly.
    ///
    /// Build a graph with two packages, split from each, verify both
    /// splits appear in the output.
    #[test]
    fn export_multiple_splits_different_targets() {
        // Build a graph with two packages.
        let mut sg = test_symbol_graph();

        // Add a second package "pkg-b" with one target.
        let sym_x = Symbol {
            file: "x.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), 7.0)]),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        };
        let sym_y = Symbol {
            file: "y.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), 3.0)]),
            dependencies: HashSet::from(["[pkg-b/lib]::X".to_string()]),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        };
        let root_b = Module {
            symbols: HashMap::from([
                ("X".to_string(), sym_x),
                ("Y".to_string(), sym_y),
            ]),
            submodules: HashMap::new(),
        };
        let target_b = Target {
            timings: TargetTimings {
                wall_time: Duration::from_secs_f64(0.010),
                event_times_ms: HashMap::new(),
            },
            dependencies: HashSet::new(),
            root: root_b,
        };
        let mut pkg_b_targets = HashMap::new();
        pkg_b_targets.insert("lib".to_string(), target_b);
        sg.packages.insert(
            "pkg-b".to_string(),
            Package {
                targets: pkg_b_targets,
            },
        );

        // Condense both targets and pick SCC IDs.
        let intra_a =
            target_graph::condense_target(&sg, "test-pkg/lib").unwrap();
        let bar_scc = intra_a
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("Bar")))
            .expect("Bar SCC");

        let intra_b = target_graph::condense_target(&sg, "pkg-b/lib").unwrap();
        let x_scc = intra_b
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains('X')))
            .expect("X SCC");

        let ops = vec![
            SplitOperation {
                source_target: "test-pkg/lib".to_string(),
                new_crate_name: "test-pkg-beta".to_string(),
                selected_sccs: vec![bar_scc.id],
            },
            SplitOperation {
                source_target: "pkg-b/lib".to_string(),
                new_crate_name: "pkg-b-core".to_string(),
                selected_sccs: vec![x_scc.id],
            },
        ];

        let exported = export_symbol_graph(&sg, &ops);

        // Should have 4 packages: test-pkg, test-pkg-beta, pkg-b, pkg-b-core.
        assert_eq!(
            exported.packages.len(),
            4,
            "should have 4 packages, got: {:?}",
            exported.packages.keys().collect::<Vec<_>>()
        );
        assert!(exported.packages.contains_key("test-pkg-beta"));
        assert!(exported.packages.contains_key("pkg-b-core"));

        // Verify symbol count is preserved.
        let original_count = count_symbols_recursive(&sg);
        let exported_count = count_symbols_recursive(&exported);
        assert_eq!(original_count, exported_count);
    }
}
