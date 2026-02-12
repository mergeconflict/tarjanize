//! Intra-target SCC condensation for the split explorer.
//!
//! Computes the SCC DAG within a single target, collapsing symbols into
//! strongly connected components. This is the graph the user sees when
//! drilling down into a target for splitting.

use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use tarjanize_schemas::{
    Module, SymbolGraph, SymbolKind, serde_duration, sum_event_times,
};

/// An SCC node in the intra-target condensation graph.
///
/// Each node represents one strongly connected component — a group of
/// symbols that mutually depend on each other and cannot be split apart.
/// Singleton SCCs (no internal cycles) are the common case; multi-symbol
/// SCCs indicate tight coupling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SccNode {
    /// Unique index of this SCC within the target.
    pub id: usize,
    /// Symbol paths belonging to this SCC (full `[pkg/target]::module::symbol` paths).
    pub symbols: Vec<String>,
    /// Module path for the primary symbol (for visual clustering).
    pub module_path: String,
    /// Total attributed cost (sum of symbol `event_times_ms`).
    #[serde(rename = "cost_ms", with = "serde_duration")]
    pub cost: Duration,
}

/// The intra-target SCC DAG, ready for frontend rendering.
///
/// Contains all SCC nodes, the edges between them, and module metadata
/// for clustering. The frontend uses this to render the drill-down view
/// when a user clicks on a target in the Gantt chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntraTargetGraph {
    /// SCC nodes, indexed by `SccNode.id`.
    pub nodes: Vec<SccNode>,
    /// Edges between SCCs: `(from_id, to_id)`. Direction: dependency -> dependent.
    pub edges: Vec<(usize, usize)>,
    /// Unique module paths found in this target, for clustering metadata.
    pub modules: Vec<String>,
}

/// Computes the SCC DAG for a single target in the symbol graph.
///
/// 1. Parses `target_id` as `{package}/{target}` to look up the target
/// 2. Walks the module tree, collecting all symbols with their full paths,
///    module paths, and costs. Filters out `Use` and `ExternCrate` symbols
///    (`SymbolKind::ModuleDef` where kind is `"Use"` or `"ExternCrate"`)
/// 3. Builds a `petgraph::DiGraph` of symbol dependencies (only intra-target
///    edges — skips dependencies that reference other targets)
/// 4. Runs `petgraph::algo::condensation()` to compute SCCs
/// 5. Builds `SccNode` for each SCC with symbols, costs, module paths
/// 6. Extracts inter-SCC edges (deduplicated)
/// 7. Returns `IntraTargetGraph`
///
/// Returns `None` if the target doesn't exist in the symbol graph.
pub fn condense_target(
    symbol_graph: &SymbolGraph,
    target_id: &str,
) -> Option<IntraTargetGraph> {
    // Step 1: Parse target_id as "{package}/{target}" and look up the target.
    let (package_name, target_key) = target_id.split_once('/')?;
    let target = symbol_graph
        .packages
        .get(package_name)?
        .targets
        .get(target_key)?;

    // Step 2: Walk the module tree, collecting all symbols with their full
    // paths, module paths, and costs. Filter out Use and ExternCrate items
    // because they are re-exports/imports, not real compilable symbols.
    let target_prefix = format!("[{target_id}]::");
    let mut symbol_paths: Vec<String> = Vec::new();
    let mut module_paths: Vec<String> = Vec::new();
    let mut costs: Vec<f64> = Vec::new();

    collect_symbols(
        &target.root,
        &target_prefix,
        "",
        &mut symbol_paths,
        &mut module_paths,
        &mut costs,
    );

    // Build a path -> index lookup for resolving dependencies.
    let path_to_idx: HashMap<&str, usize> = symbol_paths
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect();

    // Step 3: Build a petgraph DiGraph of symbol dependencies. Only include
    // intra-target edges (deps starting with the same target prefix).
    let n = symbol_paths.len();
    let mut graph = DiGraph::<usize, ()>::with_capacity(n, 0);
    let nodes: Vec<NodeIndex> = (0..n).map(|i| graph.add_node(i)).collect();

    // We need the original Symbol objects to read dependencies, so walk
    // the module tree again. This second walk is cheap for typical targets.
    let deps = collect_dependencies(&target.root, &target_prefix, "");
    for (sym_path, dep_paths) in &deps {
        if let Some(&from_idx) = path_to_idx.get(sym_path.as_str()) {
            for dep_path in dep_paths {
                if let Some(&to_idx) = path_to_idx.get(dep_path.as_str()) {
                    // Edge direction: dependency -> dependent (to -> from).
                    graph.add_edge(nodes[to_idx], nodes[from_idx], ());
                }
            }
        }
    }

    // Step 3b: Add synthetic back-edges for anchor constraints so that
    // condensation merges each impl with at least one of its anchors,
    // preserving the orphan rule. See `add_anchor_back_edges` for details.
    let anchors = collect_impl_anchors(&target.root, &target_prefix, "");
    add_anchor_back_edges(&anchors, &path_to_idx, &nodes, &mut graph);

    // Step 4: Run petgraph condensation to compute SCCs. The `make_acyclic`
    // parameter is true, which means the result is a DAG of SCCs.
    let condensed = petgraph::algo::condensation(graph, true);

    // Step 5: Build SccNode for each SCC. The condensation graph's node
    // weights are Vec<usize> (the original node indices in each SCC).
    // petgraph's condensation reverses topological order, but we just
    // need consistent IDs.
    let mut scc_nodes: Vec<SccNode> =
        Vec::with_capacity(condensed.node_count());
    let mut all_modules: BTreeSet<String> = BTreeSet::new();

    // Map from original symbol index to SCC index for edge extraction.
    let mut sym_to_scc: Vec<usize> = vec![0; n];

    for (scc_id, scc_node_idx) in condensed.node_indices().enumerate() {
        let members = &condensed[scc_node_idx];
        let mut syms: Vec<String> = Vec::new();
        let mut total_cost = 0.0;
        let mut primary_module = String::new();

        for &orig_idx in members {
            syms.push(symbol_paths[orig_idx].clone());
            total_cost += costs[orig_idx];
            sym_to_scc[orig_idx] = scc_id;

            // Track all module paths for the modules list.
            all_modules.insert(module_paths[orig_idx].clone());

            // Use the first symbol's module path as the primary.
            if primary_module.is_empty() {
                primary_module.clone_from(&module_paths[orig_idx]);
            }
        }

        syms.sort();
        scc_nodes.push(SccNode {
            id: scc_id,
            symbols: syms,
            module_path: primary_module,
            cost: Duration::from_secs_f64(total_cost / 1000.0),
        });
    }

    // Step 6: Extract inter-SCC edges (deduplicated). The condensed graph
    // already has the correct edges between SCCs, but with petgraph's
    // internal node indices. Map them to our SCC IDs.
    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for edge in condensed.edge_indices() {
        let (src, dst) = condensed
            .edge_endpoints(edge)
            .expect("edge index from edge_indices() must be valid");
        let from_scc = condensed
            .node_indices()
            .position(|n| n == src)
            .expect("source node must exist");
        let to_scc = condensed
            .node_indices()
            .position(|n| n == dst)
            .expect("target node must exist");
        edge_set.insert((from_scc, to_scc));
    }

    // Step 7: Return IntraTargetGraph.
    Some(IntraTargetGraph {
        nodes: scc_nodes,
        edges: edge_set.into_iter().collect(),
        modules: all_modules.into_iter().collect(),
    })
}

/// Adds synthetic back-edges from each impl to its best intra-target anchor.
///
/// For each impl block, this finds the anchor with minimum in-degree (a rough
/// proxy for "most niche" — fewer dependents means less blast radius from the
/// forced merge) and adds an edge `impl -> anchor`. Combined with the existing
/// forward edge `anchor -> impl` (from the dependency graph, where edge
/// direction is `dependency -> dependent`), this creates a cycle that
/// `petgraph::algo::condensation()` collapses into a single SCC.
///
/// Without these back-edges, a threshold cut could place an impl in one
/// partition and all its anchors in another, violating the orphan rule.
///
/// This mirrors step 2b in `tarjanize-condense/src/scc.rs`.
fn add_anchor_back_edges(
    impl_anchors: &[(String, Vec<String>)],
    path_to_idx: &HashMap<&str, usize>,
    nodes: &[NodeIndex],
    graph: &mut DiGraph<usize, ()>,
) {
    for (impl_path, anchor_paths) in impl_anchors {
        let Some(&impl_idx) = path_to_idx.get(impl_path.as_str()) else {
            continue;
        };

        // Resolve anchor paths to graph indices, keeping only those
        // that exist in this target's symbol set.
        let valid_anchor_indices: Vec<usize> = anchor_paths
            .iter()
            .filter_map(|p| path_to_idx.get(p.as_str()).copied())
            .collect();

        if valid_anchor_indices.is_empty() {
            continue;
        }

        // Pick the anchor with minimum in-degree (fewest incoming edges).
        let best_anchor_idx = valid_anchor_indices
            .into_iter()
            .min_by_key(|&idx| {
                graph
                    .neighbors_directed(
                        nodes[idx],
                        petgraph::Direction::Incoming,
                    )
                    .count()
            })
            .expect("valid_anchor_indices is non-empty");

        // Back-edge: impl -> anchor. In this graph, edges go from
        // dependency -> dependent, so the existing forward edge for
        // "impl depends on Foo" is Foo -> impl. Adding impl -> Foo
        // closes the cycle: Foo -> impl -> Foo.
        graph.add_edge(nodes[impl_idx], nodes[best_anchor_idx], ());
    }
}

/// Recursively collects impl anchor relationships from the module tree.
///
/// For each `SymbolKind::Impl` symbol, returns `(impl_path, anchor_paths)`
/// where `anchor_paths` are the workspace-local types/traits that satisfy
/// the orphan rule. Only anchors within the same target (starting with
/// `target_prefix`) are included — external anchors are irrelevant for
/// intra-target SCC computation.
///
/// This mirrors the pattern of `collect_symbols()` and
/// `collect_dependencies()`, walking the same module tree recursively.
fn collect_impl_anchors(
    module: &Module,
    target_prefix: &str,
    module_path: &str,
) -> Vec<(String, Vec<String>)> {
    let mut result = Vec::new();

    for (name, symbol) in &module.symbols {
        if let SymbolKind::Impl { ref anchors, .. } = symbol.kind {
            let full_path = if module_path.is_empty() {
                format!("{target_prefix}{name}")
            } else {
                format!("{target_prefix}{module_path}::{name}")
            };

            // Only keep anchors within this target — external anchors
            // cannot participate in intra-target SCC merging.
            let intra_anchors: Vec<String> = anchors
                .iter()
                .filter(|a| a.starts_with(target_prefix))
                .cloned()
                .collect();

            if !intra_anchors.is_empty() {
                result.push((full_path, intra_anchors));
            }
        }
    }

    for (submod_name, submod) in &module.submodules {
        let child_path = if module_path.is_empty() {
            submod_name.clone()
        } else {
            format!("{module_path}::{submod_name}")
        };
        result.extend(collect_impl_anchors(submod, target_prefix, &child_path));
    }

    result
}

/// Recursively collects symbols from the module tree, filtering out
/// `Use` and `ExternCrate` items.
///
/// Builds the full symbol path (`[pkg/target]::module::symbol`), the
/// module path (just the module portion), and the total cost for each
/// symbol.
fn collect_symbols(
    module: &Module,
    target_prefix: &str,
    module_path: &str,
    symbol_paths: &mut Vec<String>,
    module_paths_out: &mut Vec<String>,
    costs: &mut Vec<f64>,
) {
    for (name, symbol) in &module.symbols {
        // Filter out Use and ExternCrate items — they're re-exports/imports,
        // not actual compilable symbols that contribute to build time.
        if let SymbolKind::ModuleDef { ref kind, .. } = symbol.kind
            && (kind == "Use" || kind == "ExternCrate")
        {
            continue;
        }

        let full_path = if module_path.is_empty() {
            format!("{target_prefix}{name}")
        } else {
            format!("{target_prefix}{module_path}::{name}")
        };
        symbol_paths.push(full_path);
        module_paths_out.push(module_path.to_string());
        costs.push(sum_event_times(&symbol.event_times_ms));
    }

    for (submod_name, submod) in &module.submodules {
        let child_path = if module_path.is_empty() {
            submod_name.clone()
        } else {
            format!("{module_path}::{submod_name}")
        };
        collect_symbols(
            submod,
            target_prefix,
            &child_path,
            symbol_paths,
            module_paths_out,
            costs,
        );
    }
}

/// Recursively collects symbol dependencies from the module tree.
///
/// Returns a list of `(symbol_path, intra_target_deps)` pairs. Only
/// dependencies that start with the target prefix are included (cross-target
/// deps are irrelevant for intra-target SCC computation).
fn collect_dependencies(
    module: &Module,
    target_prefix: &str,
    module_path: &str,
) -> Vec<(String, Vec<String>)> {
    let mut result = Vec::new();

    for (name, symbol) in &module.symbols {
        // Skip Use/ExternCrate — same filtering as collect_symbols.
        if let SymbolKind::ModuleDef { ref kind, .. } = symbol.kind
            && (kind == "Use" || kind == "ExternCrate")
        {
            continue;
        }

        let full_path = if module_path.is_empty() {
            format!("{target_prefix}{name}")
        } else {
            format!("{target_prefix}{module_path}::{name}")
        };

        // Only keep intra-target dependencies.
        let intra_deps: Vec<String> = symbol
            .dependencies
            .iter()
            .filter(|dep| dep.starts_with(target_prefix))
            .cloned()
            .collect();

        result.push((full_path, intra_deps));
    }

    for (submod_name, submod) in &module.submodules {
        let child_path = if module_path.is_empty() {
            submod_name.clone()
        } else {
            format!("{module_path}::{submod_name}")
        };
        result.extend(collect_dependencies(submod, target_prefix, &child_path));
    }

    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use tarjanize_schemas::*;

    use super::*;

    #[test]
    fn condense_target_returns_none_for_missing_target() {
        let sg = SymbolGraph::default();
        assert!(condense_target(&sg, "nonexistent/lib").is_none());
    }

    #[test]
    fn condense_target_filters_use_items() {
        let mut symbols = HashMap::new();
        symbols.insert(
            "real_fn".to_string(),
            Symbol {
                file: "lib.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 10.0)]),
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        symbols.insert(
            "use_item".to_string(),
            Symbol {
                file: "lib.rs".to_string(),
                event_times_ms: HashMap::new(),
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Use".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );

        let root = Module {
            symbols,
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root,
        };

        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("test-pkg".to_string(), Package { targets });
        let sg = SymbolGraph { packages };

        let result = condense_target(&sg, "test-pkg/lib").unwrap();
        // Only the real_fn should appear, not the Use item.
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].symbols.len(), 1);
        assert!(result.nodes[0].symbols[0].contains("real_fn"));
    }

    /// Helper: builds a `SymbolGraph` with a single target `test-pkg/lib`
    /// from a list of `(name, kind_str, cost, deps)` tuples.
    fn make_target(syms: &[(&str, &str, f64, &[&str])]) -> SymbolGraph {
        make_target_in_module(syms, &[])
    }

    /// Helper: builds a `SymbolGraph` with a single target `test-pkg/lib`
    /// from symbols in the root module and symbols in named submodules.
    /// Submodule symbols are given as `(module_name, name, kind, cost, deps)`.
    fn make_target_in_module(
        root_syms: &[(&str, &str, f64, &[&str])],
        sub_syms: &[(&str, &str, &str, f64, &[&str])],
    ) -> SymbolGraph {
        let prefix = "[test-pkg/lib]::";

        let mut symbols = HashMap::new();
        for &(name, kind_str, cost, deps) in root_syms {
            let dep_set: HashSet<String> =
                deps.iter().map(|d| format!("{prefix}{d}")).collect();
            let event_times = if cost > 0.0 {
                HashMap::from([("typeck".to_string(), cost)])
            } else {
                HashMap::new()
            };
            symbols.insert(
                name.to_string(),
                Symbol {
                    file: "lib.rs".to_string(),
                    event_times_ms: event_times,
                    dependencies: dep_set,
                    kind: SymbolKind::ModuleDef {
                        kind: kind_str.to_string(),
                        visibility: Visibility::Public,
                    },
                },
            );
        }

        let mut submodules = HashMap::new();
        for &(mod_name, name, kind_str, cost, deps) in sub_syms {
            let dep_set: HashSet<String> =
                deps.iter().map(|d| format!("{prefix}{d}")).collect();
            let event_times = if cost > 0.0 {
                HashMap::from([("typeck".to_string(), cost)])
            } else {
                HashMap::new()
            };
            let submod =
                submodules.entry(mod_name.to_string()).or_insert_with(|| {
                    Module {
                        symbols: HashMap::new(),
                        submodules: HashMap::new(),
                    }
                });
            submod.symbols.insert(
                name.to_string(),
                Symbol {
                    file: format!("{mod_name}.rs"),
                    event_times_ms: event_times,
                    dependencies: dep_set,
                    kind: SymbolKind::ModuleDef {
                        kind: kind_str.to_string(),
                        visibility: Visibility::Public,
                    },
                },
            );
        }

        let root = Module {
            symbols,
            submodules,
        };
        let target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root,
        };

        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("test-pkg".to_string(), Package { targets });
        SymbolGraph { packages }
    }

    #[test]
    fn condense_target_merges_cyclic_symbols() {
        // sym_a depends on sym_b, sym_b depends on sym_a — mutual cycle.
        let sg = make_target(&[
            ("sym_a", "Function", 5.0, &["sym_b"]),
            ("sym_b", "Function", 3.0, &["sym_a"]),
        ]);

        let result = condense_target(&sg, "test-pkg/lib").unwrap();
        // Both symbols should be merged into a single SCC.
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].symbols.len(), 2);
        assert!(result.edges.is_empty());
        // Cost should be the sum of both symbols.
        assert_eq!(result.nodes[0].cost, Duration::from_millis(8));
    }

    #[test]
    fn condense_target_chain_produces_two_sccs_with_edge() {
        // A depends on B, no cycle. Should produce two separate SCCs
        // with one edge from B's SCC to A's SCC.
        let sg = make_target(&[
            ("fn_a", "Function", 10.0, &["fn_b"]),
            ("fn_b", "Function", 20.0, &[]),
        ]);

        let result = condense_target(&sg, "test-pkg/lib").unwrap();
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 1);

        // Each SCC should have exactly one symbol.
        for node in &result.nodes {
            assert_eq!(node.symbols.len(), 1);
        }
    }

    #[test]
    fn condense_target_module_path_annotation() {
        // Symbols in different submodules should have their module paths
        // correctly reflected in SccNode.module_path.
        let sg = make_target_in_module(
            &[("root_fn", "Function", 1.0, &[])],
            &[
                ("parser", "parse_item", "Function", 2.0, &[]),
                ("codegen", "emit_code", "Function", 3.0, &[]),
            ],
        );

        let result = condense_target(&sg, "test-pkg/lib").unwrap();
        // Three independent symbols → three SCCs.
        assert_eq!(result.nodes.len(), 3);

        // Collect all module paths from the SCC nodes.
        let mod_paths: HashSet<&str> = result
            .nodes
            .iter()
            .map(|n| n.module_path.as_str())
            .collect();
        // Should have root (""), "parser", and "codegen".
        assert!(mod_paths.contains(""), "root module path missing");
        assert!(mod_paths.contains("parser"), "parser module path missing");
        assert!(mod_paths.contains("codegen"), "codegen module path missing");

        // The modules list should contain all unique module paths.
        assert!(result.modules.contains(&String::new()));
        assert!(result.modules.contains(&"parser".to_string()));
        assert!(result.modules.contains(&"codegen".to_string()));
    }

    #[test]
    fn condense_target_filters_extern_crate_items() {
        // ExternCrate items should be filtered out just like Use items.
        let sg = make_target(&[
            ("real_fn", "Function", 5.0, &[]),
            ("ext_crate", "ExternCrate", 0.0, &[]),
        ]);

        let result = condense_target(&sg, "test-pkg/lib").unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].symbols.len(), 1);
        assert!(result.nodes[0].symbols[0].contains("real_fn"));
    }

    #[test]
    fn condense_target_cross_target_deps_ignored() {
        // A dependency on a symbol in another target should not create
        // an edge in the intra-target graph.
        let other_prefix = "[other-pkg/lib]::";

        let mut symbols = HashMap::new();
        symbols.insert(
            "local_fn".to_string(),
            Symbol {
                file: "lib.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 5.0)]),
                dependencies: HashSet::from([format!(
                    "{other_prefix}external_fn"
                )]),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );

        let root = Module {
            symbols,
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root,
        };
        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("test-pkg".to_string(), Package { targets });
        let sg = SymbolGraph { packages };

        let result = condense_target(&sg, "test-pkg/lib").unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert!(result.edges.is_empty());
    }

    #[test]
    fn condense_target_merges_impl_with_anchor() {
        // A struct "Foo" and an impl block "{{impl}}[0]" with anchor pointing
        // to Foo. The anchor back-edge should create a cycle, merging them
        // into a single SCC even without an explicit dependency edge.
        let prefix = "[test-pkg/lib]::";

        let mut symbols = HashMap::new();
        symbols.insert(
            "Foo".to_string(),
            Symbol {
                file: "lib.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 5.0)]),
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Struct".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        symbols.insert(
            "{{impl}}[0]".to_string(),
            Symbol {
                file: "lib.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 3.0)]),
                // The impl depends on Foo (normal forward edge).
                dependencies: HashSet::from([format!("{prefix}Foo")]),
                kind: SymbolKind::Impl {
                    name: "impl Foo".to_string(),
                    anchors: HashSet::from([format!("{prefix}Foo")]),
                },
            },
        );

        let root = Module {
            symbols,
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root,
        };
        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("test-pkg".to_string(), Package { targets });
        let sg = SymbolGraph { packages };

        let result = condense_target(&sg, "test-pkg/lib").unwrap();

        // Both Foo and {{impl}}[0] must be in the same SCC because the
        // anchor back-edge (Foo → {{impl}}[0]) creates a cycle with the
        // forward dependency edge ({{impl}}[0] → Foo via dep resolution
        // which becomes Foo → {{impl}}[0] in the graph since edge
        // direction is dependency → dependent).
        assert_eq!(
            result.nodes.len(),
            1,
            "Struct and its impl should be merged into one SCC"
        );
        assert_eq!(result.nodes[0].symbols.len(), 2);
        assert_eq!(result.nodes[0].cost, Duration::from_millis(8));
    }

    #[test]
    fn condense_target_external_anchor_no_back_edge() {
        // An impl with two anchors: one external (different target) and one
        // intra-target. Only the intra-target anchor should cause merging.
        // The external anchor is ignored because it's not in this target.
        let prefix = "[test-pkg/lib]::";

        let mut symbols = HashMap::new();
        symbols.insert(
            "LocalType".to_string(),
            Symbol {
                file: "lib.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 4.0)]),
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Struct".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        symbols.insert(
            "{{impl}}[0]".to_string(),
            Symbol {
                file: "lib.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 2.0)]),
                // The impl depends on LocalType.
                dependencies: HashSet::from([format!("{prefix}LocalType")]),
                kind: SymbolKind::Impl {
                    name: "impl ExternalTrait for LocalType".to_string(),
                    anchors: HashSet::from([
                        // External anchor — should be ignored (different target).
                        "[other-pkg/lib]::ExternalTrait".to_string(),
                        // Intra-target anchor — should cause merging.
                        format!("{prefix}LocalType"),
                    ]),
                },
            },
        );

        let root = Module {
            symbols,
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root,
        };
        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("test-pkg".to_string(), Package { targets });
        let sg = SymbolGraph { packages };

        let result = condense_target(&sg, "test-pkg/lib").unwrap();

        // The intra-target anchor (LocalType) causes merging despite the
        // external anchor being unreachable. Both symbols end up in one SCC.
        assert_eq!(
            result.nodes.len(),
            1,
            "Impl with intra-target anchor should merge with that anchor"
        );
        assert_eq!(result.nodes[0].symbols.len(), 2);
    }
}
