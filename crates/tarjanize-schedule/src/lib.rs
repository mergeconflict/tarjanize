//! Build schedule computation: forward/backward DP, critical path, swim lanes.
//!
//! Extracted from `tarjanize-viz` so both the static HTML visualization and the
//! interactive split explorer can share scheduling primitives.

pub mod data;
pub mod export;
pub mod heatmap;
pub mod recommend;
pub mod schedule;
pub mod split;
pub mod target_graph;

use tarjanize_schemas::{CostModel, Module, SymbolGraph, sum_event_times};

use crate::schedule::TargetGraph;

/// Auto-fits a cost model from a `SymbolGraph`'s profiling data.
///
/// Uses lib targets with wall-clock profiling data (`wall_time_ms > 0`)
/// to fit the MAGSAC++ regression model. Returns `None` if insufficient
/// profiled targets are available (fewer than 4 lib targets with data).
///
/// This replaces the manual `tarjanize cost --output-model` step for
/// interactive use, where we want to auto-fit on startup without
/// requiring a separate CLI invocation.
pub fn auto_fit_cost_model(symbol_graph: &SymbolGraph) -> Option<CostModel> {
    // Fit using lib targets only because they have the most reliable
    // profiling data. Non-lib targets (tests, examples) often lack
    // wall-clock timing or have inflated costs from re-compilation.
    let options = tarjanize_cost::CostOptions {
        fit_libs_only: true,
    };
    let result = tarjanize_cost::fit(symbol_graph, options);
    tarjanize_cost::build_cost_model(&result)
}

/// Builds a target graph from a `SymbolGraph` and optional `CostModel`.
///
/// For each target in the symbol graph:
/// 1. Computes the three regression predictors (attr, meta, other)
/// 2. Predicts cost via `CostModel::predict()`, or falls back to the
///    effective timing (wall-clock if available, per-symbol sum otherwise)
/// 3. Counts symbols recursively through the module tree
///
/// Test targets are augmented with their lib's per-symbol costs when
/// the test target has no wall-clock profiling data (same logic as
/// `tarjanize-cost`'s `build_target_graph`).
pub fn build_target_graph(
    symbol_graph: &SymbolGraph,
    cost_model: Option<&CostModel>,
) -> TargetGraph {
    use std::collections::HashMap;
    use std::time::Duration;

    use indexmap::IndexSet;
    use petgraph::graph::DiGraph;

    /// Converts f64 milliseconds to Duration at the boundary between
    /// regression math (f64) and scheduling (Duration).
    fn ms_to_duration(ms: f64) -> Duration {
        Duration::from_secs_f64(ms / 1000.0)
    }

    let mut names: IndexSet<String> = IndexSet::new();
    let mut costs: Vec<Duration> = Vec::new();
    let mut symbol_counts: Vec<usize> = Vec::new();

    // First pass: register targets, compute costs and symbol counts.
    // Costs stay as f64 ms during computation (mixing regression
    // outputs), then convert to Duration at the end.
    let mut cost_map: HashMap<String, f64> = HashMap::new();
    let mut sym_count_map: HashMap<String, usize> = HashMap::new();

    for (package_name, package) in &symbol_graph.packages {
        for (target_key, target_data) in &package.targets {
            let target_id = format!("{package_name}/{target_key}");
            names.insert(target_id.clone());

            let syms = count_symbols(&target_data.root);
            sym_count_map.insert(target_id.clone(), syms);

            // Compute cost: use CostModel if available, else effective.
            let attr = collect_frontend_cost(&target_data.root);
            let cost = if let Some(model) = cost_model {
                let meta: f64 = target_data
                    .timings
                    .event_times_ms
                    .iter()
                    .filter(|(k, _)| k.starts_with("metadata_decode_"))
                    .map(|(_, v)| v)
                    .sum();
                let other: f64 = target_data
                    .timings
                    .event_times_ms
                    .iter()
                    .filter(|(k, _)| !k.starts_with("metadata_decode_"))
                    .map(|(_, v)| v)
                    .sum();
                model.predict(attr, meta, other)
            } else if !target_data.timings.wall_time.is_zero() {
                // Use wall-clock when profiled.
                target_data.timings.wall_time.as_secs_f64() * 1000.0
            } else {
                // Fall back to per-symbol sum.
                attr
            };

            cost_map.insert(target_id, cost);
        }
    }

    // Augment test targets with lib costs when the test has no wall-clock
    // profiling data and no cost model is in use. When a cost model IS
    // provided, predictions already account for the full compilation.
    if cost_model.is_none() {
        for (package_name, package) in &symbol_graph.packages {
            if !package.targets.contains_key("test") {
                continue;
            }
            let test_id = format!("{package_name}/test");
            let lib_id = format!("{package_name}/lib");

            let test_has_wall = package
                .targets
                .get("test")
                .is_some_and(|t| !t.timings.wall_time.is_zero());

            // Augment when test lacks wall-clock data.
            if !test_has_wall
                && let Some(&lib_cost) = cost_map.get(&lib_id)
                && let Some(test_cost) = cost_map.get_mut(&test_id)
            {
                *test_cost += lib_cost;
            }
        }
    }

    // Collect vectors in names order, converting f64 ms to Duration.
    for name in &names {
        costs.push(ms_to_duration(*cost_map.get(name).unwrap_or(&0.0)));
        symbol_counts.push(*sym_count_map.get(name).unwrap_or(&0));
    }

    // Build dependency graph.
    let mut graph = DiGraph::<usize, ()>::with_capacity(names.len(), 0);
    for _ in 0..names.len() {
        graph.add_node(0);
    }

    for (package_name, package) in &symbol_graph.packages {
        for (target_key, target_data) in &package.targets {
            let target_id = format!("{package_name}/{target_key}");
            let Some(target_idx) = names.get_index_of(&target_id) else {
                continue;
            };

            for dep_target in &target_data.dependencies {
                if dep_target == &target_id {
                    continue;
                }
                if let Some(dep_idx) = names.get_index_of(dep_target) {
                    graph.add_edge(
                        petgraph::graph::NodeIndex::new(dep_idx),
                        petgraph::graph::NodeIndex::new(target_idx),
                        (),
                    );
                }
            }
        }
    }

    TargetGraph {
        names,
        costs,
        symbol_counts,
        graph,
    }
}

/// Recursively sums all symbol `event_times_ms` in a module tree.
fn collect_frontend_cost(module: &Module) -> f64 {
    let mut total = 0.0;
    for symbol in module.symbols.values() {
        total += sum_event_times(&symbol.event_times_ms);
    }
    for submodule in module.submodules.values() {
        total += collect_frontend_cost(submodule);
    }
    total
}

/// Recursively counts symbols in a module tree.
fn count_symbols(module: &Module) -> usize {
    let mut count = module.symbols.len();
    for submodule in module.submodules.values() {
        count += count_symbols(submodule);
    }
    count
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::time::Duration;

    use tarjanize_schemas::{
        Package, Symbol, SymbolKind, Target, TargetTimings, Visibility,
    };

    use super::*;
    use crate::split::SplitOperation;

    /// Converts f64 milliseconds to Duration for test convenience.
    fn ms(val: f64) -> Duration {
        Duration::from_secs_f64(val / 1000.0)
    }

    #[test]
    fn auto_fit_returns_none_for_empty_graph() {
        let sg = SymbolGraph::default();
        assert!(auto_fit_cost_model(&sg).is_none());
    }

    // =================================================================
    // Helpers
    // =================================================================

    /// Builds a simple `Symbol` with a single `typeck` event cost.
    fn make_symbol(event_cost: f64) -> Symbol {
        Symbol {
            file: "lib.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), event_cost)]),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        }
    }

    // =================================================================
    // build_target_graph: cost model prediction path
    // =================================================================

    /// When a `CostModel` is provided, `build_target_graph` should use
    /// the model's `predict()` for cost, not the wall-clock fallback.
    /// The model decomposes timing into attr (per-symbol sum), meta
    /// (`metadata_decode_*` events), and other (remaining events).
    #[test]
    fn build_target_graph_with_cost_model() {
        let mut symbols = HashMap::new();
        symbols.insert("Foo".to_string(), make_symbol(10.0));
        symbols.insert("Bar".to_string(), make_symbol(5.0));

        let root = Module {
            symbols,
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings {
                wall_time: ms(20.0),
                event_times_ms: HashMap::from([
                    ("metadata_decode_something".to_string(), 3.0),
                    ("other_event".to_string(), 2.0),
                ]),
            },
            dependencies: HashSet::new(),
            root,
        };

        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("pkg-a".to_string(), Package { targets });
        let sg = SymbolGraph { packages };

        // Model: attr * 1.0 + meta * 0.5 + other * 0.3
        let model = CostModel {
            coeff_attr: 1.0,
            coeff_meta: 0.5,
            coeff_other: 0.3,
            r_squared: 0.9,
            inlier_threshold: 1.0,
        };

        let tg = build_target_graph(&sg, Some(&model));

        // attr = 10.0 + 5.0 = 15.0
        // meta = 3.0 (metadata_decode_something)
        // other = 2.0 (other_event)
        // predicted = 1.0*15.0 + 0.5*3.0 + 0.3*2.0 = 15.0 + 1.5 + 0.6 = 17.1
        let idx = tg
            .names
            .get_index_of("pkg-a/lib")
            .expect("pkg-a/lib must exist");
        let expected_ms = 1.0 * 15.0 + 0.5 * 3.0 + 0.3 * 2.0;
        assert_eq!(
            tg.costs[idx],
            ms(expected_ms),
            "cost should be {expected_ms}ms",
        );
        // Verify it is NOT the wall-clock value.
        assert_ne!(
            tg.costs[idx],
            ms(20.0),
            "cost should not be the wall-clock fallback"
        );
    }

    // =================================================================
    // build_target_graph: wall-clock fallback (no cost model)
    // =================================================================

    /// Without a cost model, `build_target_graph` should use the target's
    /// `wall_time_ms` when it is positive.
    #[test]
    fn build_target_graph_wall_clock_fallback() {
        let mut symbols = HashMap::new();
        symbols.insert("Foo".to_string(), make_symbol(10.0));
        symbols.insert("Bar".to_string(), make_symbol(5.0));

        let root = Module {
            symbols,
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings {
                wall_time: ms(20.0),
                event_times_ms: HashMap::new(),
            },
            dependencies: HashSet::new(),
            root,
        };

        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("pkg-a".to_string(), Package { targets });
        let sg = SymbolGraph { packages };

        let tg = build_target_graph(&sg, None);

        let idx = tg
            .names
            .get_index_of("pkg-a/lib")
            .expect("pkg-a/lib must exist");
        assert_eq!(tg.costs[idx], ms(20.0), "cost should be wall_time=20ms",);
    }

    // =================================================================
    // build_target_graph: per-symbol sum fallback (no wall-clock)
    // =================================================================

    /// When there is no cost model and `wall_time_ms` is zero, the cost
    /// falls back to the sum of all per-symbol `event_times_ms`.
    #[test]
    fn build_target_graph_symbol_sum_fallback() {
        let mut symbols = HashMap::new();
        symbols.insert("Foo".to_string(), make_symbol(10.0));
        symbols.insert("Bar".to_string(), make_symbol(5.0));

        let root = Module {
            symbols,
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings {
                wall_time: Duration::ZERO,
                event_times_ms: HashMap::new(),
            },
            dependencies: HashSet::new(),
            root,
        };

        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), target);
        let mut packages = HashMap::new();
        packages.insert("pkg-a".to_string(), Package { targets });
        let sg = SymbolGraph { packages };

        let tg = build_target_graph(&sg, None);

        let idx = tg
            .names
            .get_index_of("pkg-a/lib")
            .expect("pkg-a/lib must exist");
        // Sum of symbol costs: 10.0 + 5.0 = 15.0
        assert_eq!(
            tg.costs[idx],
            ms(15.0),
            "cost should be per-symbol sum=15ms",
        );
    }

    // =================================================================
    // build_target_graph: test target augmentation
    // =================================================================

    /// When a package has both "lib" and "test" targets and the test
    /// target has no wall-clock data (`wall_time_ms` == 0), the test's
    /// cost is augmented with the lib's cost. This compensates for test
    /// targets re-compiling the lib's code.
    #[test]
    fn build_target_graph_test_augmentation() {
        // lib target: cost = 10ms (wall-clock).
        let lib_target = Target {
            timings: TargetTimings {
                wall_time: ms(10.0),
                event_times_ms: HashMap::new(),
            },
            dependencies: HashSet::new(),
            root: Module {
                symbols: HashMap::from([(
                    "LibFn".to_string(),
                    make_symbol(10.0),
                )]),
                submodules: HashMap::new(),
            },
        };

        // test target: per-symbol cost = 5ms, no wall-clock data.
        let test_target = Target {
            timings: TargetTimings {
                wall_time: Duration::ZERO,
                event_times_ms: HashMap::new(),
            },
            dependencies: HashSet::new(),
            root: Module {
                symbols: HashMap::from([(
                    "TestFn".to_string(),
                    make_symbol(5.0),
                )]),
                submodules: HashMap::new(),
            },
        };

        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), lib_target);
        targets.insert("test".to_string(), test_target);
        let mut packages = HashMap::new();
        packages.insert("pkg-a".to_string(), Package { targets });
        let sg = SymbolGraph { packages };

        let tg = build_target_graph(&sg, None);

        let test_idx = tg
            .names
            .get_index_of("pkg-a/test")
            .expect("pkg-a/test must exist");
        // test cost = per-symbol sum (5.0) + lib cost (10.0) = 15.0
        assert_eq!(
            tg.costs[test_idx],
            ms(15.0),
            "test cost should be 5+10=15ms",
        );
    }

    // =================================================================
    // build_target_graph: dependency edges
    // =================================================================

    /// Cross-package dependencies should produce edges in the target
    /// graph. Edge direction: dependency -> dependent.
    #[test]
    fn build_target_graph_dependencies() {
        let pkg_a_target = Target {
            timings: TargetTimings {
                wall_time: ms(10.0),
                event_times_ms: HashMap::new(),
            },
            dependencies: HashSet::new(),
            root: Module {
                symbols: HashMap::from([(
                    "FnA".to_string(),
                    make_symbol(10.0),
                )]),
                submodules: HashMap::new(),
            },
        };

        let pkg_b_target = Target {
            timings: TargetTimings {
                wall_time: ms(20.0),
                event_times_ms: HashMap::new(),
            },
            // pkg-b/lib depends on pkg-a/lib.
            dependencies: HashSet::from(["pkg-a/lib".to_string()]),
            root: Module {
                symbols: HashMap::from([(
                    "FnB".to_string(),
                    make_symbol(20.0),
                )]),
                submodules: HashMap::new(),
            },
        };

        let mut packages = HashMap::new();
        packages.insert(
            "pkg-a".to_string(),
            Package {
                targets: HashMap::from([("lib".to_string(), pkg_a_target)]),
            },
        );
        packages.insert(
            "pkg-b".to_string(),
            Package {
                targets: HashMap::from([("lib".to_string(), pkg_b_target)]),
            },
        );
        let sg = SymbolGraph { packages };

        let tg = build_target_graph(&sg, None);

        let a_idx = tg
            .names
            .get_index_of("pkg-a/lib")
            .expect("pkg-a/lib must exist");
        let b_idx = tg
            .names
            .get_index_of("pkg-b/lib")
            .expect("pkg-b/lib must exist");

        // There should be an edge from pkg-a/lib -> pkg-b/lib
        // (dep -> dependent).
        let has_edge = tg
            .graph
            .neighbors_directed(
                petgraph::graph::NodeIndex::new(a_idx),
                petgraph::Direction::Outgoing,
            )
            .any(|n| n.index() == b_idx);
        assert!(has_edge, "expected edge from pkg-a/lib -> pkg-b/lib");
    }

    // =================================================================
    // Full pipeline: build graph -> split -> verify schedule improves
    // =================================================================

    /// Builds a 3-package chain: pkg-a (10ms) -> pkg-b (50ms, 2
    /// independent SCCs of ~25ms each) -> pkg-c (5ms).
    ///
    /// Critical path before splitting: 10 + 50 + 5 = 65ms.
    fn make_three_package_chain() -> SymbolGraph {
        let pkg_a_target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root: Module {
                symbols: HashMap::from([(
                    "FnA".to_string(),
                    make_symbol(10.0),
                )]),
                submodules: HashMap::new(),
            },
        };

        let pkg_b_target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::from(["pkg-a/lib".to_string()]),
            root: Module {
                symbols: HashMap::from([
                    ("alpha".to_string(), make_symbol(25.0)),
                    ("beta".to_string(), make_symbol(25.0)),
                ]),
                submodules: HashMap::new(),
            },
        };

        let pkg_c_target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::from(["pkg-b/lib".to_string()]),
            root: Module {
                symbols: HashMap::from([("FnC".to_string(), make_symbol(5.0))]),
                submodules: HashMap::new(),
            },
        };

        let mut packages = HashMap::new();
        packages.insert(
            "pkg-a".to_string(),
            Package {
                targets: HashMap::from([("lib".to_string(), pkg_a_target)]),
            },
        );
        packages.insert(
            "pkg-b".to_string(),
            Package {
                targets: HashMap::from([("lib".to_string(), pkg_b_target)]),
            },
        );
        packages.insert(
            "pkg-c".to_string(),
            Package {
                targets: HashMap::from([("lib".to_string(), pkg_c_target)]),
            },
        );
        SymbolGraph { packages }
    }

    /// Splitting a large target into two parallel pieces should reduce
    /// the critical path length.
    #[test]
    fn full_pipeline_split_reduces_critical_path() {
        let sg = make_three_package_chain();
        let tg = build_target_graph(&sg, None);
        let original_schedule = schedule::compute_schedule(&tg);
        let original_cp = original_schedule.summary.critical_path;

        // Sanity: original critical path should be ~65ms
        // (10 + 50 + 5, sequential chain).
        assert_eq!(
            original_cp,
            ms(65.0),
            "original critical path should be ~65ms"
        );

        // Condense pkg-b/lib to find its 2 independent SCCs.
        let intra = target_graph::condense_target(&sg, "pkg-b/lib")
            .expect("pkg-b/lib must be condensable");
        assert_eq!(intra.nodes.len(), 2);

        // Split out the SCC containing "alpha" (~25ms).
        let scc_alpha = intra
            .nodes
            .iter()
            .find(|n| n.symbols.iter().any(|s| s.contains("alpha")))
            .expect("alpha SCC must exist");

        let ops = vec![SplitOperation {
            source_target: "pkg-b/lib".to_string(),
            new_crate_name: "pkg-b-split".to_string(),
            selected_sccs: vec![scc_alpha.id],
        }];

        let (split_tg, results) = split::apply_splits(&tg, &sg, &ops);
        assert_eq!(results.len(), 1);
        assert!(!results[0].sccs_in_new_crate.is_empty());

        // The split critical path should be shorter: one ~25ms SCC
        // now compiles in parallel with the residual.
        let split_schedule = schedule::compute_schedule(&split_tg);
        let split_cp = split_schedule.summary.critical_path;
        assert!(
            split_cp < original_cp,
            "split critical path ({split_cp:?}) should be shorter \
             than original ({original_cp:?})"
        );
    }
}
