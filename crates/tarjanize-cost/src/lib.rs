//! Cost analysis for symbol graphs.
//!
//! This crate computes build cost metrics for `SymbolGraph` structures,
//! including critical path analysis. The critical path represents the
//! minimum build time achievable with infinite parallelism.
//!
//! ## Cost Model
//!
//! Rustc compilation has phases with different parallelism characteristics:
//!
//! - **Frontend** (parsing, type checking, borrow checking) — serial within a crate
//! - **Backend** (LLVM optimization, codegen) — parallel across Codegen Units (CGUs)
//! - **Overhead** (linking, metadata) — per-crate fixed costs
//!
//! We model crate compile time using the wall-clock prediction formula:
//!
//! ```text
//! crate_cost = frontend_time + backend_time + overhead
//!
//! where:
//!   frontend_time = Σ symbol.frontend_cost_ms           (serial - sum all)
//!   backend_time  = max(module.backend_cost_ms())       (parallel - max across modules)
//!   overhead      = crate.linking_ms + crate.metadata_ms
//! ```
//!
//! The key insight: backend work is parallelized via CGUs, roughly 2 per module.
//! Only the slowest module's backend cost affects wall-clock time.
//!
//! ## Algorithm
//!
//! The critical path is the longest weighted path through the crate dependency
//! DAG. We compute this using dynamic programming in topological order:
//!
//! 1. Build a directed graph of crate dependencies
//! 2. Process crates in topological order (dependencies before dependents)
//! 3. For each crate: `dist[c] = cost[c] + max(dist[dep] for dep in deps)`
//! 4. The critical path length is the maximum `dist[c]` across all crates

use std::collections::{HashMap, HashSet};
use std::io::Read;

use indexmap::IndexSet;
use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use tarjanize_schemas::{Module, SymbolGraph};

/// Result of critical path analysis.
#[derive(Debug, Clone)]
pub struct CriticalPathResult {
    /// Total cost of the critical path (sum of crate costs along the path).
    pub cost: f64,

    /// Crates on the critical path, from deepest dependency to top-level.
    /// Only populated if `compute_path` was true.
    pub path: Vec<String>,

    /// Total cost of all crates (sequential build time).
    pub total_cost: f64,

    /// Number of crates in the graph.
    pub crate_count: usize,

    /// Number of symbols in the graph.
    pub symbol_count: usize,
}

/// Computes the critical path of a symbol graph at the crate level.
///
/// The critical path is the longest weighted path through the crate dependency
/// DAG. This represents the minimum build time with infinite parallelism.
///
/// # Arguments
///
/// * `symbol_graph` - The symbol graph to analyze
/// * `compute_path` - If true, also compute the actual path (list of crates)
pub fn critical_path(
    symbol_graph: &SymbolGraph,
    compute_path: bool,
) -> CriticalPathResult {
    let (crate_names, crate_costs, symbol_count, graph) =
        build_crate_graph(symbol_graph);

    if crate_names.is_empty() {
        return CriticalPathResult {
            cost: 0.0,
            path: Vec::new(),
            total_cost: 0.0,
            crate_count: 0,
            symbol_count: 0,
        };
    }

    let total_cost: f64 = crate_costs.iter().sum();

    // Topological sort. The graph should be a DAG, but if there are cycles
    // (shouldn't happen after SCC condensation), report total cost.
    let Ok(sorted) = toposort(&graph, None) else {
        return CriticalPathResult {
            cost: total_cost,
            path: Vec::new(),
            total_cost,
            crate_count: crate_names.len(),
            symbol_count,
        };
    };

    // DP: dist[c] = cost to build crate c and all its transitive dependencies.
    // Process in topological order (dependencies come before dependents).
    let mut dist: Vec<f64> = crate_costs.clone();
    let mut predecessor: Vec<Option<NodeIndex>> = vec![None; crate_names.len()];

    for &node in &sorted {
        let node_cost = crate_costs[node.index()];

        // Find max distance among dependencies (incoming edges).
        let mut max_dep_dist = 0.0f64;
        let mut max_dep_node: Option<NodeIndex> = None;

        for dep in graph.neighbors_directed(node, Direction::Incoming) {
            if dist[dep.index()] > max_dep_dist {
                max_dep_dist = dist[dep.index()];
                max_dep_node = Some(dep);
            }
        }

        dist[node.index()] = node_cost + max_dep_dist;
        if compute_path {
            predecessor[node.index()] = max_dep_node;
        }
    }

    // Find the crate with maximum distance (end of critical path).
    let (max_node, &max_cost) = dist
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // Reconstruct path if requested.
    let path = if compute_path {
        let mut result = Vec::new();
        let mut current = Some(NodeIndex::new(max_node));
        while let Some(node) = current {
            result.push(crate_names.get_index(node.index()).unwrap().clone());
            current = predecessor[node.index()];
        }
        result.reverse();
        result
    } else {
        Vec::new()
    };

    CriticalPathResult {
        cost: max_cost,
        path,
        total_cost,
        crate_count: crate_names.len(),
        symbol_count,
    }
}

/// Convenience function to compute critical path from JSON input.
pub fn critical_path_from_reader(
    mut input: impl Read,
    compute_path: bool,
) -> Result<CriticalPathResult, serde_json::Error> {
    let mut json = String::new();
    input
        .read_to_string(&mut json)
        .map_err(serde_json::Error::io)?;
    let symbol_graph: SymbolGraph = serde_json::from_str(&json)?;
    Ok(critical_path(&symbol_graph, compute_path))
}

/// Builds a crate-level dependency graph.
///
/// Returns (`crate_names`, `crate_costs`, `symbol_count`, graph) where:
/// - `crate_names`: `IndexSet` of crate names for O(1) lookup
/// - `crate_costs`: `Vec` of crate wall-clock costs indexed by crate
/// - `symbol_count`: total number of symbols across all crates
/// - `graph`: `DiGraph` where edges point from dependency crates to dependent
///   crates (edge B → A means crate A depends on crate B)
fn build_crate_graph(
    symbol_graph: &SymbolGraph,
) -> (IndexSet<String>, Vec<f64>, usize, DiGraph<usize, ()>) {
    // First, collect all symbol paths and map them to their crate.
    let mut symbol_to_crate: HashMap<String, String> = HashMap::new();
    let mut crate_costs: HashMap<String, f64> = HashMap::new();
    let mut symbol_count = 0;

    for (crate_name, crate_data) in &symbol_graph.crates {
        // Collect symbol paths for dependency resolution.
        collect_symbol_paths(crate_name, &crate_data.root, &mut symbol_to_crate);
        symbol_count += count_symbols_in_module(&crate_data.root);

        // Compute wall-clock cost using the formula:
        // cost = frontend_time + backend_time + overhead
        let frontend_time = collect_frontend_cost(&crate_data.root);
        let backend_time = max_module_backend_cost(&crate_data.root);
        let overhead = crate_data.linking_ms + crate_data.metadata_ms;

        crate_costs.insert(crate_name.clone(), frontend_time + backend_time + overhead);
    }

    // Build crate name index.
    let crate_names: IndexSet<String> =
        symbol_graph.crates.keys().cloned().collect();
    let costs: Vec<f64> = crate_names
        .iter()
        .map(|name| crate_costs.get(name).copied().unwrap_or(0.0))
        .collect();

    // Build crate dependency graph.
    let mut graph = DiGraph::<usize, ()>::with_capacity(crate_names.len(), 0);
    for i in 0..crate_names.len() {
        graph.add_node(i);
    }

    // For each crate, find which other crates it depends on.
    for (crate_name, crate_data) in &symbol_graph.crates {
        if let Some(crate_idx) = crate_names.get_index_of(crate_name) {
            let deps =
                collect_crate_dependencies(&crate_data.root, &symbol_to_crate);
            for dep_crate in deps {
                // Skip self-dependencies.
                if &dep_crate == crate_name {
                    continue;
                }
                if let Some(dep_idx) = crate_names.get_index_of(&dep_crate) {
                    // Edge from dependency → dependent.
                    graph.add_edge(
                        NodeIndex::new(dep_idx),
                        NodeIndex::new(crate_idx),
                        (),
                    );
                }
            }
        }
    }

    (crate_names, costs, symbol_count, graph)
}

/// Recursively collects symbol paths for dependency resolution.
fn collect_symbol_paths(
    module_path: &str,
    module: &Module,
    symbol_to_crate: &mut HashMap<String, String>,
) {
    for symbol_name in module.symbols.keys() {
        let path = format!("{module_path}::{symbol_name}");
        // Extract crate name from module path (first component).
        let crate_name = module_path.split("::").next().unwrap_or(module_path);
        symbol_to_crate.insert(path, crate_name.to_string());
    }

    for (submodule_name, submodule) in &module.submodules {
        let submodule_path = format!("{module_path}::{submodule_name}");
        collect_symbol_paths(&submodule_path, submodule, symbol_to_crate);
    }
}

/// Collects total frontend cost across all symbols in a module tree.
/// Frontend work is serial, so we sum all costs.
fn collect_frontend_cost(module: &Module) -> f64 {
    let mut total = 0.0;

    for symbol in module.symbols.values() {
        total += symbol.frontend_cost_ms;
    }

    for submodule in module.submodules.values() {
        total += collect_frontend_cost(submodule);
    }

    total
}

/// Computes the maximum backend cost across all modules.
///
/// Backend work is parallel via CGUs (roughly 2 per module). The wall-clock
/// backend time is determined by the slowest module, not the sum.
///
/// For a module tree, we compute backend cost at each level and take the max.
fn max_module_backend_cost(module: &Module) -> f64 {
    // Backend cost for this module = sum of its direct symbols' backend costs.
    let this_module_cost: f64 =
        module.symbols.values().map(|s| s.backend_cost_ms).sum();

    // Recursively get max backend cost from submodules.
    let max_submodule_cost = module
        .submodules
        .values()
        .map(max_module_backend_cost)
        .fold(0.0f64, f64::max);

    // The effective backend time is the max of this module and any submodule.
    this_module_cost.max(max_submodule_cost)
}

/// Counts symbols in a module tree.
fn count_symbols_in_module(module: &Module) -> usize {
    let mut count = module.symbols.len();
    for submodule in module.submodules.values() {
        count += count_symbols_in_module(submodule);
    }
    count
}

/// Collects all crates that symbols in this module depend on.
fn collect_crate_dependencies(
    module: &Module,
    symbol_to_crate: &HashMap<String, String>,
) -> HashSet<String> {
    let mut deps = HashSet::new();

    for symbol in module.symbols.values() {
        for dep_path in &symbol.dependencies {
            if let Some(dep_crate) = symbol_to_crate.get(dep_path) {
                deps.insert(dep_crate.clone());
            }
        }
    }

    for submodule in module.submodules.values() {
        deps.extend(collect_crate_dependencies(submodule, symbol_to_crate));
    }

    deps
}

#[cfg(test)]
mod tests {
    use tarjanize_schemas::{Symbol, SymbolKind, Visibility};

    use super::*;

    /// Creates a symbol with only frontend cost (backend = 0).
    fn make_symbol(frontend_cost: f64, deps: &[&str]) -> Symbol {
        make_symbol_split(frontend_cost, 0.0, deps)
    }

    /// Creates a symbol with separate frontend and backend costs.
    fn make_symbol_split(frontend: f64, backend: f64, deps: &[&str]) -> Symbol {
        Symbol {
            file: "test.rs".to_string(),
            frontend_cost_ms: frontend,
            backend_cost_ms: backend,
            dependencies: deps.iter().map(|&s| s.to_string()).collect(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        }
    }

    #[test]
    fn test_single_crate_frontend_only() {
        // One crate with two symbols, frontend-only: cost = 30
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0, &[]));
        symbols
            .insert("bar".to_string(), make_symbol(20.0, &["my_crate::foo"]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
            },
        );

        let graph = SymbolGraph { crates };
        let result = critical_path(&graph, true);

        // Frontend is summed: 10 + 20 = 30
        assert!((result.cost - 30.0).abs() < f64::EPSILON);
        assert_eq!(result.path, vec!["my_crate"]);
        assert_eq!(result.crate_count, 1);
        assert_eq!(result.symbol_count, 2);
    }

    #[test]
    fn test_backend_summed_within_module() {
        // Within a module, backend costs are summed (same CGU).
        // Across modules, we take max (parallel CGUs).
        //
        // Here we have one module with two symbols:
        // - frontend: 10 + 10 = 20 (always summed)
        // - backend: 100 + 50 = 150 (summed within module)
        // - total: 20 + 150 = 170
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol_split(10.0, 100.0, &[]));
        symbols.insert("bar".to_string(), make_symbol_split(10.0, 50.0, &[]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
            },
        );

        let graph = SymbolGraph { crates };
        let result = critical_path(&graph, true);

        assert!((result.cost - 170.0).abs() < f64::EPSILON);
    }

    #[test]
    #[expect(clippy::similar_names, reason = "mod_a/mod_b are intentionally parallel")]
    fn test_backend_max_across_submodules() {
        // Submodules compile in parallel - take max of their backend costs.
        //
        // root module: frontend=5, backend=10
        // submodule A: frontend=5, backend=100
        // submodule B: frontend=5, backend=50
        //
        // frontend = 5 + 5 + 5 = 15 (sum all)
        // backend = max(10, 100, 50) = 100 (max across modules)
        // total = 115
        let mut root_symbols = HashMap::new();
        root_symbols.insert("root_fn".to_string(), make_symbol_split(5.0, 10.0, &[]));

        let mut mod_a_symbols = HashMap::new();
        mod_a_symbols.insert("a_fn".to_string(), make_symbol_split(5.0, 100.0, &[]));

        let mut mod_b_symbols = HashMap::new();
        mod_b_symbols.insert("b_fn".to_string(), make_symbol_split(5.0, 50.0, &[]));

        let mut submodules = HashMap::new();
        submodules.insert(
            "mod_a".to_string(),
            Module {
                symbols: mod_a_symbols,
                submodules: HashMap::new(),
            },
        );
        submodules.insert(
            "mod_b".to_string(),
            Module {
                symbols: mod_b_symbols,
                submodules: HashMap::new(),
            },
        );

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: root_symbols,
                    submodules,
                },
            },
        );

        let graph = SymbolGraph { crates };
        let result = critical_path(&graph, true);

        // frontend = 15, backend = max(10, 100, 50) = 100
        assert!((result.cost - 115.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_crate_overhead_included() {
        // Crate with linking and metadata overhead.
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0, &[]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 5.0,
                metadata_ms: 3.0,
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
            },
        );

        let graph = SymbolGraph { crates };
        let result = critical_path(&graph, true);

        // frontend(10) + backend(0) + linking(5) + metadata(3) = 18
        assert!((result.cost - 18.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_crate() {
        // One crate with two symbols: total cost = 30
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0, &[]));
        symbols
            .insert("bar".to_string(), make_symbol(20.0, &["my_crate::foo"]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
            },
        );

        let graph = SymbolGraph { crates };
        let result = critical_path(&graph, true);

        // Single crate, so critical path = total cost = 30
        assert!((result.cost - 30.0).abs() < f64::EPSILON);
        assert_eq!(result.path, vec!["my_crate"]);
        assert_eq!(result.crate_count, 1);
        assert_eq!(result.symbol_count, 2);
    }

    #[test]
    fn test_two_independent_crates() {
        // Two independent crates: can build in parallel
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a.insert("foo".to_string(), make_symbol(100.0, &[]));
        crates.insert(
            "crate_a".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: symbols_a,
                    submodules: HashMap::new(),
                },
            },
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert("bar".to_string(), make_symbol(50.0, &[]));
        crates.insert(
            "crate_b".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: symbols_b,
                    submodules: HashMap::new(),
                },
            },
        );

        let graph = SymbolGraph { crates };
        let result = critical_path(&graph, true);

        // Critical path is max(100, 50) = 100
        assert!((result.cost - 100.0).abs() < f64::EPSILON);
        // Total cost is 150 (sequential)
        assert!((result.total_cost - 150.0).abs() < f64::EPSILON);
        // Parallelism ratio = 150/100 = 1.5x
        assert_eq!(result.crate_count, 2);
    }

    #[test]
    fn test_crate_chain() {
        // crate_a (100) depends on crate_b (50)
        // Critical path = 150
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a
            .insert("foo".to_string(), make_symbol(100.0, &["crate_b::bar"]));
        crates.insert(
            "crate_a".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: symbols_a,
                    submodules: HashMap::new(),
                },
            },
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert("bar".to_string(), make_symbol(50.0, &[]));
        crates.insert(
            "crate_b".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: symbols_b,
                    submodules: HashMap::new(),
                },
            },
        );

        let graph = SymbolGraph { crates };
        let result = critical_path(&graph, true);

        // Critical path = crate_b + crate_a = 150
        assert!((result.cost - 150.0).abs() < f64::EPSILON);
        assert_eq!(result.path, vec!["crate_b", "crate_a"]);
        // Total cost = 150 (no parallelism possible)
        assert!((result.total_cost - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diamond_crates() {
        // crate_a depends on crate_b (100) and crate_c (50)
        // crate_b and crate_c both depend on crate_d (30)
        // Critical path: d(30) → b(100) → a(10) = 140
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a.insert(
            "a".to_string(),
            make_symbol(10.0, &["crate_b::b", "crate_c::c"]),
        );
        crates.insert(
            "crate_a".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: symbols_a,
                    submodules: HashMap::new(),
                },
            },
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert("b".to_string(), make_symbol(100.0, &["crate_d::d"]));
        crates.insert(
            "crate_b".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: symbols_b,
                    submodules: HashMap::new(),
                },
            },
        );

        let mut symbols_c = HashMap::new();
        symbols_c.insert("c".to_string(), make_symbol(50.0, &["crate_d::d"]));
        crates.insert(
            "crate_c".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: symbols_c,
                    submodules: HashMap::new(),
                },
            },
        );

        let mut symbols_d = HashMap::new();
        symbols_d.insert("d".to_string(), make_symbol(30.0, &[]));
        crates.insert(
            "crate_d".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols: symbols_d,
                    submodules: HashMap::new(),
                },
            },
        );

        let graph = SymbolGraph { crates };
        let result = critical_path(&graph, true);

        // Critical path = d(30) + b(100) + a(10) = 140
        assert!((result.cost - 140.0).abs() < f64::EPSILON);
        assert_eq!(result.path, vec!["crate_d", "crate_b", "crate_a"]);
        // Total cost = 190
        assert!((result.total_cost - 190.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_graph() {
        let graph = SymbolGraph {
            crates: HashMap::new(),
        };
        let result = critical_path(&graph, true);

        assert!((result.cost).abs() < f64::EPSILON);
        assert!(result.path.is_empty());
        assert_eq!(result.crate_count, 0);
        assert_eq!(result.symbol_count, 0);
    }
}
