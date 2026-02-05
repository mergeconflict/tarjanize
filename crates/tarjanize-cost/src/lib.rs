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
//!   overhead      = crate.metadata_ms
//! ```
//!
//! The key insight: backend work is parallelized via CGUs, roughly 2 per module.
//! Only the slowest module's backend cost affects wall-clock time.
//!
//! ## Target-Level Analysis
//!
//! The critical path is computed at the **target** level, not package level.
//! Each compilation target (lib, test, bin/cli, etc.) is a separate node in
//! the dependency graph. This is important because:
//!
//! - Dev-dependencies only affect test targets, not lib targets
//! - The lib→test dependency is explicit (tests depend on their lib)
//! - No artificial cycles from dev-dependencies
//!
//! Target identifiers use the format `{package}/{target}`:
//! - `my-package/lib` - library target
//! - `my-package/test` - unit test target
//! - `my-package/bin/cli` - binary target named "cli"
//!
//! ## Algorithm
//!
//! The critical path is the longest weighted path through the target dependency
//! DAG. We compute this using dynamic programming in topological order:
//!
//! 1. Build a directed graph of target dependencies
//! 2. Process targets in topological order (dependencies before dependents)
//! 3. For each target: `dist[t] = cost[t] + max(dist[dep] for dep in deps)`
//! 4. The critical path length is the maximum `dist[t]` across all targets

use std::collections::{HashMap, HashSet};
use std::io::Read;

use indexmap::IndexSet;
use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use tarjanize_schemas::{Module, SymbolGraph};

/// Details about a single target on the critical path.
#[derive(Debug, Clone)]
pub struct TargetOnPath {
    /// Target identifier in `{package}/{target}` format.
    /// Examples: `my-package/lib`, `my-package/test`, `my-package/bin/cli`.
    pub name: String,

    /// Estimated cost of this target alone (frontend + backend + overhead).
    pub cost: f64,

    /// Cumulative cost from the start of the critical path to this target
    /// (includes this target's cost and all its transitive dependencies on the path).
    pub cumulative_cost: f64,

    /// Direct dependencies of this target (target identifiers).
    pub dependencies: Vec<String>,
}

/// Result of critical path analysis.
#[derive(Debug, Clone)]
pub struct CriticalPathResult {
    /// Total cost of the critical path (sum of target costs along the path).
    pub cost: f64,

    /// Targets on the critical path, from deepest dependency to top-level.
    /// Each entry is a target identifier like `my-package/lib`.
    pub path: Vec<String>,

    /// Detailed information for each target on the critical path.
    pub path_details: Vec<TargetOnPath>,

    /// All targets with their costs, sorted by cost descending.
    pub all_targets: Vec<TargetOnPath>,

    /// Total cost of all targets (sequential build time).
    pub total_cost: f64,

    /// Number of targets in the graph.
    pub target_count: usize,

    /// Number of symbols in the graph.
    pub symbol_count: usize,
}

/// Computes the critical path of a symbol graph at the target level.
///
/// The critical path is the longest weighted path through the target dependency
/// DAG. This represents the minimum build time with infinite parallelism.
///
/// Each compilation target (lib, test, bin, etc.) is a separate node. This
/// naturally resolves dev-dependency "cycles" because test targets depend on
/// lib targets, not vice versa.
#[expect(
    clippy::too_many_lines,
    reason = "core algorithm, splitting would obscure logic"
)]
pub fn critical_path(symbol_graph: &SymbolGraph) -> CriticalPathResult {
    let (target_names, target_costs, symbol_count, graph) =
        build_target_graph(symbol_graph);

    if target_names.is_empty() {
        return CriticalPathResult {
            cost: 0.0,
            path: Vec::new(),
            path_details: Vec::new(),
            all_targets: Vec::new(),
            total_cost: 0.0,
            target_count: 0,
            symbol_count: 0,
        };
    }

    let total_cost: f64 = target_costs.iter().sum();

    // Topological sort. With target-level analysis, the graph should always
    // be a DAG (no cycles from dev-dependencies). If there are cycles, it
    // indicates a real circular dependency which is a configuration error.
    let Ok(sorted) = toposort(&graph, None) else {
        eprintln!(
            "WARNING: cycle detected in target dependency graph, \
             skipping critical path computation"
        );
        // Return early with just the target costs.
        let mut all_targets: Vec<TargetOnPath> = target_names
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                let node = NodeIndex::new(idx);
                let cost = target_costs[idx];
                let dependencies: Vec<String> = graph
                    .neighbors_directed(node, Direction::Incoming)
                    .map(|dep| {
                        target_names.get_index(dep.index()).unwrap().clone()
                    })
                    .collect();
                TargetOnPath {
                    name: name.clone(),
                    cost,
                    cumulative_cost: cost,
                    dependencies,
                }
            })
            .collect();
        all_targets.sort_by(|a, b| b.cost.partial_cmp(&a.cost).unwrap());

        return CriticalPathResult {
            cost: total_cost,
            path: Vec::new(),
            path_details: Vec::new(),
            all_targets,
            total_cost,
            target_count: target_names.len(),
            symbol_count,
        };
    };

    // DP: dist[t] = cost to build target t and all its transitive dependencies.
    let mut dist: Vec<f64> = target_costs.clone();
    let mut predecessor: Vec<Option<NodeIndex>> =
        vec![None; target_names.len()];

    for &node in &sorted {
        let node_cost = target_costs[node.index()];

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
        predecessor[node.index()] = max_dep_node;
    }

    // Find the target with maximum distance (end of critical path).
    let (max_node, &max_cost) = dist
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // Reconstruct the critical path by following predecessors from the end node.
    let mut path_nodes = Vec::new();
    let mut current = Some(NodeIndex::new(max_node));
    while let Some(node) = current {
        path_nodes.push(node);
        current = predecessor[node.index()];
    }
    path_nodes.reverse();

    // Build path names and details.
    let path: Vec<String> = path_nodes
        .iter()
        .map(|node| target_names.get_index(node.index()).unwrap().clone())
        .collect();

    let path_details: Vec<TargetOnPath> = path_nodes
        .iter()
        .map(|&node| {
            let name = target_names.get_index(node.index()).unwrap().clone();
            let cost = target_costs[node.index()];
            let cumulative_cost = dist[node.index()];

            let dependencies: Vec<String> = graph
                .neighbors_directed(node, Direction::Incoming)
                .map(|dep| target_names.get_index(dep.index()).unwrap().clone())
                .collect();

            TargetOnPath {
                name,
                cost,
                cumulative_cost,
                dependencies,
            }
        })
        .collect();

    // Build all_targets: every target with its cost, sorted by cost descending.
    let mut all_targets: Vec<TargetOnPath> = target_names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let node = NodeIndex::new(idx);
            let cost = target_costs[idx];
            let cumulative_cost = dist[idx];

            let dependencies: Vec<String> = graph
                .neighbors_directed(node, Direction::Incoming)
                .map(|dep| target_names.get_index(dep.index()).unwrap().clone())
                .collect();

            TargetOnPath {
                name: name.clone(),
                cost,
                cumulative_cost,
                dependencies,
            }
        })
        .collect();

    all_targets.sort_by(|a, b| b.cost.partial_cmp(&a.cost).unwrap());

    CriticalPathResult {
        cost: max_cost,
        path,
        path_details,
        all_targets,
        total_cost,
        target_count: target_names.len(),
        symbol_count,
    }
}

/// Convenience function to compute critical path from JSON input.
pub fn critical_path_from_reader(
    mut input: impl Read,
) -> Result<CriticalPathResult, serde_json::Error> {
    let mut json = String::new();
    input
        .read_to_string(&mut json)
        .map_err(serde_json::Error::io)?;
    let symbol_graph: SymbolGraph = serde_json::from_str(&json)?;
    Ok(critical_path(&symbol_graph))
}

/// Builds a target-level dependency graph.
///
/// Returns (`target_names`, `target_costs`, `symbol_count`, graph) where:
/// - `target_names`: `IndexSet` of target identifiers (`{package}/{target}`) for O(1) lookup
/// - `target_costs`: `Vec` of target wall-clock costs indexed by target
/// - `symbol_count`: total number of symbols across all targets
/// - `graph`: `DiGraph` where edges point from dependency targets to dependent
///   targets (edge B → A means target A depends on target B)
fn build_target_graph(
    symbol_graph: &SymbolGraph,
) -> (IndexSet<String>, Vec<f64>, usize, DiGraph<usize, ()>) {
    // Build target name index. Each target gets a unique identifier: `{package}/{target}`.
    // Example: "my-package/lib", "my-package/test", "my-package/bin/cli".
    let mut target_names: IndexSet<String> = IndexSet::new();
    let mut target_costs: HashMap<String, f64> = HashMap::new();
    let mut symbol_count = 0;

    for (package_name, package) in &symbol_graph.packages {
        for (target_key, crate_data) in &package.targets {
            // Target identifier: {package}/{target}
            let target_id = format!("{package_name}/{target_key}");
            target_names.insert(target_id.clone());

            // Count symbols in this target.
            symbol_count += count_symbols_in_module(&crate_data.root);

            // Compute wall-clock cost using the formula:
            // cost = frontend_time + backend_time + overhead
            let frontend_time = collect_frontend_cost(&crate_data.root);
            let backend_time = max_module_backend_cost(&crate_data.root);
            let overhead = crate_data.metadata_ms;

            target_costs
                .insert(target_id, frontend_time + backend_time + overhead);
        }
    }

    let costs: Vec<f64> = target_names
        .iter()
        .map(|name| target_costs.get(name).copied().unwrap_or(0.0))
        .collect();

    // Build target dependency graph.
    let mut graph = DiGraph::<usize, ()>::with_capacity(target_names.len(), 0);
    for i in 0..target_names.len() {
        graph.add_node(i);
    }

    // For each target, find which other targets it depends on.
    for (package_name, package) in &symbol_graph.packages {
        for (target_key, crate_data) in &package.targets {
            let target_id = format!("{package_name}/{target_key}");
            let Some(target_idx) = target_names.get_index_of(&target_id) else {
                continue;
            };

            // Collect dependencies from symbol paths.
            let deps =
                collect_target_dependencies(&crate_data.root, &target_names);

            for dep_target in deps {
                // Skip self-dependencies.
                if dep_target == target_id {
                    continue;
                }
                if let Some(dep_idx) = target_names.get_index_of(&dep_target) {
                    // Edge from dependency → dependent.
                    graph.add_edge(
                        NodeIndex::new(dep_idx),
                        NodeIndex::new(target_idx),
                        (),
                    );
                }
            }
        }
    }

    (target_names, costs, symbol_count, graph)
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

/// Collects all targets that symbols in this module depend on.
///
/// Parses the `[package/target]::path` format from symbol dependencies.
/// Returns target identifiers like `my-package/lib`.
fn collect_target_dependencies(
    module: &Module,
    known_targets: &IndexSet<String>,
) -> HashSet<String> {
    let mut deps = HashSet::new();

    for symbol in module.symbols.values() {
        for dep_path in &symbol.dependencies {
            // Parse the [package/target]::path format.
            if let Some(target_id) = parse_target_from_path(dep_path) {
                // Only include if it's a known target in our graph.
                if known_targets.contains(&target_id) {
                    deps.insert(target_id);
                }
            }
        }
    }

    for submodule in module.submodules.values() {
        deps.extend(collect_target_dependencies(submodule, known_targets));
    }

    deps
}

/// Parses a target identifier from a symbol path.
///
/// Symbol paths use the format `[package/target]::module::symbol`.
/// This function extracts the `package/target` portion.
///
/// Returns `None` if the path doesn't match the expected format.
fn parse_target_from_path(path: &str) -> Option<String> {
    // Check for bracketed prefix: [package/target]::...
    if !path.starts_with('[') {
        return None;
    }

    // Find the closing bracket.
    let bracket_end = path.find(']')?;

    // Extract the content inside brackets (package/target).
    let inner = &path[1..bracket_end];

    // Validate it contains a slash (package/target format).
    if !inner.contains('/') {
        return None;
    }

    Some(inner.to_string())
}

#[cfg(test)]
mod tests {
    use tarjanize_schemas::{Package, Symbol, SymbolKind, Visibility};

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

    /// Creates a crate with the given root module and default overhead.
    fn make_crate(root: Module) -> tarjanize_schemas::Crate {
        tarjanize_schemas::Crate {
            root,
            ..Default::default()
        }
    }

    /// Creates a crate with the given root module and specified overhead.
    fn make_crate_with_overhead(
        root: Module,
        metadata_ms: f64,
    ) -> tarjanize_schemas::Crate {
        tarjanize_schemas::Crate {
            metadata_ms,
            root,
            ..Default::default()
        }
    }

    /// Creates a package with a single "lib" target containing the given crate.
    fn make_package(crate_data: tarjanize_schemas::Crate) -> Package {
        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), crate_data);
        Package { targets }
    }

    /// Creates a `SymbolGraph` from a map of package names to crates.
    /// Each crate becomes the "lib" target of a package with the same name.
    fn make_graph(
        crates: HashMap<String, tarjanize_schemas::Crate>,
    ) -> SymbolGraph {
        let packages = crates
            .into_iter()
            .map(|(name, crate_data)| (name, make_package(crate_data)))
            .collect();
        SymbolGraph { packages }
    }

    /// Helper to create a dependency path in the new format.
    /// `dep("pkg", "lib", "foo")` returns `"[pkg/lib]::foo"`.
    fn dep(package: &str, target: &str, path: &str) -> String {
        format!("[{package}/{target}]::{path}")
    }

    /// Helper to create a target identifier.
    /// `target_id("pkg", "lib")` returns `"pkg/lib"`.
    fn target_id(package: &str, target: &str) -> String {
        format!("{package}/{target}")
    }

    #[test]
    fn test_single_target_frontend_only() {
        // One target with two symbols, frontend-only: cost = 30
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0, &[]));
        symbols.insert(
            "bar".to_string(),
            make_symbol(20.0, &[&dep("my-pkg", "lib", "foo")]),
        );

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate(Module {
                symbols,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Frontend is summed: 10 + 20 = 30
        assert!((result.cost - 30.0).abs() < f64::EPSILON);
        assert_eq!(result.path, vec![target_id("my-pkg", "lib")]);
        assert_eq!(result.target_count, 1);
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
            "my-pkg".to_string(),
            make_crate(Module {
                symbols,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        assert!((result.cost - 170.0).abs() < f64::EPSILON);
    }

    #[test]
    #[expect(
        clippy::similar_names,
        reason = "mod_a/mod_b are intentionally parallel"
    )]
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
        root_symbols
            .insert("root_fn".to_string(), make_symbol_split(5.0, 10.0, &[]));

        let mut mod_a_symbols = HashMap::new();
        mod_a_symbols
            .insert("a_fn".to_string(), make_symbol_split(5.0, 100.0, &[]));

        let mut mod_b_symbols = HashMap::new();
        mod_b_symbols
            .insert("b_fn".to_string(), make_symbol_split(5.0, 50.0, &[]));

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
            "my-pkg".to_string(),
            make_crate(Module {
                symbols: root_symbols,
                submodules,
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // frontend = 15, backend = max(10, 100, 50) = 100
        assert!((result.cost - 115.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_target_overhead_included() {
        // Target with metadata overhead.
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0, &[]));

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate_with_overhead(
                Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                3.0,
            ),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // frontend(10) + backend(0) + metadata(3) = 13
        assert!((result.cost - 13.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_target() {
        // One target with two symbols: total cost = 30
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0, &[]));
        symbols.insert(
            "bar".to_string(),
            make_symbol(20.0, &[&dep("my-pkg", "lib", "foo")]),
        );

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate(Module {
                symbols,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Single target, so critical path = total cost = 30
        assert!((result.cost - 30.0).abs() < f64::EPSILON);
        assert_eq!(result.path, vec![target_id("my-pkg", "lib")]);
        assert_eq!(result.target_count, 1);
        assert_eq!(result.symbol_count, 2);
    }

    #[test]
    fn test_two_independent_targets() {
        // Two independent targets: can build in parallel
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a.insert("foo".to_string(), make_symbol(100.0, &[]));
        crates.insert(
            "pkg-a".to_string(),
            make_crate(Module {
                symbols: symbols_a,
                submodules: HashMap::new(),
            }),
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert("bar".to_string(), make_symbol(50.0, &[]));
        crates.insert(
            "pkg-b".to_string(),
            make_crate(Module {
                symbols: symbols_b,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Critical path is max(100, 50) = 100
        assert!((result.cost - 100.0).abs() < f64::EPSILON);
        // Total cost is 150 (sequential)
        assert!((result.total_cost - 150.0).abs() < f64::EPSILON);
        // Parallelism ratio = 150/100 = 1.5x
        assert_eq!(result.target_count, 2);
    }

    #[test]
    fn test_target_chain() {
        // pkg-a/lib (100) depends on pkg-b/lib (50)
        // Critical path = 150
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a.insert(
            "foo".to_string(),
            make_symbol(100.0, &[&dep("pkg-b", "lib", "bar")]),
        );
        crates.insert(
            "pkg-a".to_string(),
            make_crate(Module {
                symbols: symbols_a,
                submodules: HashMap::new(),
            }),
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert("bar".to_string(), make_symbol(50.0, &[]));
        crates.insert(
            "pkg-b".to_string(),
            make_crate(Module {
                symbols: symbols_b,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Critical path = pkg-b/lib + pkg-a/lib = 150
        assert!((result.cost - 150.0).abs() < f64::EPSILON);
        assert_eq!(
            result.path,
            vec![target_id("pkg-b", "lib"), target_id("pkg-a", "lib")]
        );
        // Total cost = 150 (no parallelism possible)
        assert!((result.total_cost - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diamond_targets() {
        // pkg-a/lib depends on pkg-b/lib (100) and pkg-c/lib (50)
        // pkg-b/lib and pkg-c/lib both depend on pkg-d/lib (30)
        // Critical path: d(30) → b(100) → a(10) = 140
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a.insert(
            "a".to_string(),
            make_symbol(
                10.0,
                &[&dep("pkg-b", "lib", "b"), &dep("pkg-c", "lib", "c")],
            ),
        );
        crates.insert(
            "pkg-a".to_string(),
            make_crate(Module {
                symbols: symbols_a,
                submodules: HashMap::new(),
            }),
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert(
            "b".to_string(),
            make_symbol(100.0, &[&dep("pkg-d", "lib", "d")]),
        );
        crates.insert(
            "pkg-b".to_string(),
            make_crate(Module {
                symbols: symbols_b,
                submodules: HashMap::new(),
            }),
        );

        let mut symbols_c = HashMap::new();
        symbols_c.insert(
            "c".to_string(),
            make_symbol(50.0, &[&dep("pkg-d", "lib", "d")]),
        );
        crates.insert(
            "pkg-c".to_string(),
            make_crate(Module {
                symbols: symbols_c,
                submodules: HashMap::new(),
            }),
        );

        let mut symbols_d = HashMap::new();
        symbols_d.insert("d".to_string(), make_symbol(30.0, &[]));
        crates.insert(
            "pkg-d".to_string(),
            make_crate(Module {
                symbols: symbols_d,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Critical path = d(30) + b(100) + a(10) = 140
        assert!((result.cost - 140.0).abs() < f64::EPSILON);
        assert_eq!(
            result.path,
            vec![
                target_id("pkg-d", "lib"),
                target_id("pkg-b", "lib"),
                target_id("pkg-a", "lib")
            ]
        );
        // Total cost = 190
        assert!((result.total_cost - 190.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_graph() {
        let graph = SymbolGraph {
            packages: HashMap::new(),
        };
        let result = critical_path(&graph);

        assert!((result.cost).abs() < f64::EPSILON);
        assert!(result.path.is_empty());
        assert_eq!(result.target_count, 0);
        assert_eq!(result.symbol_count, 0);
    }

    #[test]
    fn test_lib_and_test_targets() {
        // Test that lib and test targets are handled separately.
        // pkg-a has both lib (100ms) and test (50ms) targets.
        // The test target depends on the lib target.
        // Critical path: lib(100) → test(50) = 150

        let mut lib_symbols = HashMap::new();
        lib_symbols.insert("lib_fn".to_string(), make_symbol(100.0, &[]));

        let mut test_symbols = HashMap::new();
        test_symbols.insert(
            "test_fn".to_string(),
            make_symbol(50.0, &[&dep("pkg-a", "lib", "lib_fn")]),
        );

        let mut targets = HashMap::new();
        targets.insert(
            "lib".to_string(),
            make_crate(Module {
                symbols: lib_symbols,
                submodules: HashMap::new(),
            }),
        );
        targets.insert(
            "test".to_string(),
            make_crate(Module {
                symbols: test_symbols,
                submodules: HashMap::new(),
            }),
        );

        let mut packages = HashMap::new();
        packages.insert("pkg-a".to_string(), Package { targets });

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        // Critical path: lib(100) → test(50) = 150
        assert!((result.cost - 150.0).abs() < f64::EPSILON);
        assert_eq!(
            result.path,
            vec![target_id("pkg-a", "lib"), target_id("pkg-a", "test")]
        );
        assert_eq!(result.target_count, 2);
    }

    #[test]
    #[expect(
        clippy::similar_names,
        reason = "lib_a/lib_b and pkg_a/pkg_b are intentional"
    )]
    fn test_dev_dependency_no_cycle() {
        // This test verifies that dev-dependencies don't create cycles
        // at the target level.
        //
        // Setup:
        // - pkg-a/lib (100ms)
        // - pkg-a/test (50ms) depends on pkg-a/lib and pkg-b/lib
        // - pkg-b/lib (30ms) depends on pkg-a/lib
        //
        // At the package level, this would look like a cycle:
        //   pkg-a → pkg-b → pkg-a (via dev-deps)
        //
        // At the target level, there's no cycle:
        //   pkg-a/lib → pkg-b/lib → pkg-a/test
        //   pkg-a/lib → pkg-a/test
        //
        // Critical path: pkg-a/lib(100) → pkg-b/lib(30) → pkg-a/test(50) = 180
        // (or pkg-a/lib → pkg-a/test = 150, but the longer path through pkg-b wins)

        // pkg-a/lib
        let mut lib_a_symbols = HashMap::new();
        lib_a_symbols.insert("lib_a_fn".to_string(), make_symbol(100.0, &[]));

        // pkg-a/test (depends on pkg-a/lib and pkg-b/lib)
        let mut test_a_symbols = HashMap::new();
        test_a_symbols.insert(
            "test_a_fn".to_string(),
            make_symbol(
                50.0,
                &[
                    &dep("pkg-a", "lib", "lib_a_fn"),
                    &dep("pkg-b", "lib", "lib_b_fn"),
                ],
            ),
        );

        // pkg-b/lib (depends on pkg-a/lib)
        let mut lib_b_symbols = HashMap::new();
        lib_b_symbols.insert(
            "lib_b_fn".to_string(),
            make_symbol(30.0, &[&dep("pkg-a", "lib", "lib_a_fn")]),
        );

        let mut pkg_a_targets = HashMap::new();
        pkg_a_targets.insert(
            "lib".to_string(),
            make_crate(Module {
                symbols: lib_a_symbols,
                submodules: HashMap::new(),
            }),
        );
        pkg_a_targets.insert(
            "test".to_string(),
            make_crate(Module {
                symbols: test_a_symbols,
                submodules: HashMap::new(),
            }),
        );

        let mut pkg_b_targets = HashMap::new();
        pkg_b_targets.insert(
            "lib".to_string(),
            make_crate(Module {
                symbols: lib_b_symbols,
                submodules: HashMap::new(),
            }),
        );

        let mut packages = HashMap::new();
        packages.insert(
            "pkg-a".to_string(),
            Package {
                targets: pkg_a_targets,
            },
        );
        packages.insert(
            "pkg-b".to_string(),
            Package {
                targets: pkg_b_targets,
            },
        );

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        // Should compute successfully (no cycle)
        assert!(!result.path.is_empty(), "path should not be empty");

        // Critical path: pkg-a/lib(100) → pkg-b/lib(30) → pkg-a/test(50) = 180
        assert!((result.cost - 180.0).abs() < f64::EPSILON);
        assert_eq!(
            result.path,
            vec![
                target_id("pkg-a", "lib"),
                target_id("pkg-b", "lib"),
                target_id("pkg-a", "test")
            ]
        );

        // Total cost = 100 + 30 + 50 = 180
        assert!((result.total_cost - 180.0).abs() < f64::EPSILON);
        assert_eq!(result.target_count, 3);
    }

    #[test]
    fn test_parse_target_from_path() {
        // Valid paths
        assert_eq!(
            parse_target_from_path("[my-pkg/lib]::foo::bar"),
            Some("my-pkg/lib".to_string())
        );
        assert_eq!(
            parse_target_from_path("[pkg/test]::tests::test_fn"),
            Some("pkg/test".to_string())
        );
        assert_eq!(
            parse_target_from_path("[pkg/bin/cli]::main"),
            Some("pkg/bin/cli".to_string())
        );

        // Invalid paths
        assert_eq!(parse_target_from_path("old_crate::foo"), None);
        assert_eq!(parse_target_from_path("[no_slash]::foo"), None);
        assert_eq!(parse_target_from_path("not_bracketed/lib::foo"), None);
        assert_eq!(parse_target_from_path(""), None);
    }
}
