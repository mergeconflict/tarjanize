//! SCC computation, union-find merging, and optimized graph construction.
//!
//! This module implements the core Phase 2 algorithm:
//! 1. Build a directed graph and compute SCCs via petgraph's condensation
//! 2. Merge SCCs into optimal crate groupings using union-find
//! 3. Fix anchor constraints for orphan rule compliance
//! 4. Build output `SymbolGraph` with new crate structure

use std::collections::{HashMap, HashSet};

use indexmap::IndexSet;
use petgraph::algo::condensation;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::unionfind::UnionFind;
use tarjanize_schemas::{
    Crate, Module, Package, Symbol, SymbolGraph, SymbolKind,
};
use tracing::debug;

/// Extracts impl anchors from a symbol kind, if it's an impl.
fn impl_anchors(kind: &SymbolKind) -> Option<&HashSet<String>> {
    match kind {
        SymbolKind::Impl { anchors, .. } => Some(anchors),
        SymbolKind::ModuleDef { .. } => None,
    }
}

/// Parse a symbol path in the `[package/target]::module::symbol` format.
///
/// Returns `(prefix, rest)` where:
/// - `prefix` is the bracketed portion like `[package/target]`
/// - `rest` is the remaining path like `module::symbol`
///
/// If the path doesn't start with `[`, returns `None` (old format path).
fn parse_bracketed_path(path: &str) -> Option<(&str, &str)> {
    if !path.starts_with('[') {
        return None;
    }
    // Find the closing bracket.
    let bracket_end = path.find(']')?;
    let prefix = &path[..=bracket_end];
    // The rest should start with `::`
    let rest = path.get(bracket_end + 1..)?.strip_prefix("::")?;
    Some((prefix, rest))
}

/// Applies HMR (Habib-Morvan-Rampon) transitive reduction to a set of deps.
///
/// Given a starting reachable set and a list of dependencies, returns the
/// non-redundant dependencies and the final reachable set. A dependency is
/// redundant if it's already reachable through another (non-redundant) dep.
///
/// Dependencies are processed in reverse condensation order (higher indices
/// first, i.e., closer to roots) so that deps with larger reachable sets are
/// processed first, maximizing redundancy elimination.
fn hmr_reduce(
    initial_reachable: HashSet<usize>,
    deps: &[usize],
    scc_reachable: &[HashSet<usize>],
) -> (HashSet<usize>, HashSet<usize>) {
    let mut reachable = initial_reachable;
    let mut non_redundant = HashSet::new();

    // Sort deps by reverse condensation order (higher index = closer to roots).
    let mut sorted_deps = deps.to_vec();
    sorted_deps.sort_unstable_by(|a, b| b.cmp(a));

    for dep in sorted_deps {
        if !reachable.contains(&dep) {
            // Not yet reachable - this edge is not redundant.
            non_redundant.insert(dep);
            reachable.extend(scc_reachable[dep].iter().copied());
        }
    }

    (non_redundant, reachable)
}

/// Indexes into a `SymbolGraph`.
///
/// Symbols and their paths are indexed by their insertion order; when we build
/// the index, we traverse the module tree and insert each symbol and its path
/// into `symbols` and `paths` respectively.
#[derive(Default)]
struct SymbolIndex<'a> {
    /// Symbols indexed by position.
    symbols: Vec<&'a Symbol>,
    /// Paths indexed by position. Also provides reverse lookup.
    paths: IndexSet<String>,
    /// Original crate name for each symbol.
    original_crates: Vec<String>,
}

impl<'a> SymbolIndex<'a> {
    /// Builds an index by walking the entire `SymbolGraph`.
    ///
    /// Iterates over all packages and their targets, indexing symbols
    /// with paths like `[package-name/target]::module::symbol`.
    ///
    /// The path format uses brackets to delimit the package/target portion,
    /// which makes parsing unambiguous (brackets don't appear in Rust paths).
    fn build(symbol_graph: &'a SymbolGraph) -> Self {
        let mut index = SymbolIndex::default();
        for (package_name, package) in &symbol_graph.packages {
            for (target_key, crate_data) in &package.targets {
                // Paths use [package/target]::module::symbol format.
                // This matches the format used by the orchestrator for dependencies.
                let crate_prefix = format!("[{package_name}/{target_key}]");
                index.add_module(package_name, &crate_prefix, &crate_data.root);
            }
        }
        index
    }

    /// Recursively adds symbols from a module to the index.
    fn add_module(
        &mut self,
        crate_name: &str,
        module_path: &str,
        module: &'a Module,
    ) {
        for (symbol_name, symbol) in &module.symbols {
            let path = format!("{module_path}::{symbol_name}");
            self.paths.insert(path);
            self.symbols.push(symbol);
            self.original_crates.push(crate_name.to_string());
        }

        for (submodule_name, submodule) in &module.submodules {
            let submodule_path = format!("{module_path}::{submodule_name}");
            self.add_module(crate_name, &submodule_path, submodule);
        }
    }

    /// Returns the index for a path, if it exists.
    fn get_index(&self, path: &str) -> Option<usize> {
        self.paths.get_index_of(path)
    }

    /// Returns the path at the given index.
    fn get_path(&self, index: usize) -> &str {
        self.paths.get_index(index).expect("valid index")
    }

    /// Returns the symbol at the given index.
    fn get_symbol(&self, index: usize) -> &'a Symbol {
        self.symbols[index]
    }

    /// Returns the original crate name for a symbol.
    fn get_original_crate(&self, index: usize) -> &str {
        &self.original_crates[index]
    }

    /// Returns the number of symbols in the index.
    fn len(&self) -> usize {
        self.symbols.len()
    }
}

/// Condenses the symbol graph and partitions it into optimal crate groupings.
///
/// This is the main entry point for Phase 2. It:
/// 1. Builds a directed graph from the symbol graph
/// 2. Computes SCCs using petgraph's condensation
/// 3. Merges SCCs using union-find based on the merge criterion
/// 4. Fixes anchor constraints for orphan rule compliance
/// 5. Builds and returns the optimized `SymbolGraph`
#[expect(
    clippy::cast_possible_truncation,
    reason = "SCC count limited by symbol count; real workspaces have far fewer than 2^32 symbols"
)]
#[expect(
    clippy::too_many_lines,
    reason = "Complex algorithm with many steps; splitting would obscure the flow"
)]
pub(crate) fn condense_and_partition(
    symbol_graph: &SymbolGraph,
) -> SymbolGraph {
    // Handle empty graph case.
    if symbol_graph.packages.is_empty() {
        return SymbolGraph {
            packages: HashMap::new(),
        };
    }

    // Step 1: Build symbol index for O(1) lookups.
    let index = SymbolIndex::build(symbol_graph);
    debug!(symbol_count = index.len(), "Built symbol index");

    if index.len() == 0 {
        return SymbolGraph {
            packages: HashMap::new(),
        };
    }

    // Step 2: Build DiGraph with symbol indices as node weights.
    let mut graph = DiGraph::<usize, ()>::with_capacity(index.len(), 0);
    for i in 0..index.len() {
        graph.add_node(i);
    }
    for from in 0..index.len() {
        for dep in &index.get_symbol(from).dependencies {
            if let Some(to) = index.get_index(dep) {
                graph.add_edge(NodeIndex::new(from), NodeIndex::new(to), ());
            }
        }
    }

    // Step 2b: Add synthetic back-edges for anchor constraints.
    // For each impl, add an edge from one of its anchors back to the impl.
    // This creates a cycle that condensation will collapse into a single SCC,
    // ensuring the impl and anchor end up in the same crate (orphan rule).
    //
    // TODO: Investigate better heuristics for choosing which anchor to use.
    // Currently we use minimum in-degree as a rough proxy for "niche" — fewer
    // things depend on it, so merging with it affects fewer other SCCs. But
    // ideally we'd use transitive dependents (requires computing reachability).
    let mut back_edge_count = 0;
    for impl_idx in 0..index.len() {
        if let Some(anchors) = impl_anchors(&index.get_symbol(impl_idx).kind) {
            // Find valid anchors (those that exist in the index).
            let valid_anchors: Vec<usize> =
                anchors.iter().filter_map(|p| index.get_index(p)).collect();

            if valid_anchors.is_empty() {
                continue;
            }

            // Choose anchor with minimum in-degree (most "niche").
            let best_anchor = valid_anchors
                .iter()
                .copied()
                .min_by_key(|&anchor_idx| {
                    graph
                        .neighbors_directed(
                            NodeIndex::new(anchor_idx),
                            petgraph::Direction::Incoming,
                        )
                        .count()
                })
                .expect("valid_anchors is non-empty");

            // Add back-edge: anchor → impl (creates cycle impl → ... → anchor → impl).
            graph.add_edge(
                NodeIndex::new(best_anchor),
                NodeIndex::new(impl_idx),
                (),
            );
            back_edge_count += 1;
        }
    }
    debug!(
        back_edge_count,
        "Added synthetic back-edges for anchor constraints"
    );

    // Step 3: Run condensation. Returns DiGraph<Vec<usize>, ()>.
    // make_acyclic=true removes self-loops and deduplicates edges between SCCs.
    // We only need to know whether dependencies exist, not their multiplicity.
    let condensed = condensation(graph, true);
    let scc_count = condensed.node_count();
    debug!(scc_count, "Computed SCCs");

    // Step 4: Precompute reachable sets for each SCC using HMR algorithm.
    // For each SCC, reachable[scc] is the set of all SCCs reachable from it.
    // We process in reverse topo order (leaves first) so dependencies are
    // computed before dependents.
    //
    // Condensation returns nodes in reverse topo order (sinks/leaves first),
    // which is exactly what we need for computing reachable sets.
    let mut scc_reachable: Vec<HashSet<usize>> =
        (0..scc_count).map(|i| HashSet::from([i])).collect();

    // Build adjacency list (SCC dependencies).
    let scc_deps: Vec<Vec<usize>> = (0..scc_count)
        .map(|i| {
            condensed
                .neighbors(NodeIndex::new(i))
                .map(NodeIndex::index)
                .collect()
        })
        .collect();

    // Build reverse adjacency list (SCC dependents).
    let mut scc_dependents: Vec<Vec<usize>> = vec![Vec::new(); scc_count];
    for (from, deps) in scc_deps.iter().enumerate() {
        for &to in deps {
            scc_dependents[to].push(from);
        }
    }

    // Process in forward order (leaves first in condensation order).
    // For each SCC, compute reachable set by HMR: process dependencies in topo
    // order, keeping only non-redundant edges.
    for scc_id in 0..scc_count {
        let mut deps: Vec<usize> = scc_deps[scc_id].clone();
        // Sort dependencies by topo order (lower index = earlier in condensation
        // = closer to leaves). This ensures we process closer deps first.
        deps.sort_unstable();

        for dep in deps {
            if !scc_reachable[scc_id].contains(&dep) {
                // Not yet reachable - this edge is not redundant.
                // Add all SCCs reachable from dep to our reachable set.
                let dep_reachable = scc_reachable[dep].clone();
                scc_reachable[scc_id].extend(dep_reachable);
            }
            // If dep is already reachable, the edge scc_id→dep is redundant.
        }
    }
    debug!("Computed SCC reachable sets");

    // Step 5: Build symbol_to_scc mapping and log SCC contents.
    let mut symbol_to_scc = vec![0u32; index.len()];
    for node_idx in condensed.node_indices() {
        let scc_id = node_idx.index() as u32;
        let symbols: Vec<&str> = condensed[node_idx]
            .iter()
            .map(|&i| index.get_path(i))
            .collect();
        debug!(scc_id, ?symbols, "SCC contents");
        for &symbol_idx in &condensed[node_idx] {
            symbol_to_scc[symbol_idx] = scc_id;
        }
    }

    // Step 6: Initialize union-find and set-level tracking structures.
    // petgraph's UnionFind uses path compression (find_mut) and union-by-rank.
    let mut uf: UnionFind<u32> = UnionFind::new(scc_count);

    // For each set (initially each SCC is its own set), track:
    // - set_external_deps: external dependencies (SCCs outside the set), after
    //   removing redundant edges via HMR
    // - set_reachable: reachable SCCs from this set (for HMR redundancy checks)
    // We use the representative's index as the key.
    //
    // Initialize by applying HMR to each SCC's dependencies to remove redundant
    // edges. An edge scc→dep is redundant if dep is reachable through some other
    // dependency.
    let mut set_external_deps: Vec<HashSet<usize>> = scc_deps
        .iter()
        .enumerate()
        .map(|(scc_id, deps)| {
            let (non_redundant, _) =
                hmr_reduce(HashSet::from([scc_id]), deps, &scc_reachable);
            non_redundant
        })
        .collect();
    let mut set_reachable: Vec<HashSet<usize>> = scc_reachable.clone();

    // Step 7: Process SCCs in reverse order (dependents before dependencies).
    // petgraph's condensation returns SCCs in postorder (dependencies first),
    // so we iterate in reverse to get dependents first.
    // We use incremental HMR: after each merge, we recompute the set's external
    // deps and reachable set, removing edges that become redundant at set level.
    debug!("=== Dependent-based merging (using incremental HMR) ===");
    for scc_id in (0..scc_count).rev() {
        // Get the symbols in this SCC for logging.
        let scc_symbols: Vec<&str> = condensed[NodeIndex::new(scc_id)]
            .iter()
            .map(|&i| index.get_path(i))
            .collect();

        // Get all dependents of this SCC.
        let all_dependents = &scc_dependents[scc_id];

        // Find the effective dependent sets: sets where the edge to this SCC
        // is not redundant (i.e., scc_id is in that set's external_deps).
        let effective_dependent_sets: HashSet<u32> = all_dependents
            .iter()
            .filter_map(|&dep| {
                let dep_set = uf.find_mut(dep as u32);
                let dep_set_idx = dep_set as usize;
                // Check if this set has a non-redundant edge to scc_id.
                set_external_deps[dep_set_idx]
                    .contains(&scc_id)
                    .then_some(dep_set)
            })
            .collect();

        match effective_dependent_sets.len() {
            0 => {
                // No effective dependents - this is a root, stays in its own set.
                debug!(
                    scc_id,
                    ?scc_symbols,
                    "Root SCC (no effective dependents), stays in own set"
                );
            }
            1 => {
                // All effective dependents in same set - merge this SCC into
                // that set, then update the set's external deps and reachable.
                let target_set =
                    *effective_dependent_sets.iter().next().unwrap();
                uf.union(scc_id as u32, target_set);

                // Get the actual representative after the union (might differ
                // from target_set due to union-by-rank).
                let new_rep = uf.find_mut(scc_id as u32);
                let new_rep_idx = new_rep as usize;

                // If the representative changed, move the old set's data.
                let target_set_idx = target_set as usize;
                if new_rep != target_set {
                    set_external_deps[new_rep_idx] =
                        std::mem::take(&mut set_external_deps[target_set_idx]);
                    set_reachable[new_rep_idx] =
                        std::mem::take(&mut set_reachable[target_set_idx]);
                }

                // Update set tracking: merge this SCC into the target set.
                // Add this SCC's external deps (except those internal to the
                // set), then recompute reachable using HMR.
                set_external_deps[new_rep_idx].remove(&scc_id);
                for &dep in &scc_deps[scc_id] {
                    // Only add if external to the merged set.
                    if uf.find_mut(dep as u32) != new_rep {
                        set_external_deps[new_rep_idx].insert(dep);
                    }
                }

                // Recompute non-redundant external deps via HMR.
                // Start with the SCCs in this merged set as the initial reachable.
                let sccs_in_set: HashSet<usize> = (0..scc_count)
                    .filter(|&s| uf.find_mut(s as u32) == new_rep)
                    .collect();
                let all_deps: Vec<usize> =
                    set_external_deps[new_rep_idx].iter().copied().collect();
                let (new_external_deps, new_reachable) =
                    hmr_reduce(sccs_in_set, &all_deps, &scc_reachable);

                set_external_deps[new_rep_idx] = new_external_deps;
                set_reachable[new_rep_idx] = new_reachable;

                debug!(
                    scc_id,
                    ?scc_symbols,
                    target_set,
                    external_deps = ?set_external_deps[new_rep_idx],
                    "Merged into dependent's set"
                );
            }
            _ => {
                // Effective dependents in different sets - this is a boundary.
                debug!(
                    scc_id,
                    ?scc_symbols,
                    ?effective_dependent_sets,
                    "Boundary SCC (effective dependents in multiple sets), stays separate"
                );
            }
        }
    }

    // Step 8: Group symbols by union-find set.
    debug!("=== Final partition assignments ===");
    let mut set_to_symbols: HashMap<u32, Vec<usize>> = HashMap::new();
    for (symbol_idx, &scc_id) in symbol_to_scc.iter().enumerate() {
        let set_id = uf.find_mut(scc_id);
        set_to_symbols.entry(set_id).or_default().push(symbol_idx);
    }

    // Log each partition's contents.
    for (&set_id, symbol_indices) in &set_to_symbols {
        let symbols: Vec<&str> =
            symbol_indices.iter().map(|&i| index.get_path(i)).collect();
        debug!(set_id, ?symbols, "Final partition");
    }
    debug!(
        crate_count = set_to_symbols.len(),
        "Grouped symbols into crates"
    );

    // Step 9: Build output SymbolGraph.
    build_output_graph(&index, set_to_symbols, symbol_graph)
}

/// Coefficient for estimating metadata time from frontend cost.
///
/// Based on linear regression against real build data (R² = 0.705):
/// `metadata_ms = METADATA_SLOPE * frontend_ms + METADATA_INTERCEPT`
///
/// See `docs/cost-model-validation.md` Appendix A.7 for derivation.
const METADATA_SLOPE: f64 = 0.26;

/// Fixed per-crate overhead for metadata generation in milliseconds.
///
/// This represents the baseline cost of generating metadata for any crate,
/// independent of its symbol count or complexity.
const METADATA_INTERCEPT: f64 = 1662.0;

/// Builds the output `SymbolGraph` from grouped symbols.
///
/// This uses a two-pass approach:
/// 1. Compute all new paths (old path → new path mapping)
/// 2. Build the output graph, rewriting dependencies using the mapping
///
/// Crate-level overhead (`metadata_ms`) is estimated from total frontend cost
/// using a linear model (R² = 0.705). See `docs/cost-model-validation.md`.
fn build_output_graph(
    index: &SymbolIndex<'_>,
    set_to_symbols: HashMap<u32, Vec<usize>>,
    _original_graph: &SymbolGraph,
) -> SymbolGraph {
    // Sort sets by ID for deterministic output.
    let mut sets: Vec<_> = set_to_symbols.into_iter().collect();
    sets.sort_by_key(|(id, _)| *id);

    // Track used crate names to avoid collisions.
    let mut used_names: HashSet<String> = HashSet::new();

    // Compute crate names and overhead for each set.
    let mut set_crate_data: Vec<(u32, String, Vec<usize>, f64)> = Vec::new();
    for (set_id, symbol_indices) in sets {
        // Collect unique original crates contributing to this set.
        let original_crates_set: HashSet<&str> = symbol_indices
            .iter()
            .map(|&i| index.get_original_crate(i))
            .collect();
        let mut original_crates: Vec<&str> =
            original_crates_set.iter().copied().collect();
        original_crates.sort_unstable();

        // Estimate metadata time from total frontend cost of symbols in this crate.
        // Frontend cost is the best single predictor of metadata time (R² = 0.705).
        let total_frontend_ms: f64 = symbol_indices
            .iter()
            .map(|&i| index.get_symbol(i).frontend_cost_ms)
            .sum();
        let metadata_ms =
            METADATA_SLOPE * total_frontend_ms + METADATA_INTERCEPT;

        let base_name = original_crates.join("-");
        let crate_name = if used_names.contains(&base_name) {
            format!("{base_name}-{set_id}")
        } else {
            base_name
        };
        used_names.insert(crate_name.clone());
        set_crate_data.push((set_id, crate_name, symbol_indices, metadata_ms));
    }

    // Pass 1: Compute old path → new path mapping.
    let set_crate_names: Vec<_> = set_crate_data
        .iter()
        .map(|(id, name, indices, _)| (*id, name.clone(), indices.clone()))
        .collect();
    let path_mapping = compute_path_mapping(index, &set_crate_names);

    // Pass 2: Build output graph using the mapping.
    // Each partition becomes a separate package. The target type is determined
    // from the original target type of the symbols (they all share the same type
    // since symbols from different target types can't form cycles).
    let mut packages = HashMap::new();
    for (_set_id, crate_name, symbol_indices, metadata_ms) in set_crate_data {
        let root_module =
            build_module_tree(index, &symbol_indices, &path_mapping);
        let crate_data = Crate {
            metadata_ms,
            root: root_module,
            // Dependencies for synthetic crates would need to be computed
            // from symbol dependencies if needed for downstream analysis.
            ..Default::default()
        };

        let mut targets = HashMap::new();
        targets.insert("synthetic".to_string(), crate_data);
        packages.insert(crate_name, Package { targets });
    }

    SymbolGraph { packages }
}

/// Computes a mapping from old symbol paths to new symbol paths.
///
/// This handles conflict resolution: if two symbols from different original
/// crates would have the same path in the new crate, they get placed in
/// `conflict_from_{original_crate}` submodules.
fn compute_path_mapping(
    index: &SymbolIndex<'_>,
    set_crate_names: &[(u32, String, Vec<usize>)],
) -> HashMap<String, String> {
    let mut mapping = HashMap::new();

    for (_set_id, crate_name, symbol_indices) in set_crate_names {
        // Group symbols by their relative path (module path + symbol name).
        let mut path_to_symbols: HashMap<SymbolPathKey, SymbolOccurrences> =
            HashMap::new();

        for &symbol_idx in symbol_indices {
            let full_path = index.get_path(symbol_idx);
            let original_crate = index.get_original_crate(symbol_idx);

            // Parse path in new format: [package/target]::module::symbol
            let rest = if let Some((_, rest)) = parse_bracketed_path(full_path)
            {
                rest
            } else {
                // Fallback to old format: crate::module::symbol
                full_path.split_once("::").map_or(full_path, |(_, r)| r)
            };

            let parts: Vec<&str> = rest.split("::").collect();
            if parts.is_empty() {
                continue;
            }

            // Last part is symbol name, everything else is module path.
            let symbol_name = parts[parts.len() - 1].to_string();
            let module_parts: Vec<String> = parts[..parts.len() - 1]
                .iter()
                .map(|s| (*s).to_string())
                .collect();

            path_to_symbols
                .entry((module_parts, symbol_name))
                .or_default()
                .push((symbol_idx, original_crate.to_string()));
        }

        // Compute new paths, handling conflicts.
        // Output paths use the same bracketed format as input: [package/synthetic]::path
        for ((module_path, symbol_name), occurrences) in path_to_symbols {
            if occurrences.len() == 1 {
                // No conflict - symbol keeps its relative path in the new crate.
                let (symbol_idx, _) = &occurrences[0];
                let old_path = index.get_path(*symbol_idx).to_string();

                let new_path = if module_path.is_empty() {
                    format!("[{crate_name}/synthetic]::{symbol_name}")
                } else {
                    format!(
                        "[{}/synthetic]::{}::{}",
                        crate_name,
                        module_path.join("::"),
                        symbol_name
                    )
                };

                mapping.insert(old_path, new_path);
            } else {
                // Conflict - each symbol goes into a conflict submodule.
                for (symbol_idx, original_crate) in &occurrences {
                    let old_path = index.get_path(*symbol_idx).to_string();

                    let conflict_module =
                        format!("conflict_from_{original_crate}");
                    let new_path = if module_path.is_empty() {
                        format!(
                            "[{crate_name}/synthetic]::{conflict_module}::{symbol_name}"
                        )
                    } else {
                        format!(
                            "[{}/synthetic]::{}::{conflict_module}::{symbol_name}",
                            crate_name,
                            module_path.join("::")
                        )
                    };

                    mapping.insert(old_path, new_path);
                }
            }
        }
    }

    mapping
}

/// Builds a module tree from a list of symbol indices.
///
/// Handles conflict detection: if two symbols from different original crates
/// would have the same path, they are placed in `conflict_from_{original_crate}`
/// submodules.
/// Key for grouping symbols: (module path segments, symbol name).
type SymbolPathKey = (Vec<String>, String);

/// Value for symbol grouping: list of (symbol index, original crate).
type SymbolOccurrences = Vec<(usize, String)>;

fn build_module_tree(
    index: &SymbolIndex<'_>,
    symbol_indices: &[usize],
    path_mapping: &HashMap<String, String>,
) -> Module {
    // Group symbols by their module path (relative to the new crate).
    let mut path_to_symbols: HashMap<SymbolPathKey, SymbolOccurrences> =
        HashMap::new();

    for &symbol_idx in symbol_indices {
        let full_path = index.get_path(symbol_idx);
        let original_crate = index.get_original_crate(symbol_idx);

        // Parse path in new format: [package/target]::module::symbol
        let rest = if let Some((_, rest)) = parse_bracketed_path(full_path) {
            rest
        } else {
            // Fallback to old format: crate::module::symbol
            full_path.split_once("::").map_or(full_path, |(_, r)| r)
        };

        let parts: Vec<&str> = rest.split("::").collect();
        if parts.is_empty() {
            continue; // Invalid path
        }

        // Last part is symbol name, everything else is module path.
        let symbol_name = parts[parts.len() - 1].to_string();
        let module_parts: Vec<String> = parts[..parts.len() - 1]
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        path_to_symbols
            .entry((module_parts, symbol_name))
            .or_default()
            .push((symbol_idx, original_crate.to_string()));
    }

    // Build the module tree, handling conflicts.
    let mut root = Module {
        symbols: HashMap::new(),
        submodules: HashMap::new(),
    };

    for ((module_path, symbol_name), occurrences) in path_to_symbols {
        if occurrences.len() == 1 {
            // No conflict - place symbol at its original path.
            let (symbol_idx, _) = &occurrences[0];
            let symbol = index.get_symbol(*symbol_idx);
            let target_module = get_or_create_module(&mut root, &module_path);
            let new_symbol = rewrite_symbol(symbol, path_mapping);
            target_module.symbols.insert(symbol_name, new_symbol);
        } else {
            // Conflict - place each symbol in a conflict submodule.
            for (symbol_idx, original_crate) in &occurrences {
                let symbol = index.get_symbol(*symbol_idx);

                // Create path with conflict submodule.
                let mut conflict_path = module_path.clone();
                conflict_path.push(format!("conflict_from_{original_crate}"));

                let target_module =
                    get_or_create_module(&mut root, &conflict_path);
                let new_symbol = rewrite_symbol(symbol, path_mapping);
                target_module
                    .symbols
                    .insert(symbol_name.clone(), new_symbol);
            }
        }
    }

    root
}

/// Rewrites a symbol's paths (dependencies and anchors) using the path mapping.
fn rewrite_symbol(
    symbol: &Symbol,
    path_mapping: &HashMap<String, String>,
) -> Symbol {
    let rewrite = |path: &String| -> String {
        path_mapping
            .get(path)
            .cloned()
            .unwrap_or_else(|| path.clone())
    };

    let new_dependencies = symbol.dependencies.iter().map(rewrite).collect();

    let new_kind = match &symbol.kind {
        SymbolKind::ModuleDef { kind, visibility } => SymbolKind::ModuleDef {
            kind: kind.clone(),
            visibility: *visibility,
        },
        SymbolKind::Impl { name, anchors } => SymbolKind::Impl {
            name: name.clone(),
            anchors: anchors.iter().map(rewrite).collect(),
        },
    };

    Symbol {
        file: symbol.file.clone(),
        frontend_cost_ms: symbol.frontend_cost_ms,
        backend_cost_ms: symbol.backend_cost_ms,
        dependencies: new_dependencies,
        kind: new_kind,
    }
}

/// Gets or creates a nested module at the given path.
fn get_or_create_module<'a>(
    root: &'a mut Module,
    path: &[String],
) -> &'a mut Module {
    let mut current = root;
    for segment in path {
        current =
            current
                .submodules
                .entry(segment.clone())
                .or_insert_with(|| Module {
                    symbols: HashMap::new(),
                    submodules: HashMap::new(),
                });
    }
    current
}

#[cfg(test)]
mod tests {
    use tarjanize_schemas::Visibility;

    use super::*;

    /// Helper to create a path in the new `[package/target]::symbol` format.
    ///
    /// For test crates, we use the package name as both package and crate name,
    /// with "lib" as the default target.
    fn path(package: &str, symbol: &str) -> String {
        format!("[{package}/lib]::{symbol}")
    }

    /// Helper to create a simple symbol for testing.
    fn make_symbol(deps: &[&str]) -> Symbol {
        Symbol {
            file: "test.rs".to_string(),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: deps.iter().map(|&s| s.to_string()).collect(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        }
    }

    /// Helper to create a crate with default overhead for testing.
    fn make_crate(symbols: HashMap<String, Symbol>) -> Crate {
        Crate {
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        }
    }

    /// Creates a package with a single "lib" target containing the given crate.
    fn make_package(crate_data: Crate) -> Package {
        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), crate_data);
        Package { targets }
    }

    /// Creates a `SymbolGraph` from a map of package names to crates.
    fn make_graph(crates: HashMap<String, Crate>) -> SymbolGraph {
        let packages = crates
            .into_iter()
            .map(|(name, crate_data)| (name, make_package(crate_data)))
            .collect();
        SymbolGraph { packages }
    }

    /// Helper to get all synthetic crates from packages (for test assertions).
    /// Returns a map from package name to crate data.
    #[expect(dead_code, reason = "available for future tests")]
    fn get_synthetic_crates(graph: &SymbolGraph) -> HashMap<&str, &Crate> {
        graph
            .packages
            .iter()
            .filter_map(|(name, pkg)| {
                pkg.targets.get("synthetic").map(|c| (name.as_str(), c))
            })
            .collect()
    }

    /// Helper to get the synthetic crate from a package (condense output).
    fn get_synthetic(pkg: &Package) -> &tarjanize_schemas::Crate {
        pkg.targets
            .get("synthetic")
            .expect("expected synthetic target")
    }

    /// Helper to get the root module from a package (assumes single target).
    fn get_root(pkg: &Package) -> &Module {
        &get_synthetic(pkg).root
    }

    /// Helper to count total symbols across all packages.
    #[expect(dead_code, reason = "utility function for future tests")]
    fn count_total_symbols(graph: &SymbolGraph) -> usize {
        graph
            .packages
            .values()
            .flat_map(|pkg| pkg.targets.values())
            .map(|c| c.root.symbols.len())
            .sum()
    }

    #[test]
    fn test_single_symbol_stays_in_crate() {
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Single symbol stays in its original crate.
        assert_eq!(result.packages.len(), 1);
        assert!(result.packages.contains_key("my_crate"));
    }

    #[test]
    fn test_chain_merges_into_one_crate() {
        // a → b → c (chain with single dependents)
        // All should merge into one crate.
        let mut symbols = HashMap::new();
        symbols.insert("a".to_string(), make_symbol(&[&path("my_crate", "b")]));
        symbols.insert("b".to_string(), make_symbol(&[&path("my_crate", "c")]));
        symbols.insert("c".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Chain should merge into one crate.
        assert_eq!(result.packages.len(), 1);

        // All three symbols should be in that crate.
        let crate_module = result.packages.values().next().unwrap();
        assert_eq!(get_root(crate_module).symbols.len(), 3);
    }

    #[test]
    fn test_fork_creates_multiple_crates() {
        // A and B both depend on C.
        // A and B are independent roots.
        // C → D (chain).
        // Result: {A}, {B}, {C, D} - three crates.
        let mut symbols = HashMap::new();
        symbols.insert("a".to_string(), make_symbol(&[&path("my_crate", "c")]));
        symbols.insert("b".to_string(), make_symbol(&[&path("my_crate", "c")]));
        symbols.insert("c".to_string(), make_symbol(&[&path("my_crate", "d")]));
        symbols.insert("d".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Should have 3 crates: {A}, {B}, {C, D}.
        assert_eq!(result.packages.len(), 3);

        // Count total symbols across all crates.
        let total_symbols: usize = result
            .packages
            .values()
            .map(|m| get_root(m).symbols.len())
            .sum();
        assert_eq!(total_symbols, 4);
    }

    #[test]
    fn test_diamond_merges_into_one() {
        // Diamond: A depends on B and C, both B and C depend on D.
        // All should merge into one crate.
        let mut symbols = HashMap::new();
        symbols.insert(
            "a".to_string(),
            make_symbol(&[&path("my_crate", "b"), &path("my_crate", "c")]),
        );
        symbols.insert("b".to_string(), make_symbol(&[&path("my_crate", "d")]));
        symbols.insert("c".to_string(), make_symbol(&[&path("my_crate", "d")]));
        symbols.insert("d".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Diamond should merge into one crate.
        assert_eq!(result.packages.len(), 1);

        let crate_module = result.packages.values().next().unwrap();
        assert_eq!(get_root(crate_module).symbols.len(), 4);
    }

    #[test]
    fn test_cycle_forms_single_scc() {
        // foo and bar form a cycle - they must stay together.
        let mut symbols = HashMap::new();
        // foo depends on bar, bar depends on foo (cycle).
        symbols.insert(
            "foo".to_string(),
            make_symbol(&[&path("my_crate", "bar")]),
        );
        symbols.insert(
            "bar".to_string(),
            make_symbol(&[&path("my_crate", "foo")]),
        );

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Cycle = one SCC = one crate.
        assert_eq!(result.packages.len(), 1);

        let crate_module = result.packages.values().next().unwrap();
        assert_eq!(get_root(crate_module).symbols.len(), 2);
    }

    #[test]
    fn test_empty_graph() {
        let symbol_graph = SymbolGraph {
            packages: HashMap::new(),
        };

        let result = condense_and_partition(&symbol_graph);

        assert!(result.packages.is_empty());
    }

    #[test]
    fn test_external_dependent_preserves_parallelism() {
        // Diamond with external E depending on D.
        // A depends on B and C, B and C depend on D, E also depends on D.
        // Result: {A, B, C}, {D}, {E} - D stays separate.
        let mut symbols = HashMap::new();
        symbols.insert(
            "a".to_string(),
            make_symbol(&[&path("my_crate", "b"), &path("my_crate", "c")]),
        );
        symbols.insert("b".to_string(), make_symbol(&[&path("my_crate", "d")]));
        symbols.insert("c".to_string(), make_symbol(&[&path("my_crate", "d")]));
        symbols.insert("d".to_string(), make_symbol(&[]));
        symbols.insert("e".to_string(), make_symbol(&[&path("my_crate", "d")]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Should have 3 crates: {A, B, C}, {D}, {E}.
        assert_eq!(result.packages.len(), 3);
    }

    #[test]
    fn test_dependencies_rewritten_to_new_paths() {
        // A depends on C, B depends on C.
        // Result: {A}, {B}, {C} - three crates.
        // A's dependency on C should be rewritten to the new crate's path.
        let mut symbols = HashMap::new();
        symbols.insert("a".to_string(), make_symbol(&[&path("my_crate", "c")]));
        symbols.insert("b".to_string(), make_symbol(&[&path("my_crate", "c")]));
        symbols.insert("c".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Should have 3 crates.
        assert_eq!(result.packages.len(), 3);

        // Find the crate containing 'a' and verify its dependency on 'c'.
        let mut found_a = false;
        for module in result.packages.values() {
            if let Some(symbol_a) = get_root(module).symbols.get("a") {
                found_a = true;
                // The dependency should be rewritten to [my_crate/synthetic]::c
                // (same crate name since all symbols came from my_crate).
                assert!(
                    symbol_a.dependencies.contains("[my_crate/synthetic]::c"),
                    "Expected dependency on [my_crate/synthetic]::c, got {:?}",
                    symbol_a.dependencies
                );
            }
        }
        assert!(found_a, "Symbol 'a' not found in result");
    }

    #[test]
    fn test_cross_crate_dependencies_rewritten() {
        // Two crates: crate_a has 'foo' depending on 'bar' in crate_b.
        // foo → bar (chain), so they merge into one crate named "crate_a-crate_b".
        // foo's dependency should be rewritten from "[crate_b/lib]::bar" to
        // "[crate_a-crate_b/synthetic]::bar".
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a
            .insert("foo".to_string(), make_symbol(&[&path("crate_b", "bar")]));
        crates.insert("crate_a".to_string(), make_crate(symbols_a));

        let mut symbols_b = HashMap::new();
        symbols_b.insert("bar".to_string(), make_symbol(&[]));
        crates.insert("crate_b".to_string(), make_crate(symbols_b));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Should merge into one crate since it's a simple chain.
        assert_eq!(result.packages.len(), 1);

        // The merged crate name should combine both original crate names.
        let (crate_name, module) = result.packages.iter().next().unwrap();
        assert!(
            crate_name.contains("crate_a") && crate_name.contains("crate_b"),
            "Expected merged crate name, got {crate_name}"
        );

        // foo's dependency should now point to the new path with bracketed format.
        let foo = get_root(module)
            .symbols
            .get("foo")
            .expect("foo should exist");
        assert_eq!(foo.dependencies.len(), 1);
        let dep = foo.dependencies.iter().next().unwrap();
        // Path format: [package/synthetic]::symbol
        assert!(
            dep.starts_with(&format!("[{crate_name}/synthetic]")),
            "Dependency should start with [{crate_name}/synthetic], got {dep}"
        );
        assert!(dep.ends_with("::bar"), "Dependency should end with ::bar");
    }

    #[test]
    fn test_redundant_dependency_should_not_create_boundary() {
        // Graph with a redundant dependency edge:
        //
        //         B
        //         │
        //         ↓
        //  A ───→ C ───→ D
        //  │             ↑
        //  └─────────────┘
        //
        // A depends on both C and D, but C also depends on D. The A→D edge
        // is redundant: A cannot compile until C is built, and C cannot
        // compile until D is built.
        //
        // Without effective dependents, D would be a boundary (dependents A, C
        // in different sets), giving 4 crates: {A}, {B}, {C}, {D}.
        //
        // With effective dependents, A is dominated by C (since A→*C), so
        // effective_dependents(D) = {C}. D merges into C's set, giving 3
        // crates: {A}, {B}, {C, D}.
        let graph: SymbolGraph = serde_json::from_str(
            r#"{
            "packages": {
                "c": {
                    "targets": {
                        "lib": {
                            "metadata_ms": 0,
                            "root": {
                                "symbols": {
                                    "A": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C", "[c/lib]::D"] },
                                    "B": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C"] },
                                    "C": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::D"] },
                                    "D": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" } }
                                }
                            }
                        }
                    }
                }
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph);

        // Expected: {A}, {B}, {C, D}
        let mut partitions: Vec<Vec<&str>> = result
            .packages
            .values()
            .map(|m| {
                let mut syms: Vec<_> =
                    get_root(m).symbols.keys().map(String::as_str).collect();
                syms.sort_unstable();
                syms
            })
            .collect();
        partitions.sort();
        assert_eq!(partitions, vec![vec!["A"], vec!["B"], vec!["C", "D"]]);
    }

    #[test]
    fn test_set_level_transitive_reduction() {
        // Graph where redundancy only becomes apparent after merging:
        //
        //            B
        //            │
        //            ↓
        //     A ───→ C ───→ D
        //     │             ↑
        //     └───→ E ──────┘
        //
        // A depends on C and E. B depends on C. C depends on D. E depends on D.
        //
        // Unlike test_redundant_dependency_should_not_create_boundary, A does NOT
        // directly depend on D. The transitive reduction on individual SCCs keeps
        // all edges (no redundant edges at the SCC level).
        //
        // The key insight is that the transitive reduction changes every time we
        // union an SCC into another set. Before merging E into A, there are no
        // redundant edges. After merging, the set {A,E} has edges to both C and D,
        // and {A,E}→D becomes redundant because {A,E}→C→D exists.
        //
        // What we need is **incremental transitive reduction**: as we merge SCCs,
        // we must update our view of which edges are redundant at the set level.
        //
        // Processing order (dependents first): A, B, E, C, D
        //
        // | Step | SCC | Dependents   | Their sets   | Action       | Sets after          |
        // |------|-----|--------------|--------------|--------------|---------------------|
        // | 1    | A   | none         | —            | root         | {A}, {B}, {E}, {C}, {D} |
        // | 2    | B   | none         | —            | root         | {A}, {B}, {E}, {C}, {D} |
        // | 3    | E   | A            | {A}          | merge into A | {A,E}, {B}, {C}, {D}    |
        // | 4    | C   | A, B         | {A,E}, {B}   | boundary     | {A,E}, {B}, {C}, {D}    |
        // | 5    | D   | C, E         | {C}, {A,E}   | boundary     | {A,E}, {B}, {C}, {D}    |
        //
        // Current result: {A, E}, {B}, {C}, {D} — 4 partitions
        //
        // But this is suboptimal! After E merges into A, the set {A,E} has edges to
        // both C and D. At the SET level, {A,E}→D is redundant because {A,E}→C→D
        // exists (A→C and C→D). So D's only effective dependent SET is {C}.
        //
        // With set-level transitive reduction:
        // | Step | SCC | Dependent sets | Reduced dependents | Action       | Sets after       |
        // |------|-----|----------------|--------------------|--------------| -----------------|
        // | 5    | D   | {C}, {A,E}     | {C} only           | merge into C | {A,E}, {B}, {C,D}|
        //
        // Expected result: {A, E}, {B}, {C, D} — 3 partitions
        let graph: SymbolGraph = serde_json::from_str(
            r#"{
            "packages": {
                "c": {
                    "targets": {
                        "lib": {
                            "metadata_ms": 0,
                            "root": {
                                "symbols": {
                                    "A": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C", "[c/lib]::E"] },
                                    "B": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C"] },
                                    "C": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::D"] },
                                    "D": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" } },
                                    "E": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::D"] }
                                }
                            }
                        }
                    }
                }
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph);

        // Expected: {A, E}, {B}, {C, D} — 3 partitions
        let mut partitions: Vec<Vec<&str>> = result
            .packages
            .values()
            .map(|m| {
                let mut syms: Vec<_> =
                    get_root(m).symbols.keys().map(String::as_str).collect();
                syms.sort_unstable();
                syms
            })
            .collect();
        partitions.sort();
        assert_eq!(partitions, vec![vec!["A", "E"], vec!["B"], vec!["C", "D"]]);
    }

    #[test]
    fn test_anchor_constraints_handled_by_back_edges() {
        // Test that anchor constraints are satisfied via synthetic back-edges.
        //
        // Graph (arrows point from dependent to dependency):
        //
        //            B
        //            │
        //            ↓
        //     A ───→ C ───→ D
        //     │             ↑
        //     └── impl_D ───┘ (anchor: D)
        //
        // Step 2b adds a synthetic back-edge D → impl_D, creating a cycle.
        // Condensation groups {D, impl_D} into one SCC.
        //
        // After condensation: {A}, {B}, {C}, {D, impl_D}
        //
        // Dependent-based merging:
        // - A's deps: {C}, {D,impl_D}. HMR: {D,impl_D} reachable via C, so redundant.
        // - A's effective deps: {C}
        // - {D,impl_D}'s effective dependents: only {C}
        // - {D,impl_D} merges into {C}
        //
        // Final: {A}, {B}, {C, D, impl_D}
        let graph: SymbolGraph = serde_json::from_str(r#"{
            "packages": {
                "c": {
                    "targets": {
                        "lib": {
                            "metadata_ms": 0,
                            "root": {
                                "symbols": {
                                    "A":      { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C", "[c/lib]::impl_D"] },
                                    "B":      { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C"] },
                                    "C":      { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Struct" }, "dependencies": ["[c/lib]::D"] },
                                    "D":      { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "module_def": { "kind": "Struct" } },
                                    "impl_D": { "file": "", "frontend_cost_ms": 1, "backend_cost_ms": 0, "impl": { "name": "impl D", "anchors": ["[c/lib]::D"] }, "dependencies": ["[c/lib]::D"] }
                                }
                            }
                        }
                    }
                }
            }
        }"#).unwrap();

        let result = condense_and_partition(&graph);

        assert_eq!(
            result
                .packages
                .values()
                .map(|m| get_root(m).symbols.len())
                .sum::<usize>(),
            5,
            "Should have all 5 symbols"
        );
        // Expected partitioning: {A}, {B}, {C, D, impl_D}
        let mut partitions: Vec<Vec<&str>> = result
            .packages
            .values()
            .map(|m| {
                let mut syms: Vec<_> =
                    get_root(m).symbols.keys().map(String::as_str).collect();
                syms.sort_unstable();
                syms
            })
            .collect();
        partitions.sort();
        assert_eq!(
            partitions,
            vec![vec!["A"], vec!["B"], vec!["C", "D", "impl_D"]]
        );
    }

    #[test]
    fn test_metadata_estimated_from_frontend_cost() {
        // Test that metadata is estimated from frontend cost using the formula:
        // metadata_ms = 0.26 * frontend_ms + 1662
        //
        // Two independent symbols with different frontend costs become separate
        // crates, each with metadata estimated from its frontend cost.
        let mut symbols = HashMap::new();
        symbols.insert(
            "a".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                frontend_cost_ms: 1000.0, // Expected: 0.26 * 1000 + 1662 = 1922
                backend_cost_ms: 0.0,
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        symbols.insert(
            "b".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                frontend_cost_ms: 5000.0, // Expected: 0.26 * 5000 + 1662 = 2962
                backend_cost_ms: 0.0,
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            tarjanize_schemas::Crate {
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            },
        );

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph);

        // Two independent symbols → two crates.
        assert_eq!(result.packages.len(), 2);

        // Find crates by their symbol content.
        let crate_a = result
            .packages
            .values()
            .find(|c| get_root(c).symbols.contains_key("a"))
            .expect("crate with symbol a");
        let crate_b = result
            .packages
            .values()
            .find(|c| get_root(c).symbols.contains_key("b"))
            .expect("crate with symbol b");

        // Verify metadata estimated from formula: 0.26 * frontend + 1662
        let expected_a = METADATA_SLOPE * 1000.0 + METADATA_INTERCEPT; // 1922
        let expected_b = METADATA_SLOPE * 5000.0 + METADATA_INTERCEPT; // 2962

        let metadata_a = get_synthetic(crate_a).metadata_ms;
        let metadata_b = get_synthetic(crate_b).metadata_ms;

        assert!(
            (metadata_a - expected_a).abs() < 0.01,
            "Expected metadata_ms ~{expected_a}, got {metadata_a}"
        );
        assert!(
            (metadata_b - expected_b).abs() < 0.01,
            "Expected metadata_ms ~{expected_b}, got {metadata_b}"
        );
    }
}
