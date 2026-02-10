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
use tarjanize_cost::{CostModel, ModelFit, model_fit};
use tarjanize_schemas::{
    Crate, Module, Package, Symbol, SymbolGraph, SymbolKind, TargetTimings,
    sum_event_times,
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

/// Tries to resolve a sub-symbol dependency path to a known container.
///
/// When a dependency path like `[pkg/lib]::{{impl}}[2]::parse` isn't in the
/// index, this strips trailing `::segment` suffixes one at a time to find
/// the closest ancestor that IS in the index. Only strips one level to avoid
/// creating spurious edges to unrelated ancestors.
///
/// Returns `None` if the path doesn't start with `[` (not a bracketed path)
/// or if stripping the last segment still doesn't resolve.
fn resolve_sub_symbol_path(
    index: &SymbolIndex<'_>,
    dep: &str,
) -> Option<usize> {
    // Only attempt resolution for bracketed paths (workspace-internal deps).
    if !dep.starts_with('[') {
        return None;
    }

    // Strip the last `::segment` and retry. Only strip once to avoid
    // over-generalizing (e.g., resolving to a module instead of a symbol).
    let parent = dep.rsplit_once("::")?;
    index.get_index(parent.0)
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
    /// Original target key (e.g., `"pkg/lib"`) for each symbol.
    /// Used to look up target-level metadata (e.g., `metadata_decode_*` costs)
    /// from the original graph when predicting synthetic target wall times.
    original_targets: Vec<String>,
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
                let target_id = format!("{package_name}/{target_key}");
                index.add_module(
                    package_name,
                    &target_id,
                    &crate_prefix,
                    &crate_data.root,
                );
            }
        }
        index
    }

    /// Recursively adds symbols from a module to the index.
    fn add_module(
        &mut self,
        crate_name: &str,
        target_id: &str,
        module_path: &str,
        module: &'a Module,
    ) {
        for (symbol_name, symbol) in &module.symbols {
            let path = format!("{module_path}::{symbol_name}");
            self.paths.insert(path);
            self.symbols.push(symbol);
            self.original_crates.push(crate_name.to_string());
            self.original_targets.push(target_id.to_string());
        }

        for (submodule_name, submodule) in &module.submodules {
            let submodule_path = format!("{module_path}::{submodule_name}");
            self.add_module(crate_name, target_id, &submodule_path, submodule);
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

    /// Returns the original target key (e.g., `"pkg/lib"`) for a symbol.
    fn get_original_target(&self, index: usize) -> &str {
        &self.original_targets[index]
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
    cost_model: Option<&CostModel>,
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
    let mut sub_symbol_resolved = 0usize;
    for from in 0..index.len() {
        for dep in &index.get_symbol(from).dependencies {
            if let Some(to) = index.get_index(dep) {
                graph.add_edge(NodeIndex::new(from), NodeIndex::new(to), ());
            } else if let Some(to) = resolve_sub_symbol_path(&index, dep) {
                // Sub-symbol path: the dependency points to an associated
                // item (e.g., `[pkg/lib]::{{impl}}[2]::parse`) but the
                // index only has the container (`[pkg/lib]::{{impl}}[2]`).
                //
                // TODO: The root cause is in extract.rs — `normalize_def_id`
                // doesn't collapse some associated items to their container.
                // This fallback resolves the edge by stripping trailing path
                // segments until we find a known symbol.
                graph.add_edge(NodeIndex::new(from), NodeIndex::new(to), ());
                sub_symbol_resolved += 1;
            }
        }
    }
    if sub_symbol_resolved > 0 {
        debug!(
            sub_symbol_resolved,
            "Resolved sub-symbol dependencies by stripping trailing segments"
        );
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
                // No effective dependents — this is a root, stays in its own set.
                //
                // TODO: Downward merging. ~62% of SCCs are roots with 0
                // dependents. A second pass should merge roots that have
                // exactly 1 dependency into that dependency's set, iterating
                // until stable. This would reduce the number of isolated
                // single-symbol crates that add per-crate overhead without
                // enabling any parallelism.
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
    build_output_graph(&index, set_to_symbols, symbol_graph, cost_model)
}

/// Builds the output `SymbolGraph` from grouped symbols.
///
/// This uses a two-pass approach:
/// 1. Compute all new paths (old path → new path mapping)
/// 2. Build the output graph, rewriting dependencies using the mapping
#[expect(
    clippy::too_many_lines,
    reason = "three-variable model setup adds unavoidable per-target computation"
)]
fn build_output_graph(
    index: &SymbolIndex<'_>,
    set_to_symbols: HashMap<u32, Vec<usize>>,
    original_graph: &SymbolGraph,
    cost_model: Option<&CostModel>,
) -> SymbolGraph {
    // Sort sets by ID for deterministic output.
    let mut sets: Vec<_> = set_to_symbols.into_iter().collect();
    sets.sort_by_key(|(id, _)| *id);

    // Track used crate names to avoid collisions.
    let mut used_names: HashSet<String> = HashSet::new();

    // Compute crate names for each set.
    let mut set_crate_data: Vec<(u32, String, Vec<usize>)> = Vec::new();
    for (set_id, symbol_indices) in sets {
        // Collect unique original crates contributing to this set.
        let original_crates_set: HashSet<&str> = symbol_indices
            .iter()
            .map(|&i| index.get_original_crate(i))
            .collect();
        let mut original_crates: Vec<&str> =
            original_crates_set.iter().copied().collect();
        original_crates.sort_unstable();

        let base_name = original_crates.join("-");
        let crate_name = if used_names.contains(&base_name) {
            format!("{base_name}-{set_id}")
        } else {
            base_name
        };
        used_names.insert(crate_name.clone());
        set_crate_data.push((set_id, crate_name, symbol_indices));
    }

    // Pass 1: Compute old path → new path mapping.
    let set_crate_names: Vec<_> = set_crate_data
        .iter()
        .map(|(id, name, indices)| (*id, name.clone(), indices.clone()))
        .collect();
    let path_mapping = compute_path_mapping(index, &set_crate_names);

    // Build mapping from symbol index → synthetic target name.
    // Used to derive synthetic target dependencies from symbol-level edges.
    //
    // We derive deps from the symbol graph (which is acyclic after SCC
    // condensation) rather than the original target-level deps. The original
    // target deps include dev-dependency edges (test→lib) that create cycles
    // when condensation cross-partitions test and lib symbols from different
    // packages.
    let mut symbol_to_synthetic: Vec<String> = vec![String::new(); index.len()];
    for (_set_id, crate_name, symbol_indices) in &set_crate_data {
        let target_name = format!("{crate_name}/synthetic");
        for &sym_idx in symbol_indices {
            symbol_to_synthetic[sym_idx].clone_from(&target_name);
        }
    }

    // Pre-compute per-target metadata and other (non-metadata) event sums.
    // Used by both the external CostModel path and the internal fallback.
    let target_meta: HashMap<String, f64> = original_graph
        .packages
        .iter()
        .flat_map(|(pkg, p)| {
            p.targets.iter().map(move |(tgt, crate_data)| {
                let target_id = format!("{pkg}/{tgt}");
                let meta: f64 = crate_data
                    .timings
                    .event_times_ms
                    .iter()
                    .filter(|(k, _)| k.starts_with("metadata_decode_"))
                    .map(|(_, v)| v)
                    .sum();
                (target_id, meta)
            })
        })
        .collect();

    let target_other: HashMap<String, f64> = original_graph
        .packages
        .iter()
        .flat_map(|(pkg, p)| {
            p.targets.iter().map(move |(tgt, crate_data)| {
                let target_id = format!("{pkg}/{tgt}");
                let other: f64 = crate_data
                    .timings
                    .event_times_ms
                    .iter()
                    .filter(|(k, _)| !k.starts_with("metadata_decode_"))
                    .map(|(_, v)| v)
                    .sum();
                (target_id, other)
            })
        })
        .collect();

    // Internal model fitting (only used when no external CostModel is
    // provided). Kept as the backward-compatible fallback path.
    let model = if cost_model.is_none() {
        let mut regression_data: Vec<Vec<f64>> = Vec::new();
        for (pkg, p) in &original_graph.packages {
            for (tgt, crate_data) in &p.targets {
                let wall = crate_data.timings.wall_time_ms;
                if wall <= 0.0 {
                    continue;
                }
                let target_id = format!("{pkg}/{tgt}");
                let attr = collect_symbol_attr(&crate_data.root);
                let meta = target_meta.get(&target_id).copied().unwrap_or(0.0);
                let other =
                    target_other.get(&target_id).copied().unwrap_or(0.0);
                regression_data.push(vec![attr, meta, other, wall]);
            }
        }
        model_fit(&regression_data)
    } else {
        None
    };

    // Pass 2: Build output graph using the mapping.
    // Each partition becomes a separate package. The target type is determined
    // from the original target type of the symbols (they all share the same type
    // since symbols from different target types can't form cycles).
    let mut packages = HashMap::new();
    for (_set_id, crate_name, symbol_indices) in set_crate_data {
        let root_module =
            build_module_tree(index, &symbol_indices, &path_mapping);

        // Sum per-symbol attributed event times for the synthetic target.
        let attr: f64 = symbol_indices
            .iter()
            .map(|&i| sum_event_times(&index.get_symbol(i).event_times_ms))
            .sum();

        // Collect constituent original targets for this partition.
        let constituent_targets: HashSet<&str> = symbol_indices
            .iter()
            .map(|&i| index.get_original_target(i))
            .collect();

        // Derive synthetic target dependencies from symbol-level edges.
        // For each symbol in this partition, check its dependencies — if any
        // point to symbols in a different partition, that partition's synthetic
        // target becomes a dependency. This is cycle-free by construction
        // because the symbol graph is a DAG after SCC condensation.
        //
        // External deps (symbols not in the index, e.g., std) are ignored
        // since they're outside the workspace and not in our graph.
        let my_target = format!("{crate_name}/synthetic");
        let synthetic_deps: HashSet<String> = symbol_indices
            .iter()
            .flat_map(|&sym_idx| index.get_symbol(sym_idx).dependencies.iter())
            .filter_map(|dep_path| {
                let dep_idx = index.get_index(dep_path)?;
                let dep_target = &symbol_to_synthetic[dep_idx];
                (dep_target != &my_target).then(|| dep_target.clone())
            })
            .collect();

        // Estimate meta/other for the synthetic crate using the
        // max-constituent heuristic: a synthetic crate inherits at least
        // as much metadata/other overhead as its largest constituent.
        //
        // TODO: Cost model inflation. The max-constituent heuristic assigns
        // the full overhead of the largest original target to every fragment,
        // even single-symbol crates. This inflates predicted wall times.
        // Should scale meta/other proportionally by symbol count relative
        // to the original target (e.g., a fragment with 10% of symbols
        // gets 10% of the meta/other overhead).
        let meta: f64 = constituent_targets
            .iter()
            .filter_map(|t| target_meta.get(*t))
            .copied()
            .fold(0.0_f64, f64::max);
        let other: f64 = constituent_targets
            .iter()
            .filter_map(|t| target_other.get(*t))
            .copied()
            .fold(0.0_f64, f64::max);

        // Predict wall time: external CostModel coefficients if provided,
        // otherwise internally fitted regression (or raw attr fallback).
        let wall_time_ms = if let Some(cm) = cost_model {
            cm.predict(attr, meta, other)
        } else {
            predict_wall_time(model.as_ref(), attr, meta, other)
        };
        let dependencies = synthetic_deps;

        let crate_data = Crate {
            timings: TargetTimings {
                wall_time_ms,
                ..Default::default()
            },
            root: root_module,
            dependencies,
        };

        let mut targets = HashMap::new();
        targets.insert("synthetic".to_string(), crate_data);
        packages.insert(crate_name, Package { targets });
    }

    SymbolGraph { packages }
}

/// Recursively sums all per-symbol `event_times_ms` values in a module tree.
///
/// This is the total attributed self-time for a target — the first predictor
/// in the two-variable regression model.
fn collect_symbol_attr(module: &Module) -> f64 {
    let mut total = 0.0;
    for symbol in module.symbols.values() {
        total += sum_event_times(&symbol.event_times_ms);
    }
    for submodule in module.submodules.values() {
        total += collect_symbol_attr(submodule);
    }
    total
}

/// Predicts wall time using the fitted regression model.
///
/// Falls back to the raw symbol attribution sum when no model is available
/// (e.g., insufficient profiling data for regression).
fn predict_wall_time(
    model: Option<&ModelFit>,
    attr: f64,
    meta: f64,
    other: f64,
) -> f64 {
    match model {
        Some(fit) => fit.model.predict(&[attr, meta, other]),
        None => attr,
    }
}

/// Computes a mapping from old symbol paths to new symbol paths.
///
/// This handles conflict resolution: if two symbols from different original
/// targets would have the same path in the new crate, they get placed in
/// `conflict_from_{package_target}` submodules (with `/` replaced by `_`).
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
            let original_target = index.get_original_target(symbol_idx);

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
                .push((symbol_idx, original_target.to_string()));
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
                // Conflict — use the full target ID (e.g., "pkg/lib")
                // with `/` sanitized to `_` for valid module names.
                for (symbol_idx, original_target) in &occurrences {
                    let old_path = index.get_path(*symbol_idx).to_string();

                    let conflict_module = format!(
                        "conflict_from_{}",
                        original_target.replace('/', "_")
                    );
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
/// Handles conflict detection: if two symbols from different original targets
/// would have the same path, they are placed in `conflict_from_{package_target}`
/// submodules (with `/` replaced by `_`).
/// Key for grouping symbols: (module path segments, symbol name).
type SymbolPathKey = (Vec<String>, String);

/// Value for symbol grouping: list of (symbol index, original target).
///
/// The target identifier uses the `"package/target"` format (e.g.,
/// `"mypkg/lib"`, `"mypkg/test"`) rather than just the package name.
/// This ensures symbols from different targets of the same package
/// (lib vs test) produce distinct conflict submodules.
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
        let original_target = index.get_original_target(symbol_idx);

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
            .push((symbol_idx, original_target.to_string()));
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
            // Conflict — use the full target ID (e.g., "pkg/lib")
            // with `/` sanitized to `_` for valid module names.
            for (symbol_idx, original_target) in &occurrences {
                let symbol = index.get_symbol(*symbol_idx);

                // Create path with conflict submodule.
                let mut conflict_path = module_path.clone();
                conflict_path.push(format!(
                    "conflict_from_{}",
                    original_target.replace('/', "_")
                ));

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
        event_times_ms: symbol.event_times_ms.clone(),
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
            event_times_ms: HashMap::new(),
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

    /// Helper to count total symbols across all packages (recursive).
    fn count_total_symbols(graph: &SymbolGraph) -> usize {
        fn count_module(module: &Module) -> usize {
            module.symbols.len()
                + module.submodules.values().map(count_module).sum::<usize>()
        }
        graph
            .packages
            .values()
            .flat_map(|pkg| pkg.targets.values())
            .map(|c| count_module(&c.root))
            .sum()
    }

    #[test]
    fn test_single_symbol_stays_in_crate() {
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph, None);

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
        let result = condense_and_partition(&symbol_graph, None);

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
        let result = condense_and_partition(&symbol_graph, None);

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
        let result = condense_and_partition(&symbol_graph, None);

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
        let result = condense_and_partition(&symbol_graph, None);

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

        let result = condense_and_partition(&symbol_graph, None);

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
        let result = condense_and_partition(&symbol_graph, None);

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
        let result = condense_and_partition(&symbol_graph, None);

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
        let result = condense_and_partition(&symbol_graph, None);

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
                            "timings": {},
                            "root": {
                                "symbols": {
                                    "A": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C", "[c/lib]::D"] },
                                    "B": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C"] },
                                    "C": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::D"] },
                                    "D": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" } }
                                }
                            }
                        }
                    }
                }
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph, None);

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
                            "timings": {},
                            "root": {
                                "symbols": {
                                    "A": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C", "[c/lib]::E"] },
                                    "B": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C"] },
                                    "C": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::D"] },
                                    "D": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" } },
                                    "E": { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::D"] }
                                }
                            }
                        }
                    }
                }
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph, None);

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
                            "timings": {},
                            "root": {
                                "symbols": {
                                    "A":      { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C", "[c/lib]::impl_D"] },
                                    "B":      { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Function" }, "dependencies": ["[c/lib]::C"] },
                                    "C":      { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Struct" }, "dependencies": ["[c/lib]::D"] },
                                    "D":      { "file": "", "event_times_ms": {"typeck": 1.0}, "module_def": { "kind": "Struct" } },
                                    "impl_D": { "file": "", "event_times_ms": {"typeck": 1.0}, "impl": { "name": "impl D", "anchors": ["[c/lib]::D"] }, "dependencies": ["[c/lib]::D"] }
                                }
                            }
                        }
                    }
                }
            }
        }"#).unwrap();

        let result = condense_and_partition(&graph, None);

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
    fn test_synthetic_wall_times_from_per_symbol_costs() {
        // Verify that merged crates get synthetic wall-clock times computed
        // from per-symbol costs:
        //   wall_time_ms = sum of all symbol event_times_ms
        //
        // Chain: a → b (single dependent, merges into one crate).
        // a: event_times_ms total=100
        // b: event_times_ms total=200
        //
        // Expected merged crate:
        //   wall_time_ms = 100 + 200 = 300
        let mut symbols = HashMap::new();
        symbols.insert(
            "a".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 100.0)]),
                dependencies: [path("my_crate", "b")].into_iter().collect(),
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
                event_times_ms: HashMap::from([("typeck".to_string(), 200.0)]),
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
        let result = condense_and_partition(&symbol_graph, None);

        // Chain merges into one crate.
        assert_eq!(result.packages.len(), 1);

        let pkg = result.packages.values().next().unwrap();
        let timings = &get_synthetic(pkg).timings;

        // wall_time_ms = sum of all symbol event_times_ms = 100 + 200 = 300
        assert!(
            (timings.wall_time_ms - 300.0).abs() < 0.01,
            "Expected wall_time_ms ~300, got {}",
            timings.wall_time_ms
        );
    }

    #[test]
    fn test_synthetic_wall_times_per_crate() {
        // Verify each single-symbol crate has wall_time_ms matching
        // the symbol's total event_times_ms.
        //
        // Fork: a and b both depend on c.
        // Result: {a}, {b}, {c} — three crates (c is a boundary).
        let mut symbols = HashMap::new();
        symbols.insert(
            "a".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 50.0)]),
                dependencies: [path("my_crate", "c")].into_iter().collect(),
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
                event_times_ms: HashMap::from([("typeck".to_string(), 80.0)]),
                dependencies: [path("my_crate", "c")].into_iter().collect(),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        symbols.insert(
            "c".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 30.0)]),
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
        let result = condense_and_partition(&symbol_graph, None);

        // Three crates: {a}, {b}, {c}.
        assert_eq!(result.packages.len(), 3);

        // Each single-symbol crate should have wall_time_ms matching the
        // symbol's total event_times_ms.
        for pkg in result.packages.values() {
            let timings = &get_synthetic(pkg).timings;
            let root = get_root(pkg);
            assert_eq!(root.symbols.len(), 1);

            let symbol = root.symbols.values().next().unwrap();
            assert!(
                (timings.wall_time_ms
                    - sum_event_times(&symbol.event_times_ms))
                .abs()
                    < 0.01,
                "wall_time_ms should match symbol total event_times_ms"
            );
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "test requires 5 packages with profiling data for regression"
    )]
    fn test_model_based_prediction_for_merged_crate() {
        // Verify that when enough original targets have wall-clock profiling
        // data, the three-variable regression model is fitted and used to
        // predict synthetic target wall times — not just raw symbol
        // attribution sums.
        //
        // Setup: 5 packages with a single lib target each.
        //   - pkg_a has symbols `a → b` (chain, will merge into one crate)
        //   - pkg_b, pkg_c, pkg_d, pkg_e each have one independent symbol
        //
        // All packages have profiling data that perfectly fits:
        //   wall_time = 2.0 * attr + 3.0 * meta + 1.5 * other
        //
        // After merge, the synthetic crate containing {a, b} should have
        // its wall_time_ms predicted by the model, not just attr sum.

        // Helper to make a package with one symbol and profiling data.
        fn make_profiled_package(
            symbol_name: &str,
            symbol_attr: f64,
            meta: f64,
            other: f64,
            deps: HashSet<String>,
        ) -> Package {
            let wall = 2.0 * symbol_attr + 3.0 * meta + 1.5 * other;
            let mut symbols = HashMap::new();
            symbols.insert(
                symbol_name.to_string(),
                Symbol {
                    file: "test.rs".to_string(),
                    event_times_ms: HashMap::from([(
                        "typeck".to_string(),
                        symbol_attr,
                    )]),
                    dependencies: deps,
                    kind: SymbolKind::ModuleDef {
                        kind: "Function".to_string(),
                        visibility: Visibility::Public,
                    },
                },
            );
            let crate_data = Crate {
                timings: TargetTimings {
                    wall_time_ms: wall,
                    event_times_ms: HashMap::from([
                        ("metadata_decode_entry_foo".to_string(), meta),
                        ("check_mod_type_wf".to_string(), other),
                    ]),
                },
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            };
            let mut targets = HashMap::new();
            targets.insert("lib".to_string(), crate_data);
            Package { targets }
        }

        // pkg_a: two symbols forming a chain (a → b), which will merge.
        let mut pkg_a_symbols = HashMap::new();
        pkg_a_symbols.insert(
            "a".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 100.0)]),
                dependencies: [path("pkg_a", "b")].into_iter().collect(),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        pkg_a_symbols.insert(
            "b".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 50.0)]),
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        let pkg_a_crate = Crate {
            timings: TargetTimings {
                // attr = 100 + 50 = 150, meta = 20, other = 10
                // wall = 2*150 + 3*20 + 1.5*10 = 375
                wall_time_ms: 375.0,
                event_times_ms: HashMap::from([
                    ("metadata_decode_entry_foo".to_string(), 20.0),
                    ("check_mod_type_wf".to_string(), 10.0),
                ]),
            },
            root: Module {
                symbols: pkg_a_symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        };
        let mut pkg_a_targets = HashMap::new();
        pkg_a_targets.insert("lib".to_string(), pkg_a_crate);
        let pkg_a = Package {
            targets: pkg_a_targets,
        };

        // Independent packages that provide regression data points.
        let pkg_b =
            make_profiled_package("x", 200.0, 10.0, 40.0, HashSet::new());
        let pkg_c =
            make_profiled_package("y", 80.0, 30.0, 20.0, HashSet::new());
        let pkg_d =
            make_profiled_package("z", 300.0, 5.0, 50.0, HashSet::new());
        let pkg_e =
            make_profiled_package("w", 120.0, 25.0, 30.0, HashSet::new());

        let mut packages = HashMap::new();
        packages.insert("pkg_a".to_string(), pkg_a);
        packages.insert("pkg_b".to_string(), pkg_b);
        packages.insert("pkg_c".to_string(), pkg_c);
        packages.insert("pkg_d".to_string(), pkg_d);
        packages.insert("pkg_e".to_string(), pkg_e);

        let symbol_graph = SymbolGraph { packages };
        let result = condense_and_partition(&symbol_graph, None);

        // pkg_a's two symbols merge into one crate; the other four stay
        // separate. Total = 5 packages in output.
        assert_eq!(result.packages.len(), 5);

        // Find the merged crate (contains both a and b).
        let merged = result
            .packages
            .values()
            .find(|pkg| {
                let root = get_root(pkg);
                root.symbols.len() == 2
            })
            .expect("should have a merged crate with 2 symbols");

        let timings = &get_synthetic(merged).timings;

        // The model fits wall = 2*attr + 3*meta + 1.5*other perfectly.
        // Merged crate: attr = 100 + 50 = 150, meta = max(20) = 20,
        //               other = max(10) = 10
        // Predicted wall = 2*150 + 3*20 + 1.5*10 = 375
        //
        // Without the model, wall would just be attr = 150. Verify we get
        // the model-based prediction instead.
        assert!(
            timings.wall_time_ms > 200.0,
            "Model-based prediction should be higher than raw attr (150); \
             got {}",
            timings.wall_time_ms
        );
        assert!(
            (timings.wall_time_ms - 375.0).abs() < 1.0,
            "Expected wall_time_ms ~375 from model \
             (2*150 + 3*20 + 1.5*10), got {}",
            timings.wall_time_ms
        );
    }

    #[test]
    fn test_cost_model_based_prediction() {
        // When an external CostModel is provided, synthetic crate wall
        // times should use CostModel.predict(attr, meta, other) with
        // the max-constituent heuristic for meta/other.
        //
        // Setup: fork graph where A and B both depend on C.
        // A → C, B → C. Result: {A}, {B}, {C} — three crates.
        //
        // CostModel: wall = 2·attr + 3·meta + 1·other
        //
        // For crate {C} (attr=30, meta=0, other=0 — no profiling data):
        //   wall = 2*30 + 3*0 + 1*0 = 60
        let model = CostModel {
            coeff_attr: 2.0,
            coeff_meta: 3.0,
            coeff_other: 1.0,
            r_squared: 0.99,
            inlier_threshold: 500.0,
        };

        let mut symbols = HashMap::new();
        symbols.insert(
            "a".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 50.0)]),
                dependencies: [path("my_crate", "c")].into_iter().collect(),
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
                event_times_ms: HashMap::from([("typeck".to_string(), 80.0)]),
                dependencies: [path("my_crate", "c")].into_iter().collect(),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        symbols.insert(
            "c".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                event_times_ms: HashMap::from([("typeck".to_string(), 30.0)]),
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
            Crate {
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            },
        );

        let symbol_graph = make_graph(crates);
        let result = condense_and_partition(&symbol_graph, Some(&model));

        // Three crates: {a}, {b}, {c}.
        assert_eq!(result.packages.len(), 3);

        // Find the crate containing 'c' and check its predicted wall time.
        // c has attr=30, meta=0, other=0 (no profiling data on test targets).
        // wall = 2*30 + 3*0 + 1*0 = 60
        for pkg in result.packages.values() {
            let root = get_root(pkg);
            if root.symbols.contains_key("c") {
                let timings = &get_synthetic(pkg).timings;
                assert!(
                    (timings.wall_time_ms - 60.0).abs() < 1.0,
                    "Expected wall_time_ms ~60 from CostModel, got {}",
                    timings.wall_time_ms
                );
            }
        }
    }

    #[test]
    fn test_synthetic_deps_from_symbol_edges() {
        // Verify that synthetic target dependencies are derived from
        // symbol-level edges, not original target-level deps.
        //
        // Fork: pkg_a and pkg_b both depend on pkg_c (three crates).
        // pkg_a also has a target-level dep on "ext-pkg/lib" which is NOT
        // backed by any symbol edge. It should NOT appear in synthetic deps.
        let graph: SymbolGraph = serde_json::from_str(
            r#"{
            "packages": {
                "pkg_a": { "targets": { "lib": {
                    "timings": {},
                    "dependencies": ["pkg_c/lib", "ext-pkg/lib"],
                    "root": { "symbols": {
                        "a": { "file": "", "event_times_ms": {"typeck": 50.0},
                               "module_def": {"kind": "Function"},
                               "dependencies": ["[pkg_c/lib]::c"] }
                    }}
                }}},
                "pkg_b": { "targets": { "lib": {
                    "timings": {},
                    "dependencies": ["pkg_c/lib"],
                    "root": { "symbols": {
                        "b": { "file": "", "event_times_ms": {"typeck": 40.0},
                               "module_def": {"kind": "Function"},
                               "dependencies": ["[pkg_c/lib]::c"] }
                    }}
                }}},
                "pkg_c": { "targets": { "lib": {
                    "timings": {},
                    "root": { "symbols": {
                        "c": { "file": "", "event_times_ms": {"typeck": 30.0},
                               "module_def": {"kind": "Function"} }
                    }}
                }}}
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph, None);

        // Fork: {a}, {b}, {c} — three crates.
        assert_eq!(result.packages.len(), 3);

        for (pkg_name, pkg) in &result.packages {
            if get_root(pkg).symbols.contains_key("a") {
                let deps = &get_synthetic(pkg).dependencies;
                // Symbol-level dep on [pkg_c/lib]::c → pkg_c/synthetic.
                assert!(
                    deps.contains("pkg_c/synthetic"),
                    "Should have dep on pkg_c/synthetic. \
                     Package {pkg_name} has deps: {deps:?}",
                );
                // External target dep (ext-pkg/lib) should NOT appear —
                // no symbol edge backs it.
                assert!(
                    !deps.contains("ext-pkg/lib"),
                    "External dep without symbol edge should not appear. \
                     Package {pkg_name} has deps: {deps:?}",
                );
            }
        }
    }

    #[test]
    fn test_sub_symbol_dependency_resolves_to_container() {
        // When a symbol depends on a sub-symbol path that isn't in the
        // index (e.g., `[pkg/lib]::{{impl}}[2]::parse`), but the parent
        // path `[pkg/lib]::{{impl}}[2]` IS in the index, the fallback
        // should create an edge to the parent container.
        //
        // Graph: A depends on {{impl}}[2]::parse, but only {{impl}}[2]
        // exists. The edge A → {{impl}}[2] should be created.
        let graph: SymbolGraph = serde_json::from_str(
            r#"{
            "packages": {
                "pkg": { "targets": { "lib": {
                    "timings": {},
                    "root": { "symbols": {
                        "A": { "file": "", "event_times_ms": {"typeck": 10.0},
                               "module_def": {"kind": "Function"},
                               "dependencies": ["[pkg/lib]::{{impl}}[2]::parse"] },
                        "{{impl}}[2]": { "file": "", "event_times_ms": {"typeck": 5.0},
                                         "impl": {"name": "impl Foo", "anchors": []} }
                    }}
                }}}
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph, None);

        // Both symbols should be present (A + {{impl}}[2]).
        let total_symbols: usize = result
            .packages
            .values()
            .map(|m| get_root(m).symbols.len())
            .sum();
        assert_eq!(total_symbols, 2);

        // The dependency from A to {{impl}}[2] should be resolved via
        // the sub-symbol fallback. Since it's a chain (A → {{impl}}[2]),
        // both should merge into one crate.
        assert_eq!(
            result.packages.len(),
            1,
            "A → {{{{impl}}}}[2] chain should merge into one crate"
        );
    }

    #[test]
    fn test_sub_symbol_fallback_does_not_over_strip() {
        // When stripping the last segment also doesn't resolve, the
        // fallback should NOT keep stripping. No spurious edges should
        // be created for completely unresolvable paths.
        //
        // A depends on [pkg/lib]::unknown_module::unknown_symbol.
        // Neither the full path nor [pkg/lib]::unknown_module exists.
        // B is independent.
        let graph: SymbolGraph = serde_json::from_str(
            r#"{
            "packages": {
                "pkg": { "targets": { "lib": {
                    "timings": {},
                    "root": { "symbols": {
                        "A": { "file": "", "event_times_ms": {"typeck": 10.0},
                               "module_def": {"kind": "Function"},
                               "dependencies": ["[pkg/lib]::unknown_module::unknown_symbol"] },
                        "B": { "file": "", "event_times_ms": {"typeck": 5.0},
                               "module_def": {"kind": "Function"} }
                    }}
                }}}
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph, None);

        // Both A and B are independent roots (no edges between them).
        // The unresolvable dep should be silently dropped, not create
        // a spurious edge.
        assert_eq!(
            result.packages.len(),
            2,
            "A and B should remain in separate crates (no spurious edge)"
        );
    }

    #[test]
    fn test_synthetic_deps_remapped_to_new_target_ids() {
        // Verify that dependencies between synthetic crates use new target
        // IDs ("pkg_b/synthetic"), not original ones ("pkg_b/lib").
        //
        // Three packages: pkg_a and pkg_d both depend on pkg_b (fork).
        // Result: {a}, {d}, {c} — three separate crates.
        // pkg_a's synthetic target should depend on "pkg_b/synthetic".
        let graph: SymbolGraph = serde_json::from_str(
            r#"{
            "packages": {
                "pkg_a": { "targets": { "lib": {
                    "timings": {},
                    "dependencies": ["pkg_b/lib"],
                    "root": { "symbols": {
                        "a": { "file": "", "event_times_ms": {"typeck": 50.0},
                               "module_def": {"kind": "Function"},
                               "dependencies": ["[pkg_b/lib]::c"] }
                    }}
                }}},
                "pkg_b": { "targets": { "lib": {
                    "timings": {},
                    "root": { "symbols": {
                        "c": { "file": "", "event_times_ms": {"typeck": 30.0},
                               "module_def": {"kind": "Function"} }
                    }}
                }}},
                "pkg_d": { "targets": { "lib": {
                    "timings": {},
                    "dependencies": ["pkg_b/lib"],
                    "root": { "symbols": {
                        "d": { "file": "", "event_times_ms": {"typeck": 40.0},
                               "module_def": {"kind": "Function"},
                               "dependencies": ["[pkg_b/lib]::c"] }
                    }}
                }}}
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph, None);

        // Fork: {a}, {d}, {c} — three crates.
        assert_eq!(result.packages.len(), 3);

        // Find the crate containing 'a' and verify its dep uses the new
        // target ID format.
        for (pkg_name, pkg) in &result.packages {
            if get_root(pkg).symbols.contains_key("a") {
                let deps = &get_synthetic(pkg).dependencies;
                assert!(
                    !deps.contains("pkg_b/lib"),
                    "Dep should be remapped from pkg_b/lib. \
                     Package {pkg_name} has deps: {deps:?}",
                );
                assert!(
                    deps.contains("pkg_b/synthetic"),
                    "Dep should be remapped to pkg_b/synthetic. \
                     Package {pkg_name} has deps: {deps:?}",
                );
            }
        }
    }

    /// Symbols from different targets (lib vs test) that end up in the same
    /// SCC must not be silently dropped during conflict resolution.
    ///
    /// Regression test: when lib and test targets both have a symbol at the
    /// same relative path, `build_module_tree` must disambiguate using the
    /// full target identifier (e.g., `pkg/lib` vs `pkg/test`) rather than
    /// just the package name.
    #[test]
    fn test_cross_target_symbols_not_dropped() {
        // Build a graph where one package has lib and test targets, each
        // containing a symbol with the same relative path "foo". A
        // cross-target dependency forces them into the same SCC.
        let graph: SymbolGraph = serde_json::from_str(
            r#"{
            "packages": {
                "mypkg": { "targets": {
                    "lib": {
                        "timings": {},
                        "root": { "symbols": {
                            "foo": { "file": "lib.rs", "event_times_ms": {},
                                     "module_def": {"kind": "Function"} }
                        }}
                    },
                    "test": {
                        "timings": {},
                        "dependencies": ["mypkg/lib"],
                        "root": { "symbols": {
                            "foo": { "file": "lib.rs", "event_times_ms": {},
                                     "module_def": {"kind": "Function"},
                                     "dependencies": ["[mypkg/lib]::foo"] }
                        }}
                    }
                }}
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph, None);

        // Both symbols must survive condensation (zero loss).
        assert_eq!(
            count_total_symbols(&result),
            2,
            "Both lib and test symbols must be preserved, got: {result:#?}"
        );
    }

    /// When cross-target conflicts are resolved, the conflict submodule names
    /// must use the full target identifier (with `/` sanitized to `_`) so that
    /// symbols from `pkg/lib` and `pkg/test` produce distinct submodules.
    #[test]
    fn test_cross_target_conflict_uses_target_name() {
        // Same setup as above: lib and test targets with duplicate "foo".
        let graph: SymbolGraph = serde_json::from_str(
            r#"{
            "packages": {
                "mypkg": { "targets": {
                    "lib": {
                        "timings": {},
                        "root": { "symbols": {
                            "foo": { "file": "lib.rs", "event_times_ms": {},
                                     "module_def": {"kind": "Function"} }
                        }}
                    },
                    "test": {
                        "timings": {},
                        "dependencies": ["mypkg/lib"],
                        "root": { "symbols": {
                            "foo": { "file": "lib.rs", "event_times_ms": {},
                                     "module_def": {"kind": "Function"},
                                     "dependencies": ["[mypkg/lib]::foo"] }
                        }}
                    }
                }}
            }
        }"#,
        )
        .unwrap();

        let result = condense_and_partition(&graph, None);

        // Find the package containing the merged symbols. Since they form
        // a cycle (test→lib via anchor back-edge), they'll be in one SCC.
        let mut found_lib_conflict = false;
        let mut found_test_conflict = false;

        for pkg in result.packages.values() {
            let root = get_root(pkg);
            // Conflict submodules should use target-qualified names.
            if root.submodules.contains_key("conflict_from_mypkg_lib") {
                found_lib_conflict = true;
            }
            if root.submodules.contains_key("conflict_from_mypkg_test") {
                found_test_conflict = true;
            }
        }

        assert!(
            found_lib_conflict,
            "Expected conflict_from_mypkg_lib submodule in output: {result:#?}"
        );
        assert!(
            found_test_conflict,
            "Expected conflict_from_mypkg_test submodule in output: {result:#?}"
        );
    }
}
