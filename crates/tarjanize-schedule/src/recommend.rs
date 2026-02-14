//! Horizon-based shatter helpers and cost distribution logic.
//!
//! Orchestrates shatter operations that split a target into horizon groups,
//! building both updated schedule data and symbol graphs. Delegates horizon
//! computation to `crate::horizon` and grouping/cost distribution to
//! `crate::grouping`.
//!
//! Why: shatter analysis needs horizon-aware grouping and cost reallocation
//! to keep schedules meaningful after synthetic splits.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use tarjanize_schemas::{
    CostModel, Module, SymbolGraph, TargetId, TargetTimings,
};

use crate::data::ScheduleData;
// Re-export `collect_symbol_external_targets` so external callers
// (e.g. `tarjanize-viz`) that import it from `recommend::` still compile.
pub use crate::grouping::collect_symbol_external_targets;
use crate::grouping::{
    ShatterGroups, ShatteredTargetLists, adjust_group_symbol_counts,
    build_shatter_groups, compute_group_stats,
};
use crate::horizon::compute_effective_horizons;
use crate::schedule::{TargetGraph, compute_schedule};
use crate::target_graph::{IntraTargetGraph, condense_target};

/// Holds shared inputs for building shattered graphs.
///
/// Why: consolidates repeated parameters to avoid clippy argument limits.
struct ShatterBuilder<'a> {
    symbol_graph: &'a SymbolGraph,
    schedule: &'a ScheduleData,
    target_id: &'a TargetId,
    target_prefix: String,
    intra: &'a IntraTargetGraph,
    groups: &'a ShatterGroups<'a>,
}

impl<'a> ShatterBuilder<'a> {
    /// Creates a builder for a single target shatter operation.
    ///
    /// Why: centralizes derived identifiers and shared references.
    fn new(
        symbol_graph: &'a SymbolGraph,
        schedule: &'a ScheduleData,
        target_id: &'a TargetId,
        intra: &'a IntraTargetGraph,
        groups: &'a ShatterGroups<'a>,
    ) -> Self {
        let target_prefix = format!("[{target_id}]::");
        Self {
            symbol_graph,
            schedule,
            target_id,
            target_prefix,
            intra,
            groups,
        }
    }

    /// Builds the `TargetGraph` for a horizon-shattered target.
    ///
    /// Constructs the full target graph with all original targets (except
    /// the shattered one) plus one target per horizon group. Wires edges
    /// between surviving targets, inter-group edges from the SCC DAG,
    /// external deps per group, and dependent rewiring.
    ///
    /// Why: schedule recomputation needs a graph that preserves dependencies
    /// while replacing the shattered target with group nodes.
    fn build_shattered_target_graph(
        &self,
        group_costs: &[Duration],
        group_sym_counts: &[usize],
        group_ext_deps: &[HashSet<String>],
    ) -> Option<TargetGraph> {
        use petgraph::graph::DiGraph;

        let group_count = self.groups.group_count;
        debug_assert_eq!(
            group_costs.len(),
            group_count,
            "group_costs must cover every horizon group"
        );
        debug_assert_eq!(
            group_sym_counts.len(),
            group_count,
            "group_sym_counts must cover every horizon group"
        );
        debug_assert_eq!(
            group_ext_deps.len(),
            group_count,
            "group_ext_deps must cover every horizon group"
        );

        let target_idx = self
            .schedule
            .targets
            .iter()
            .position(|t| t.name == self.target_id.as_str())?;
        let ShatteredTargetLists {
            names,
            costs,
            symbol_counts,
            old_to_new,
            group_base,
        } = self.build_shattered_target_lists(
            target_idx,
            group_costs,
            group_sym_counts,
            group_count,
        );

        self.assert_shatter_symbol_counts(
            target_idx,
            group_count,
            &names,
            group_sym_counts,
            &symbol_counts,
        );

        let total = names.len();
        let mut graph = DiGraph::<usize, ()>::with_capacity(total, 0);
        for i in 0..total {
            graph.add_node(i);
        }

        self.wire_shattered_target_edges(
            &mut graph,
            group_base,
            group_count,
            &old_to_new,
            group_ext_deps,
            &names,
        );

        Some(TargetGraph {
            names,
            costs,
            symbol_counts,
            graph,
        })
    }

    /// Builds a modified `SymbolGraph` reflecting a horizon-shattered target.
    ///
    /// Clones the original graph, then replaces the shattered target with
    /// one target per horizon group. Each group target contains only the
    /// symbols from its member SCCs, moved from the original target's module
    /// tree using the `export` helpers. Target-level event timings are
    /// pre-distributed per group and applied directly. The original target
    /// is removed.
    ///
    /// This ensures `/api/tree/{pkg}/{target}::group_N` resolves correctly
    /// after shattering.
    ///
    /// Why: UI and export logic need a consistent symbol graph after shatter.
    fn build_shattered_symbol_graph(
        &self,
        group_costs: &[Duration],
        group_event_times: &[HashMap<String, Duration>],
    ) -> SymbolGraph {
        use crate::export::{
            insert_symbol_into_module, parse_symbol_path,
            remove_symbol_from_module,
        };

        let group_count = self.groups.group_count;
        debug_assert_eq!(
            group_costs.len(),
            group_count,
            "group_costs must cover every horizon group"
        );
        debug_assert_eq!(
            group_event_times.len(),
            group_count,
            "group_event_times must cover every horizon group"
        );

        let mut graph = self.symbol_graph.clone();

        let pkg_name = self.target_id.package();
        let target_key = self.target_id.target();

        // Create empty group targets up front.
        let mut group_roots: Vec<Module> =
            (0..group_count).map(|_| Module::default()).collect();

        // Move symbols from the original target into group module trees.
        let Some(pkg) = graph.packages.get_mut(pkg_name) else {
            return graph;
        };
        let Some(source_target) = pkg.targets.get_mut(target_key) else {
            return graph;
        };

        for node in &self.intra.nodes {
            let group = self.groups.scc_to_group[node.id];
            for sym_path in &node.symbols {
                let Some((mod_segments, sym_name)) =
                    parse_symbol_path(sym_path, &self.target_prefix)
                else {
                    continue;
                };
                // Remove from original target and insert into group target.
                if let Some(symbol) = remove_symbol_from_module(
                    &mut source_target.root,
                    &mod_segments,
                    &sym_name,
                ) {
                    insert_symbol_into_module(
                        &mut group_roots[group],
                        &mod_segments,
                        sym_name,
                        symbol,
                    );
                }
            }
        }

        let original_deps = source_target.dependencies.clone();
        pkg.targets.remove(target_key);

        for (g, root) in group_roots.into_iter().enumerate() {
            let group_key = format!("{target_key}::group_{g}");

            // Use the pre-distributed per-event timings for this group.
            let event_times = group_event_times[g].clone();

            let group_target = tarjanize_schemas::Target {
                timings: TargetTimings {
                    wall_time: group_costs[g],
                    event_times_ms: event_times,
                },
                dependencies: original_deps.clone(),
                root,
            };
            pkg.targets.insert(group_key, group_target);
        }

        graph
    }

    /// Builds the target list, costs, and symbol counts for shattering.
    ///
    /// Why: isolates list construction so shatter graph logic stays small.
    fn build_shattered_target_lists(
        &self,
        target_idx: usize,
        group_costs: &[Duration],
        group_sym_counts: &[usize],
        group_count: usize,
    ) -> ShatteredTargetLists {
        use indexmap::IndexSet;
        // Build the new target set: all original targets except the shattered
        // one, plus one target per horizon group.
        let mut names: IndexSet<String> = IndexSet::new();
        let mut costs: Vec<Duration> = Vec::new();
        let mut symbol_counts: Vec<usize> = Vec::new();

        // Map from old schedule index -> new index. The shattered target gets
        // no entry; its dependents are rewired to group targets instead.
        let mut old_to_new: Vec<Option<usize>> = Vec::new();

        for (i, t) in self.schedule.targets.iter().enumerate() {
            if i == target_idx {
                old_to_new.push(None);
                continue;
            }
            let new_idx = names.len();
            names.insert(t.name.clone());
            costs.push(t.cost);
            symbol_counts.push(t.symbol_count);
            old_to_new.push(Some(new_idx));
        }

        // Add group targets.
        let group_base = names.len();
        for g in 0..group_count {
            let group_name = format!("{}::group_{g}", self.target_id);
            names.insert(group_name);
            costs.push(group_costs[g]);
            symbol_counts.push(group_sym_counts[g]);
        }

        ShatteredTargetLists {
            names,
            costs,
            symbol_counts,
            old_to_new,
            group_base,
        }
    }

    /// Validates that shatter symbol counts match the original target.
    ///
    /// Why: schedule correctness depends on preserving target symbol counts.
    fn assert_shatter_symbol_counts(
        &self,
        target_idx: usize,
        group_count: usize,
        names: &indexmap::IndexSet<String>,
        group_sym_counts: &[usize],
        symbol_counts: &[usize],
    ) {
        let total = names.len();
        let expected_targets = self.schedule.targets.len() - 1 + group_count;
        assert_eq!(
            total,
            expected_targets,
            "shatter of {}: expected {expected_targets} targets \
             (original {} - 1 + {group_count} groups), got {total}",
            self.target_id,
            self.schedule.targets.len()
        );
        let original_syms: usize =
            self.schedule.targets[target_idx].symbol_count;
        let group_syms: usize = group_sym_counts.iter().sum();
        assert_eq!(
            group_syms, original_syms,
            "shatter of {}: group symbols ({group_syms}) != \
             original symbol count ({original_syms})",
            self.target_id
        );
        debug_assert_eq!(
            symbol_counts.len(),
            total,
            "symbol_counts must align with target list"
        );
    }

    /// Wires edges for the shattered target graph.
    ///
    /// Why: keeps edge reconstruction separate from target list assembly.
    fn wire_shattered_target_edges(
        &self,
        graph: &mut petgraph::graph::DiGraph<usize, ()>,
        group_base: usize,
        group_count: usize,
        old_to_new: &[Option<usize>],
        group_ext_deps: &[HashSet<String>],
        names: &indexmap::IndexSet<String>,
    ) {
        use petgraph::graph::NodeIndex;

        // Reconstruct edges between non-shattered targets. When a target
        // depended on the shattered target, wire it to the last group as a
        // conservative fallback (it must wait for the whole target to finish
        // unless symbol-level analysis later proves it only needs earlier
        // groups). Without this, dependents with no direct symbol references
        // into the shattered target would lose their constraint entirely and
        // start too early.
        let last_group_idx = group_base + group_count - 1;
        for (i, t) in self.schedule.targets.iter().enumerate() {
            let Some(new_i) = old_to_new[i] else {
                continue;
            };
            for &dep_idx in &t.deps {
                if let Some(new_dep) = old_to_new[dep_idx] {
                    graph.add_edge(
                        NodeIndex::new(new_dep),
                        NodeIndex::new(new_i),
                        (),
                    );
                } else {
                    // dep_idx is the shattered target -- wire to last group.
                    graph.add_edge(
                        NodeIndex::new(last_group_idx),
                        NodeIndex::new(new_i),
                        (),
                    );
                }
            }
        }

        // Inter-group edges from the SCC DAG.
        wire_inter_group_edges(
            self.intra,
            &self.groups.scc_to_group,
            group_base,
            graph,
        );

        // Wire group external deps.
        wire_group_external_deps(group_ext_deps, group_base, names, graph);

        // Rewire dependents to groups.
        wire_dependent_to_groups(
            self.symbol_graph,
            self.target_id.as_str(),
            &self.target_prefix,
            &self.groups.sym_to_group,
            group_base,
            names,
            graph,
        );
    }
}

/// Shatters a target by horizon, grouping SCCs by effective horizon.
///
/// Computes per-SCC effective horizons, then merges all SCCs that share
/// the same horizon into a single target. This produces one target per
/// distinct horizon value: SCCs that can all start at the same time are
/// batched together, giving realistic parallelism gains without the
/// overhead of one-target-per-SCC.
///
/// Each group target starts from raw attr cost (sum of member SCC costs).
/// If a cost model is provided, target-level event costs are distributed
/// across groups and fed into the model to estimate per-group overhead.
/// Dependencies are wired precisely:
/// - Each group inherits the union of external deps from its SCCs.
/// - Inter-group edges are derived from the intra-target SCC DAG.
/// - Dependents are rewired to the specific groups they reference.
///
/// Returns `None` if the target doesn't exist or has no symbols.
///
/// Why: a full shatter must update both the schedule and symbol graph so
/// the UI can preview the split consistently.
pub fn shatter_target(
    symbol_graph: &SymbolGraph,
    target_id: &TargetId,
    schedule: &ScheduleData,
    cost_model: Option<&CostModel>,
) -> Option<(ScheduleData, SymbolGraph)> {
    let intra = condense_target(symbol_graph, target_id)?;
    if intra.nodes.is_empty() {
        return None;
    }

    // Compute per-SCC effective horizons, then group by distinct values.
    let horizons =
        compute_effective_horizons(&intra, symbol_graph, target_id, schedule);
    let groups = build_shatter_groups(&intra, &horizons);

    let target_obj =
        &symbol_graph.packages[target_id.package()].targets[target_id.target()];
    let builder =
        ShatterBuilder::new(symbol_graph, schedule, target_id, &intra, &groups);
    let stats = compute_group_stats(
        target_obj,
        target_id,
        &builder.target_prefix,
        &intra,
        &groups,
        cost_model,
    );
    let group_sym_counts = adjust_group_symbol_counts(
        schedule,
        target_id,
        &stats.scc_symbol_counts,
    )?;

    // Build a modified SymbolGraph with group targets so that
    // /api/tree can resolve shattered names like `lib::group_0`.
    let modified_graph =
        builder.build_shattered_symbol_graph(&stats.costs, &stats.event_times);

    let tg = builder.build_shattered_target_graph(
        &stats.costs,
        &group_sym_counts,
        &stats.ext_deps,
    )?;
    Some((compute_schedule(&tg), modified_graph))
}

/// Adds inter-group edges derived from the SCC DAG.
///
/// If any SCC in group B depends on an SCC in group A (where A != B),
/// an edge is added from group A's target to group B's target. Duplicate
/// edges are avoided via a seen set.
///
/// Why: preserves dependency ordering between groups after shattering.
fn wire_inter_group_edges(
    intra: &IntraTargetGraph,
    scc_to_group: &[usize],
    group_base: usize,
    graph: &mut petgraph::graph::DiGraph<usize, ()>,
) {
    use petgraph::graph::NodeIndex;

    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for &(from, to) in &intra.edges {
        let g_from = scc_to_group[from];
        let g_to = scc_to_group[to];
        if g_from != g_to && seen.insert((g_from, g_to)) {
            graph.add_edge(
                NodeIndex::new(group_base + g_from),
                NodeIndex::new(group_base + g_to),
                (),
            );
        }
    }
}

/// Wires each group target to the external targets its SCCs reference.
///
/// Iterates over the pre-computed external dependencies for each group
/// and adds edges from those external targets to the group target.
///
/// Why: groups still depend on the same external crates as the original.
fn wire_group_external_deps(
    group_ext_deps: &[HashSet<String>],
    group_base: usize,
    names: &indexmap::IndexSet<String>,
    graph: &mut petgraph::graph::DiGraph<usize, ()>,
) {
    use petgraph::graph::NodeIndex;

    for (g, ext_targets) in group_ext_deps.iter().enumerate() {
        let group_idx = group_base + g;
        for ext_target in ext_targets {
            if let Some(dep_new_idx) = names.get_index_of(ext_target) {
                graph.add_edge(
                    NodeIndex::new(dep_new_idx),
                    NodeIndex::new(group_idx),
                    (),
                );
            }
        }
    }
}

/// Rewires dependents of the shattered target to the groups they reference.
///
/// For each target in the symbol graph that lists `target_id` as a
/// dependency, walks its symbols to find references into the shattered
/// target, maps those to groups, and adds edges.
///
/// Why: dependents must wait on the specific groups they actually use.
fn wire_dependent_to_groups(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    target_prefix: &str,
    sym_to_group: &HashMap<&str, usize>,
    group_base: usize,
    names: &indexmap::IndexSet<String>,
    graph: &mut petgraph::graph::DiGraph<usize, ()>,
) {
    use petgraph::graph::NodeIndex;

    for (pkg_name_d, pkg_d) in &symbol_graph.packages {
        for (tgt_key_d, tgt_d) in &pkg_d.targets {
            if !tgt_d.dependencies.contains(target_id) {
                continue;
            }
            let dep_id = format!("{pkg_name_d}/{tgt_key_d}");
            let Some(dep_new_idx) = names.get_index_of(&dep_id) else {
                continue;
            };
            let needed_groups = collect_referenced_groups(
                &tgt_d.root,
                target_prefix,
                sym_to_group,
            );
            for group_id in needed_groups {
                graph.add_edge(
                    NodeIndex::new(group_base + group_id),
                    NodeIndex::new(dep_new_idx),
                    (),
                );
            }
        }
    }
}

/// Finds which groups a dependent target references via symbol-level deps.
///
/// Walks the dependent's module tree, finds symbol dependencies that
/// point into the shattered target (start with `target_prefix`), and
/// maps those symbol paths to group ids via `sym_to_group`.
///
/// Why: rewiring dependents depends on which groups are actually referenced.
fn collect_referenced_groups(
    module: &Module,
    target_prefix: &str,
    sym_to_group: &HashMap<&str, usize>,
) -> Vec<usize> {
    let mut groups: Vec<usize> = Vec::new();

    for symbol in module.symbols.values() {
        for dep in &symbol.dependencies {
            if dep.starts_with(target_prefix)
                && let Some(&group_id) = sym_to_group.get(dep.as_str())
            {
                groups.push(group_id);
            }
        }
    }

    for submod in module.submodules.values() {
        groups.extend(collect_referenced_groups(
            submod,
            target_prefix,
            sym_to_group,
        ));
    }

    groups.sort_unstable();
    groups.dedup();
    groups
}

#[cfg(test)]
mod tests {
    use tarjanize_schemas::*;

    use super::*;
    use crate::data::*;
    use crate::horizon::extract_target_from_path;

    /// Builds a `SymbolGraph` with "dep-a/lib" as an external dependency
    /// and "test-pkg/lib" with the given symbols. Returns `(sg, schedule)`
    /// where dep-a finishes at `dep_finish_ms`.
    ///
    /// Why: provides a concise fixture for horizon and shatter tests.
    #[expect(
        clippy::too_many_lines,
        reason = "test fixture construction is inherently verbose"
    )]
    fn make_graph_with_external_dep(
        syms: &[(&str, f64, &[&str])],
        dep_finish_ms: f64,
    ) -> (SymbolGraph, ScheduleData) {
        let mut symbols = HashMap::new();
        for &(name, cost, deps) in syms {
            let dep_set: HashSet<String> =
                deps.iter().map(ToString::to_string).collect();
            let event_times = if cost > 0.0 {
                HashMap::from([(
                    "typeck".to_string(),
                    Duration::from_secs_f64(cost / 1000.0),
                )])
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
                        kind: "Function".to_string(),
                        visibility: Visibility::Public,
                    },
                },
            );
        }

        let root = Module {
            symbols,
            submodules: HashMap::new(),
        };
        let test_target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::from(["dep-a/lib".to_string()]),
            root,
        };
        let dep_target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root: Module {
                symbols: HashMap::new(),
                submodules: HashMap::new(),
            },
        };

        let mut packages = HashMap::new();
        packages.insert(
            "test-pkg".to_string(),
            Package {
                targets: HashMap::from([("lib".to_string(), test_target)]),
            },
        );
        packages.insert(
            "dep-a".to_string(),
            Package {
                targets: HashMap::from([("lib".to_string(), dep_target)]),
            },
        );
        let sg = SymbolGraph { packages };

        let schedule = ScheduleData {
            summary: Summary {
                critical_path: Duration::from_secs_f64(
                    (dep_finish_ms + 100.0) / 1000.0,
                ),
                total_cost: Duration::from_secs_f64(
                    (dep_finish_ms + 100.0) / 1000.0,
                ),
                parallelism_ratio: 1.0,
                target_count: 2,
                symbol_count: syms.len(),
                lane_count: 1,
            },
            targets: vec![
                TargetData {
                    name: "dep-a/lib".to_string(),
                    start: Duration::ZERO,
                    finish: Duration::from_secs_f64(dep_finish_ms / 1000.0),
                    cost: Duration::from_secs_f64(dep_finish_ms / 1000.0),
                    slack: Duration::ZERO,
                    lane: 0,
                    symbol_count: 0,
                    deps: vec![],
                    dependents: vec![1],
                    on_critical_path: true,
                    forward_pred: None,
                    backward_succ: Some(1),
                },
                TargetData {
                    name: "test-pkg/lib".to_string(),
                    start: Duration::from_secs_f64(dep_finish_ms / 1000.0),
                    finish: Duration::from_secs_f64(
                        (dep_finish_ms + 100.0) / 1000.0,
                    ),
                    cost: Duration::from_secs_f64(100.0 / 1000.0),
                    slack: Duration::ZERO,
                    lane: 0,
                    symbol_count: syms.len(),
                    deps: vec![0],
                    dependents: vec![],
                    on_critical_path: true,
                    forward_pred: Some(0),
                    backward_succ: None,
                },
            ],
            critical_path: vec![0, 1],
        };
        (sg, schedule)
    }

    /// No external deps should yield zero horizons.
    ///
    /// Why: horizons represent external gating; without it, SCCs start at zero.
    #[test]
    fn horizon_all_same_when_no_external_deps() {
        let (sg, schedule) = make_graph_with_external_dep(
            &[("a", 10.0, &[]), ("b", 20.0, &[])],
            50.0,
        );
        let intra =
            condense_target(&sg, &TargetId::new("test-pkg", "lib")).unwrap();
        let horizons = compute_effective_horizons(
            &intra,
            &sg,
            &TargetId::new("test-pkg", "lib"),
            &schedule,
        );
        assert_eq!(horizons.len(), intra.nodes.len());
        for h in &horizons {
            assert!(
                h.is_zero(),
                "no external deps means horizon = Duration::ZERO"
            );
        }
    }

    /// External deps should set the horizon to their finish time.
    ///
    /// Why: SCCs cannot start before required external targets finish.
    #[test]
    fn horizon_reflects_external_dep_finish_time() {
        let (sg, schedule) = make_graph_with_external_dep(
            &[("a", 10.0, &["[dep-a/lib]::foo"]), ("b", 20.0, &[])],
            50.0,
        );
        let intra =
            condense_target(&sg, &TargetId::new("test-pkg", "lib")).unwrap();
        let horizons = compute_effective_horizons(
            &intra,
            &sg,
            &TargetId::new("test-pkg", "lib"),
            &schedule,
        );
        let a_scc = intra
            .nodes
            .iter()
            .position(|n| n.symbols.iter().any(|s| s.ends_with("::a")))
            .unwrap();
        let b_scc = intra
            .nodes
            .iter()
            .position(|n| n.symbols.iter().any(|s| s.ends_with("::b")))
            .unwrap();
        let expected = Duration::from_secs_f64(50.0 / 1000.0);
        assert_eq!(horizons[a_scc], expected);
        assert!(horizons[b_scc].is_zero());
    }

    /// Horizons should propagate through predecessor SCCs.
    ///
    /// Why: if A is gated by external deps, B that depends on A is also gated.
    #[test]
    fn horizon_propagates_through_predecessors() {
        let prefix = "[test-pkg/lib]::";
        let (sg, schedule) = make_graph_with_external_dep(
            &[
                ("a", 10.0, &["[dep-a/lib]::foo"]),
                ("b", 20.0, &[&format!("{prefix}a")]),
            ],
            50.0,
        );
        let intra =
            condense_target(&sg, &TargetId::new("test-pkg", "lib")).unwrap();
        let horizons = compute_effective_horizons(
            &intra,
            &sg,
            &TargetId::new("test-pkg", "lib"),
            &schedule,
        );
        let a_scc = intra
            .nodes
            .iter()
            .position(|n| n.symbols.iter().any(|s| s.ends_with("::a")))
            .unwrap();
        let b_scc = intra
            .nodes
            .iter()
            .position(|n| n.symbols.iter().any(|s| s.ends_with("::b")))
            .unwrap();
        let expected = Duration::from_secs_f64(50.0 / 1000.0);
        assert_eq!(horizons[a_scc], expected, "a directly depends on dep-a");
        assert_eq!(
            horizons[b_scc], expected,
            "b inherits a's horizon through predecessor"
        );
    }

    /// Target extraction should parse bracketed paths and reject invalid input.
    ///
    /// Why: horizon computation depends on correctly identifying external targets.
    #[test]
    fn extract_target_from_path_works() {
        assert_eq!(extract_target_from_path("[pkg/lib]::foo"), Some("pkg/lib"));
        assert_eq!(
            extract_target_from_path("[a/test]::mod::bar"),
            Some("a/test")
        );
        assert_eq!(extract_target_from_path("no_brackets"), None);
    }

    /// Shatter should use the cost model when provided.
    ///
    /// Why: overhead distribution depends on model coefficients.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Test fixture setup is verbose; splitting would obscure flow."
    )]
    fn shatter_uses_cost_model() {
        // Setup: 1 target, 1 symbol, with significant metadata overhead.
        let mut symbols = HashMap::new();
        symbols.insert(
            "MySym".to_string(),
            Symbol {
                file: "lib.rs".to_string(),
                event_times_ms: HashMap::from([(
                    "typeck".to_string(),
                    Duration::from_millis(10),
                )]),
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Function".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        let target = Target {
            timings: TargetTimings {
                wall_time: Duration::from_millis(100),
                event_times_ms: HashMap::from([
                    (
                        "metadata_decode_entry".to_string(),
                        Duration::from_millis(40),
                    ),
                    (
                        "incr_comp_persist".to_string(),
                        Duration::from_millis(50),
                    ),
                ]),
            },
            dependencies: HashSet::new(),
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
        };
        let mut packages = HashMap::new();
        packages.insert(
            "my-pkg".to_string(),
            Package {
                targets: HashMap::from([("lib".to_string(), target)]),
            },
        );
        let sg = SymbolGraph { packages };

        let schedule = ScheduleData {
            summary: Summary {
                critical_path: Duration::from_millis(100),
                total_cost: Duration::from_millis(100),
                parallelism_ratio: 1.0,
                target_count: 1,
                symbol_count: 1,
                lane_count: 1,
            },
            targets: vec![TargetData {
                name: "my-pkg/lib".to_string(),
                start: Duration::ZERO,
                finish: Duration::from_millis(100),
                cost: Duration::from_millis(100),
                slack: Duration::ZERO,
                lane: 0,
                symbol_count: 1,
                deps: vec![],
                dependents: vec![],
                on_critical_path: true,
                forward_pred: None,
                backward_succ: None,
            }],
            critical_path: vec![0],
        };

        // Model: Predict = 1.0*attr + 1.0*meta + 1.0*other
        let model = CostModel {
            coeff_attr: 1.0,
            coeff_meta: 1.0,
            coeff_other: 1.0,
            r_squared: 0.9,
            inlier_threshold: 1.0,
        };

        // Test with cost model.
        let (new_schedule, _) = shatter_target(
            &sg,
            &TargetId::new("my-pkg", "lib"),
            &schedule,
            Some(&model),
        )
        .unwrap();
        let group_target = new_schedule
            .targets
            .iter()
            .find(|t| t.name.contains("::group_"))
            .unwrap();

        // attr (10) + meta (40) + other (50) = 100ms
        assert_eq!(group_target.cost, Duration::from_millis(100));

        // Test fallback (no cost model).
        let (schedule_fallback, _) = shatter_target(
            &sg,
            &TargetId::new("my-pkg", "lib"),
            &schedule,
            None,
        )
        .unwrap();
        let group_fallback = schedule_fallback
            .targets
            .iter()
            .find(|t| t.name.contains("::group_"))
            .unwrap();

        // attr (10) = 10ms
        assert_eq!(group_fallback.cost, Duration::from_millis(10));
    }
}
