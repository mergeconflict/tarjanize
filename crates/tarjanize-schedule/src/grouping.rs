//! Horizon-based grouping and cost distribution for shatter analysis.
//!
//! Groups SCCs by effective horizon value, distributes target-level
//! overhead costs across the resulting groups, and computes per-group
//! statistics needed to build shattered target graphs.
//!
//! Why: shatter analysis must partition SCCs into horizon groups and
//! reallocate costs so that schedules remain meaningful after synthetic
//! splits.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use tarjanize_schemas::{
    CostModel, Module, ModulePath, QualifiedSymbolPath, SymbolKind, Target,
    TargetId, duration_to_ms_f64,
};

use crate::data::ScheduleData;
use crate::horizon::extract_target_from_path;
use crate::target_graph::IntraTargetGraph;

/// Per-group scaling ratios for distributing target-level event costs.
///
/// These ratios let us apply different heuristics per event category while
/// keeping all distribution logic in one place.
///
/// Why: different cost components correlate with different structural signals.
pub(crate) struct GroupRatios {
    /// Ratio based on attributed frontend cost (proxy for body complexity).
    attr: Vec<f64>,
    /// Ratio based on symbol count (proxy for crate-level passes).
    sym: Vec<f64>,
    /// Ratio based on impl count (proxy for trait-system workload).
    impls: Vec<f64>,
    /// Ratio based on dependency usage (proxy for metadata load).
    deps: Vec<f64>,
}

/// Computes normalized ratios from per-group counts.
///
/// Returns a uniform ratio if all counts are zero so we avoid division by
/// zero and still distribute costs predictably.
///
/// Why: we must preserve total cost even when a signal is missing.
pub(crate) fn normalized_ratios_from_usize(counts: &[usize]) -> Vec<f64> {
    if counts.is_empty() {
        // No groups to scale; return an empty ratio vector.
        return Vec::new();
    }
    // Counts are bounded by in-memory symbol counts; use u32 to avoid
    // clippy's precision-loss lint and fail fast if the input is corrupted.
    let len_u32 =
        u32::try_from(counts.len()).expect("symbol group count must fit u32");
    let total: usize = counts.iter().sum();
    if total == 0 {
        // Uniform fallback keeps total cost conserved when no signals exist.
        return vec![1.0 / f64::from(len_u32); counts.len()];
    }
    let total_u32 =
        u32::try_from(total).expect("total symbol count must fit u32");
    let total_f = f64::from(total_u32);
    // Normalize each group's count to a ratio so the distribution sums to 1.
    counts
        .iter()
        .map(|&count| {
            let count_u32 = u32::try_from(count)
                .expect("per-group symbol count must fit u32");
            f64::from(count_u32) / total_f
        })
        .collect()
}

/// Computes normalized ratios from floating-point counts.
///
/// Falls back to the provided ratio if the total is zero.
///
/// Why: floating-point signals can be all-zero; fallback avoids NaNs.
pub(crate) fn normalized_ratios_from_f64(
    counts: &[f64],
    fallback: &[f64],
) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        // No signal available; fall back to the provided ratio.
        return fallback.to_vec();
    }
    // Normalize by total so ratios sum to 1.
    counts.iter().map(|&count| count / total).collect()
}

/// Computes normalized ratios from per-group counts with a fallback.
///
/// Used for impl counts: if there are no impls at all, we fall back to
/// symbol-count scaling to avoid a zeroed distribution.
///
/// Why: impl density is useful, but a zero count should not erase costs.
pub(crate) fn normalized_ratios_from_usize_with_fallback(
    counts: &[usize],
    fallback: &[f64],
) -> Vec<f64> {
    let total: usize = counts.iter().sum();
    if total == 0 {
        // No impls at all; reuse the fallback distribution.
        return fallback.to_vec();
    }
    normalized_ratios_from_usize(counts)
}

/// Counts impl symbols per group using the `sym_to_group` lookup.
///
/// This lets trait-system costs scale with impl density instead of generic
/// symbol counts.
///
/// Why: impl-heavy groups tend to incur extra trait-system work.
pub(crate) fn count_impls_by_group(
    module: &Module,
    target_id: &TargetId,
    module_path: &ModulePath,
    sym_to_group: &HashMap<&str, usize>,
    counts: &mut [usize],
) {
    // Walk symbols in this module and increment the owning group's impl count.
    for (name, symbol) in &module.symbols {
        let full_path = QualifiedSymbolPath::new(target_id, module_path, name);
        if matches!(symbol.kind, SymbolKind::Impl { .. })
            && let Some(&group) = sym_to_group.get(full_path.as_str())
        {
            counts[group] += 1;
        }
    }

    // Recurse into submodules to cover the full target.
    for (submod_name, submod) in &module.submodules {
        count_impls_by_group(
            submod,
            target_id,
            &module_path.child(submod_name),
            sym_to_group,
            counts,
        );
    }
}

/// Distributes target-level event timings across shatter groups.
///
/// The event list is biased toward omicron's top-15 "other" events. Anything
/// not explicitly categorized falls back to symbol-count scaling.
///
/// TODO: Re-evaluate the top events across more workspaces and expand
/// the explicit list if needed.
///
/// Why: we must preserve total target overhead while reallocating costs
/// to the new groups.
pub(crate) fn distribute_event_times(
    event_times: &HashMap<String, Duration>,
    group_ratios: &GroupRatios,
) -> (Vec<HashMap<String, Duration>>, Vec<Duration>, Vec<Duration>) {
    let group_count = group_ratios.sym.len();
    let mut per_group: Vec<HashMap<String, Duration>> =
        vec![HashMap::new(); group_count];
    let mut group_meta = vec![Duration::ZERO; group_count];
    let mut group_other = vec![Duration::ZERO; group_count];

    // Apply per-event scaling rules to build each group's event map.
    for (label, &duration) in event_times {
        let is_meta = label.starts_with("metadata_");
        let ratios = if is_meta {
            // Metadata decode and registration scales with dependency usage
            // (per group deps).
            &group_ratios.deps
        } else {
            match label.as_str() {
                "compare_impl_item" | "specialization_graph_of" => {
                    // Trait-system costs scale with impl density.
                    &group_ratios.impls
                }
                "typeck"
                | "mir_borrowck"
                | "evaluate_obligation"
                | "type_op_prove_predicate"
                | "mir_built"
                | "check_well_formed" => {
                    // Function-body analysis scales with attributed frontend cost.
                    &group_ratios.attr
                }
                "expand_proc_macro" => {
                    // We don't know which symbols were generated by proc macros,
                    // so approximate by symbol count.
                    &group_ratios.sym
                }
                "expand_crate"
                | "hir_crate"
                | "late_resolve_crate"
                | "check_mod_privacy"
                | "incr_comp_encode_dep_graph" => {
                    // Crate-level passes scale with overall item volume.
                    &group_ratios.sym
                }
                _ => {
                    // Fallback: distribute by symbol count to avoid duplication.
                    &group_ratios.sym
                }
            }
        };

        // Distribute this event's cost across groups using the chosen ratio.
        for g in 0..group_count {
            let allocated = duration.mul_f64(ratios[g]);
            if allocated.is_zero() {
                continue;
            }
            per_group[g].insert(label.clone(), allocated);
            if is_meta {
                group_meta[g] += allocated;
            } else {
                group_other[g] += allocated;
            }
        }
    }

    (per_group, group_meta, group_other)
}

/// Defines SCC-to-group mappings and symbol group lookups for shattering.
///
/// Why: grouping and rewiring logic needs both the horizon groups and a fast
/// symbol->group mapping derived from the same SCC partition.
pub(crate) struct ShatterGroups<'a> {
    /// Mapping from SCC id to its horizon group.
    pub(crate) scc_to_group: Vec<usize>,
    /// Total number of horizon groups.
    pub(crate) group_count: usize,
    /// Lookup from symbol path to its horizon group.
    pub(crate) sym_to_group: HashMap<&'a str, usize>,
}

/// Builds the horizon-based grouping metadata for a target's SCCs.
///
/// Why: shatter logic depends on consistent SCC->group and symbol->group maps.
pub(crate) fn build_shatter_groups<'a>(
    intra: &'a IntraTargetGraph,
    horizons: &[Duration],
) -> ShatterGroups<'a> {
    let (scc_to_group, group_count) = group_sccs_by_horizon(horizons);

    let mut sym_to_group: HashMap<&'a str, usize> = HashMap::new();
    for node in &intra.nodes {
        let group = scc_to_group[node.id];
        for sym in &node.symbols {
            sym_to_group.insert(sym.as_str(), group);
        }
    }

    ShatterGroups {
        scc_to_group,
        group_count,
        sym_to_group,
    }
}

/// Bundles per-group costs, counts, and dependencies computed during shatter.
///
/// Why: shatter needs these aligned vectors for both schedule and graph updates.
pub(crate) struct GroupStats {
    /// Predicted per-group wall-time costs.
    pub(crate) costs: Vec<Duration>,
    /// Symbol counts from SCC membership (before extra re-exports).
    pub(crate) scc_symbol_counts: Vec<usize>,
    /// Per-group event timings for cost model or export.
    pub(crate) event_times: Vec<HashMap<String, Duration>>,
    /// External dependency sets per group, for target graph wiring.
    pub(crate) ext_deps: Vec<HashSet<String>>,
}

/// Holds the assembled target lists for a shatter operation.
///
/// Why: avoids complex tuple returns while keeping list construction cohesive.
pub(crate) struct ShatteredTargetLists {
    /// Names for all targets in the shattered graph.
    pub(crate) names: indexmap::IndexSet<String>,
    /// Costs aligned with the `names` list.
    pub(crate) costs: Vec<Duration>,
    /// Symbol counts aligned with the `names` list.
    pub(crate) symbol_counts: Vec<usize>,
    /// Mapping from old schedule indices to new target indices.
    pub(crate) old_to_new: Vec<Option<usize>>,
    /// Starting index for the newly added group targets.
    pub(crate) group_base: usize,
}

/// Computes dependency ratios from per-group external dependency sets.
///
/// Why: metadata scaling uses dependency counts while preserving total cost.
pub(crate) fn compute_dependency_ratios(
    group_ext_deps: &[HashSet<String>],
    total_deps: usize,
) -> Vec<f64> {
    let mut ratios = vec![1.0; group_ext_deps.len()];
    if total_deps == 0 {
        return ratios;
    }

    // Dependency counts are bounded by workspace size; use u32 to
    // avoid clippy's precision-loss lint while keeping ratios stable.
    let total_deps_u32 =
        u32::try_from(total_deps).expect("dependency count fits u32");
    let total_deps_f = f64::from(total_deps_u32);

    for (g, ext) in group_ext_deps.iter().enumerate() {
        let ext_len_u32 =
            u32::try_from(ext.len()).expect("dependency count fits u32");
        ratios[g] = f64::from(ext_len_u32) / total_deps_f;
    }

    ratios
}

/// Computes per-group costs, symbol counts, and external deps for a shatter.
///
/// Why: keeps the shatter pipeline small while centralizing cost logic.
pub(crate) fn compute_group_stats(
    target_obj: &Target,
    target_id: &TargetId,
    target_prefix: &str,
    intra: &IntraTargetGraph,
    groups: &ShatterGroups<'_>,
    cost_model: Option<&CostModel>,
) -> GroupStats {
    let group_count = groups.group_count;
    let dep_set = &target_obj.dependencies;

    let mut group_ext_deps: Vec<HashSet<String>> =
        vec![HashSet::new(); group_count];
    if !dep_set.is_empty() {
        let all_sym_deps = collect_symbol_external_targets(
            &target_obj.root,
            target_prefix,
            "",
        );
        for node in &intra.nodes {
            let group = groups.scc_to_group[node.id];
            for sym in &node.symbols {
                if let Some(targets) = all_sym_deps.get(sym.as_str()) {
                    for target in targets {
                        if dep_set.contains(target) {
                            group_ext_deps[group].insert(target.clone());
                        }
                    }
                }
            }
        }
    }

    // The "Scaled Metadata Cost" hypothesis assumes metadata decode cost is
    // proportional to the number of dependencies loaded.
    let dep_ratios = compute_dependency_ratios(&group_ext_deps, dep_set.len());

    // Aggregate per-group cost and symbol count from member SCCs.
    let mut group_costs = vec![Duration::ZERO; group_count];
    let mut group_attr = vec![0.0; group_count];
    let mut group_scc_sym_counts = vec![0_usize; group_count];
    for node in &intra.nodes {
        let group = groups.scc_to_group[node.id];
        group_attr[group] += node.cost.as_secs_f64() * 1000.0;
        group_scc_sym_counts[group] += node.symbols.len();
    }

    // Track impl counts per group for trait-system event scaling.
    let mut group_impl_counts = vec![0_usize; group_count];
    count_impls_by_group(
        &target_obj.root,
        target_id,
        &ModulePath::root(),
        &groups.sym_to_group,
        &mut group_impl_counts,
    );

    // Use SCC symbol counts for scaling (excludes non-compilable re-exports).
    let sym_ratio = normalized_ratios_from_usize(&group_scc_sym_counts);
    let attr_ratio = normalized_ratios_from_f64(&group_attr, &sym_ratio);
    let impl_ratio = normalized_ratios_from_usize_with_fallback(
        &group_impl_counts,
        &sym_ratio,
    );
    let scales = GroupRatios {
        attr: attr_ratio,
        sym: sym_ratio,
        impls: impl_ratio,
        deps: dep_ratios,
    };

    // Distribute target-level event costs across groups.
    let (group_event_times, group_meta, group_other) =
        distribute_event_times(&target_obj.timings.event_times_ms, &scales);

    // Compute final group costs using the cost model if available.
    for g in 0..group_count {
        if let Some(model) = cost_model {
            // Use per-group meta/other instead of duplicating the full target.
            group_costs[g] = Duration::from_secs_f64(
                model.predict(
                    group_attr[g],
                    duration_to_ms_f64(group_meta[g]),
                    duration_to_ms_f64(group_other[g]),
                ) / 1000.0,
            );
        } else {
            group_costs[g] = Duration::from_secs_f64(group_attr[g] / 1000.0);
        }
    }

    GroupStats {
        costs: group_costs,
        scc_symbol_counts: group_scc_sym_counts,
        event_times: group_event_times,
        ext_deps: group_ext_deps,
    }
}

/// Adjusts per-group symbol counts to include non-compilable extras.
///
/// Why: schedule symbol counts must still match the original target totals.
pub(crate) fn adjust_group_symbol_counts(
    schedule: &ScheduleData,
    target_id: &TargetId,
    scc_symbol_counts: &[usize],
) -> Option<Vec<usize>> {
    let target_idx = schedule
        .targets
        .iter()
        .position(|t| t.name == target_id.as_str())?;
    let mut group_sym_counts = scc_symbol_counts.to_vec();
    let scc_syms: usize = group_sym_counts.iter().sum();
    let extra = schedule.targets[target_idx]
        .symbol_count
        .saturating_sub(scc_syms);
    // Assign extras to the lowest-horizon group to keep group totals stable.
    group_sym_counts[0] += extra;
    Some(group_sym_counts)
}

/// Groups SCCs by their effective horizon value.
///
/// SCCs with the same horizon are placed in the same group. Duration
/// implements `Eq`, so exact equality is used (no epsilon needed).
/// Returns a mapping from SCC id to group id, plus the total number
/// of groups. Groups are numbered in ascending horizon order (group 0
/// has the lowest horizon).
///
/// Why: grouping by horizon models which SCCs can start together without
/// over-fragmenting the target.
pub(crate) fn group_sccs_by_horizon(
    horizons: &[Duration],
) -> (Vec<usize>, usize) {
    // Collect distinct horizon values, sorted ascending.
    let mut distinct: Vec<Duration> = Vec::new();
    let mut sorted_horizons: Vec<Duration> = horizons.to_vec();
    sorted_horizons.sort_unstable();

    for &h in &sorted_horizons {
        if distinct.last() != Some(&h) {
            distinct.push(h);
        }
    }

    // Map each SCC to its group (index into `distinct`).
    let scc_to_group: Vec<usize> = horizons
        .iter()
        .map(|&h| {
            // `distinct` is built from `horizons`, so this lookup must succeed.
            distinct
                .iter()
                .position(|&d| d == h)
                .expect("horizon must match a distinct value")
        })
        .collect();

    let group_count = distinct.len();
    (scc_to_group, group_count)
}

/// Collects per-symbol external target dependencies from a module tree.
///
/// Returns a map from full symbol path to the set of external target
/// names that symbol depends on. Only dependencies NOT starting with
/// `target_prefix` are included (cross-target deps).
///
/// Why: group wiring and horizon computations need target-level external
/// references derived from symbol-level deps.
pub fn collect_symbol_external_targets(
    module: &Module,
    target_prefix: &str,
    module_path: &str,
) -> HashMap<String, Vec<String>> {
    let mut result = HashMap::new();

    for (name, symbol) in &module.symbols {
        let full_path = if module_path.is_empty() {
            format!("{target_prefix}{name}")
        } else {
            format!("{target_prefix}{module_path}::{name}")
        };

        let ext_targets: Vec<String> = symbol
            .dependencies
            .iter()
            .filter(|dep| !dep.starts_with(target_prefix))
            .filter_map(|dep| extract_target_from_path(dep))
            .map(str::to_string)
            .collect();

        if !ext_targets.is_empty() {
            result.insert(full_path, ext_targets);
        }
    }

    for (submod_name, submod) in &module.submodules {
        let child_path = if module_path.is_empty() {
            submod_name.clone()
        } else {
            format!("{module_path}::{submod_name}")
        };
        result.extend(collect_symbol_external_targets(
            submod,
            target_prefix,
            &child_path,
        ));
    }

    result
}
