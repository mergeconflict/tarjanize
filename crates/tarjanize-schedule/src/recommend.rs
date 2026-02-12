//! Horizon-based shatter helpers and cost distribution logic.
//!
//! Computes effective horizons for SCCs inside a target and provides
//! utilities to shatter a target into horizon groups while distributing
//! target-level overhead costs across the new groups.

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::time::Duration;

use tarjanize_schemas::{
    CostModel, Module, SymbolGraph, SymbolKind, Target, TargetTimings,
};

use crate::data::ScheduleData;
use crate::schedule::{TargetGraph, compute_schedule};
use crate::target_graph::{IntraTargetGraph, condense_target};

/// Computes per-SCC effective horizons for a target's intra-target DAG.
///
/// The effective horizon for an SCC is the earliest wall-clock time at
/// which it could begin compiling. This accounts for both direct external
/// dependencies (symbols referencing other targets) and transitive
/// constraints propagated through predecessor SCCs in the DAG.
///
/// Returns a `Vec<Duration>` indexed by SCC id, where each value is the
/// effective horizon. SCCs with no external dependencies and no
/// predecessors with external dependencies have horizon `Duration::ZERO`.
pub fn compute_effective_horizons(
    intra: &IntraTargetGraph,
    symbol_graph: &SymbolGraph,
    target_id: &str,
    schedule: &ScheduleData,
) -> Vec<Duration> {
    let n = intra.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Step 1: Build a target name -> finish time lookup from the schedule,
    // restricted to targets that are actual dependencies of this target.
    // Symbol-level dependencies can reference targets outside the
    // target-level dependency set (e.g. benchmark/test targets in the
    // same package, or re-exported paths from non-dependencies). Including
    // those would inflate max_horizon beyond the target's real start time,
    // making shatter groups look later than they really are.
    let target_prefix = format!("[{target_id}]::");
    let (package_name, target_key) = target_id
        .split_once('/')
        .expect("target_id must be in pkg/target format");
    let target_obj = &symbol_graph.packages[package_name].targets[target_key];
    let dep_set = &target_obj.dependencies;
    let finish_times: HashMap<String, Duration> = schedule
        .targets
        .iter()
        .filter(|t| dep_set.contains(&t.name))
        .map(|t| (t.name.clone(), t.finish))
        .collect();

    // Step 2: Walk the symbol graph's module tree to collect per-symbol
    // max external finish times. We need the original module tree because
    // `IntraTargetGraph` only stores symbol paths, not the dependency sets.
    let root = &target_obj.root;
    let symbol_finish =
        collect_external_max_finish(root, &target_prefix, "", &finish_times);

    // Step 3: For each SCC, compute the direct horizon as the max external
    // finish time across all member symbols.
    let mut direct_horizon = vec![Duration::ZERO; n];
    for node in &intra.nodes {
        for sym in &node.symbols {
            if let Some(&ft) = symbol_finish.get(sym) {
                direct_horizon[node.id] = direct_horizon[node.id].max(ft);
            }
        }
    }

    // Step 4: Build predecessor adjacency lists. In IntraTargetGraph, edges
    // go from dependency -> dependent, i.e. (from, to) means `to` depends
    // on `from`. So `from` is a predecessor of `to`.
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(from, to) in &intra.edges {
        predecessors[to].push(from);
    }

    // Step 5: Topological sort the SCC DAG. We process nodes in dependency
    // order so every predecessor's horizon is computed before its dependents.
    // Kahn's algorithm: track in-degree, start from sources (in-degree 0).
    let mut in_degree = vec![0_usize; n];
    for &(_, to) in &intra.edges {
        in_degree[to] += 1;
    }
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut topo_order: Vec<usize> = Vec::with_capacity(n);

    // Build a successors list for traversal from dependency -> dependent.
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(from, to) in &intra.edges {
        successors[from].push(to);
    }

    while let Some(node) = queue.pop() {
        topo_order.push(node);
        for &succ in &successors[node] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    // Step 6: Propagate effective horizons in topological order.
    // Each SCC's effective horizon is the max of its own direct horizon
    // and the effective horizons of all its predecessors.
    let mut effective = direct_horizon;
    for &node in &topo_order {
        for &pred in &predecessors[node] {
            effective[node] = effective[node].max(effective[pred]);
        }
    }

    effective
}

/// Extracts the target identifier from a fully qualified symbol path.
///
/// Symbol paths use the format `[pkg/target]::module::symbol`. This
/// function extracts the `pkg/target` portion from between the brackets.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(extract_target_from_path("[pkg/lib]::foo"), Some("pkg/lib"));
/// assert_eq!(extract_target_from_path("no_brackets"), None);
/// ```
pub fn extract_target_from_path(path: &str) -> Option<&str> {
    // Find the opening bracket, then find the closing bracket after it.
    let start = path.find('[')? + 1;
    let end = path[start..].find(']')? + start;
    Some(&path[start..end])
}

/// Walks a module tree collecting max external finish times per symbol.
///
/// For each symbol in the module tree, examines its dependencies. External
/// dependencies (those not starting with `target_prefix`) are resolved to
/// their target's finish time via the `finish_times` lookup. Returns a map
/// from full symbol path to the maximum finish time among that symbol's
/// external dependencies.
///
/// Symbols with no external dependencies are omitted from the result.
pub fn collect_external_max_finish<S: BuildHasher>(
    module: &Module,
    target_prefix: &str,
    module_path: &str,
    finish_times: &HashMap<String, Duration, S>,
) -> HashMap<String, Duration> {
    let mut result = HashMap::new();

    for (name, symbol) in &module.symbols {
        let full_path = if module_path.is_empty() {
            format!("{target_prefix}{name}")
        } else {
            format!("{target_prefix}{module_path}::{name}")
        };

        // Find the max finish time among this symbol's external deps.
        // A dependency is external if it does not start with target_prefix.
        let mut max_finish = Duration::ZERO;
        let mut has_external = false;

        for dep in &symbol.dependencies {
            if dep.starts_with(target_prefix) {
                continue; // Intra-target dep, skip.
            }
            // Extract the target name from the dependency path and look
            // up its finish time in the schedule.
            if let Some(dep_target) = extract_target_from_path(dep)
                && let Some(&ft) = finish_times.get(dep_target)
            {
                max_finish = max_finish.max(ft);
                has_external = true;
            }
        }

        if has_external {
            result.insert(full_path, max_finish);
        }
    }

    // Recurse into submodules.
    for (submod_name, submod) in &module.submodules {
        let child_path = if module_path.is_empty() {
            submod_name.clone()
        } else {
            format!("{module_path}::{submod_name}")
        };
        result.extend(collect_external_max_finish(
            submod,
            target_prefix,
            &child_path,
            finish_times,
        ));
    }

    result
}

/// Per-group scaling ratios for distributing target-level event costs.
///
/// These ratios let us apply different heuristics per event category while
/// keeping all distribution logic in one place.
struct GroupRatios {
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
fn normalized_ratios_from_usize(counts: &[usize]) -> Vec<f64> {
    if counts.is_empty() {
        // No groups to scale; return an empty ratio vector.
        return Vec::new();
    }
    let len_u32 = u32::try_from(counts.len()).expect("ratio length fits u32");
    let total: usize = counts.iter().sum();
    if total == 0 {
        // Uniform fallback keeps total cost conserved when no signals exist.
        return vec![1.0 / f64::from(len_u32); counts.len()];
    }
    let total_u32 = u32::try_from(total).expect("count fits u32");
    let total_f = f64::from(total_u32);
    // Normalize each group's count to a ratio so the distribution sums to 1.
    counts
        .iter()
        .map(|&count| {
            let count_u32 = u32::try_from(count).expect("count fits u32");
            f64::from(count_u32) / total_f
        })
        .collect()
}

/// Computes normalized ratios from floating-point counts.
///
/// Falls back to the provided ratio if the total is zero.
fn normalized_ratios_from_f64(counts: &[f64], fallback: &[f64]) -> Vec<f64> {
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
fn normalized_ratios_from_usize_with_fallback(
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
fn count_impls_by_group(
    module: &Module,
    target_prefix: &str,
    module_path: &str,
    sym_to_group: &HashMap<&str, usize>,
    counts: &mut [usize],
) {
    // Walk symbols in this module and increment the owning group's impl count.
    for (name, symbol) in &module.symbols {
        let full_path = if module_path.is_empty() {
            format!("{target_prefix}{name}")
        } else {
            format!("{target_prefix}{module_path}::{name}")
        };
        if matches!(symbol.kind, SymbolKind::Impl { .. })
            && let Some(&group) = sym_to_group.get(full_path.as_str())
        {
            counts[group] += 1;
        }
    }

    // Recurse into submodules to cover the full target.
    for (submod_name, submod) in &module.submodules {
        let child_path = if module_path.is_empty() {
            submod_name.clone()
        } else {
            format!("{module_path}::{submod_name}")
        };
        count_impls_by_group(
            submod,
            target_prefix,
            &child_path,
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
fn distribute_event_times(
    event_times: &HashMap<String, f64>,
    group_ratios: &GroupRatios,
) -> (Vec<HashMap<String, f64>>, Vec<f64>, Vec<f64>) {
    let group_count = group_ratios.sym.len();
    let mut per_group: Vec<HashMap<String, f64>> =
        vec![HashMap::new(); group_count];
    let mut group_meta = vec![0.0; group_count];
    let mut group_other = vec![0.0; group_count];

    // Apply per-event scaling rules to build each group's event map.
    for (label, &ms) in event_times {
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
            let allocated_ms = ms * ratios[g];
            if allocated_ms <= 0.0 {
                continue;
            }
            per_group[g].insert(label.clone(), allocated_ms);
            if is_meta {
                group_meta[g] += allocated_ms;
            } else {
                group_other[g] += allocated_ms;
            }
        }
    }

    (per_group, group_meta, group_other)
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
#[expect(clippy::too_many_lines)]
pub fn shatter_target(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    schedule: &ScheduleData,
    cost_model: Option<&CostModel>,
) -> Option<(ScheduleData, SymbolGraph)> {
    let intra = condense_target(symbol_graph, target_id).or_else(|| {
        eprintln!("shatter {target_id}: condense_target returned None");
        None
    })?;
    if intra.nodes.is_empty() {
        eprintln!("shatter {target_id}: intra.nodes is empty");
        return None;
    }
    eprintln!(
        "shatter {target_id}: {} SCC nodes, {} edges",
        intra.nodes.len(),
        intra.edges.len()
    );

    // Compute per-SCC effective horizons, then group by distinct values.
    let horizons =
        compute_effective_horizons(&intra, symbol_graph, target_id, schedule);
    let (scc_to_group, group_count) = group_sccs_by_horizon(&horizons);

    let target_prefix = format!("[{target_id}]::");

    // Build symbol_path -> group id lookup. Used to resolve which group
    // a dependent's symbol-level dep lands in.
    let mut sym_to_group: HashMap<&str, usize> = HashMap::new();
    for node in &intra.nodes {
        let group = scc_to_group[node.id];
        for sym in &node.symbols {
            sym_to_group.insert(sym.as_str(), group);
        }
    }

    // Look up target-level timing data for cost model predictions.
    // We distribute target-level event costs across groups instead of
    // duplicating everything into each group.

    // Calculate dependency scaling ratios.
    // The "Scaled Metadata Cost" hypothesis assumes metadata decode cost is
    // proportional to the number of dependencies loaded.
    let (pkg_name, tgt_key) = target_id
        .split_once('/')
        .expect("target_id checked in condense_target");
    let target_obj = &symbol_graph.packages[pkg_name].targets[tgt_key];
    let dep_set = &target_obj.dependencies;
    let total_deps = dep_set.len();

    let mut group_ratios = vec![1.0; group_count];
    let mut group_ext: Vec<std::collections::HashSet<String>> =
        vec![std::collections::HashSet::new(); group_count];

    if total_deps > 0 {
        let root = &target_obj.root;
        let all_sym_deps =
            collect_symbol_external_targets(root, &target_prefix, "");

        for node in &intra.nodes {
            let g = scc_to_group[node.id];
            for sym in &node.symbols {
                if let Some(targets) = all_sym_deps.get(sym.as_str()) {
                    for t in targets {
                        if dep_set.contains(t.as_str()) {
                            group_ext[g].insert(t.clone());
                        }
                    }
                }
            }
        }
        // Dependency counts are bounded by workspace size; use u32 to
        // avoid clippy's precision-loss lint while keeping ratios stable.
        let total_deps_u32 =
            u32::try_from(total_deps).expect("dependency count fits u32");
        let total_deps_f = f64::from(total_deps_u32);

        for (g, ext) in group_ext.iter().enumerate() {
            let ext_len_u32 =
                u32::try_from(ext.len()).expect("dependency count fits u32");
            group_ratios[g] = f64::from(ext_len_u32) / total_deps_f;
        }
    }

    // Aggregate per-group cost and symbol count from member SCCs.
    let mut group_costs = vec![Duration::ZERO; group_count];
    let mut group_attr = vec![0.0; group_count];
    let mut group_scc_sym_counts = vec![0_usize; group_count];
    for node in &intra.nodes {
        let g = scc_to_group[node.id];
        group_attr[g] += node.cost.as_secs_f64() * 1000.0;
        group_scc_sym_counts[g] += node.symbols.len();
    }

    // Track impl counts per group for trait-system event scaling.
    let mut group_impl_counts = vec![0_usize; group_count];
    count_impls_by_group(
        &target_obj.root,
        &target_prefix,
        "",
        &sym_to_group,
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
        deps: group_ratios.clone(),
    };

    // Distribute target-level event costs across groups.
    let (group_event_times, group_meta, group_other) =
        distribute_event_times(&target_obj.timings.event_times_ms, &scales);

    // Compute final group costs using the cost model if available.
    for g in 0..group_count {
        if let Some(model) = cost_model {
            // Use per-group meta/other instead of duplicating the full target.
            group_costs[g] = Duration::from_secs_f64(
                model.predict(group_attr[g], group_meta[g], group_other[g])
                    / 1000.0,
            );
        } else {
            group_costs[g] = Duration::from_secs_f64(group_attr[g] / 1000.0);
        }
    }

    // Start with SCC symbol counts; we add non-compilable extras later
    // to keep the schedule's symbol_count consistent with the original.
    let mut group_sym_counts = group_scc_sym_counts.clone();

    // The SCC DAG excludes Use/ExternCrate symbols (they're re-exports,
    // not compilable). Account for them so the shattered groups' total
    // symbol count matches the original target. Assign the extras to
    // group 0 (lowest horizon) since they have zero cost.
    let target_idx = schedule
        .targets
        .iter()
        .position(|t| t.name == target_id)
        .or_else(|| {
            eprintln!(
                "shatter {target_id}: not found in schedule ({} targets)",
                schedule.targets.len()
            );
            None
        })?;
    eprintln!(
        "shatter {target_id}: found at schedule idx {target_idx}, \
         {group_count} groups, schedule sym_count={}",
        schedule.targets[target_idx].symbol_count
    );
    let scc_syms: usize = group_sym_counts.iter().sum();
    let extra = schedule.targets[target_idx]
        .symbol_count
        .saturating_sub(scc_syms);
    group_sym_counts[0] += extra;

    // Build a modified SymbolGraph with group targets so that
    // /api/tree can resolve shattered names like `lib::group_0`.
    let modified_graph = build_shattered_symbol_graph(
        symbol_graph,
        target_id,
        &target_prefix,
        &intra,
        &scc_to_group,
        group_count,
        &group_costs,
        &group_event_times,
    );

    let tg = build_shattered_target_graph(
        symbol_graph,
        target_id,
        &target_prefix,
        schedule,
        &intra,
        &scc_to_group,
        &sym_to_group,
        group_count,
        &group_costs,
        &group_sym_counts,
        &group_ext,
    )
    .or_else(|| {
        eprintln!(
            "shatter {target_id}: build_shattered_target_graph returned None"
        );
        None
    })?;
    eprintln!(
        "shatter {target_id}: success, new schedule has {} targets",
        tg.names.len()
    );
    Some((compute_schedule(&tg), modified_graph))
}

/// Builds the `TargetGraph` for a horizon-shattered target.
///
/// Constructs the full target graph with all original targets (except
/// the shattered one) plus one target per horizon group. Wires edges
/// between surviving targets, inter-group edges from the SCC DAG,
/// external deps per group, and dependent rewiring.
#[expect(
    clippy::too_many_arguments,
    reason = "extracted helper for shatter_target; bundling into a struct would obscure the data flow"
)]
fn build_shattered_target_graph(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    target_prefix: &str,
    schedule: &ScheduleData,
    intra: &IntraTargetGraph,
    scc_to_group: &[usize],
    sym_to_group: &HashMap<&str, usize>,
    group_count: usize,
    group_costs: &[Duration],
    group_sym_counts: &[usize],
    group_ext_deps: &[std::collections::HashSet<String>],
) -> Option<TargetGraph> {
    use indexmap::IndexSet;
    use petgraph::graph::{DiGraph, NodeIndex};

    // Find the shattered target's index in the schedule.
    let target_idx =
        schedule.targets.iter().position(|t| t.name == target_id)?;

    // Build the new target set: all original targets except the shattered
    // one, plus one target per horizon group.
    let mut names: IndexSet<String> = IndexSet::new();
    let mut costs: Vec<Duration> = Vec::new();
    let mut symbol_counts: Vec<usize> = Vec::new();

    // Map from old schedule index -> new index. The shattered target gets
    // no entry; its dependents are rewired to group targets instead.
    let mut old_to_new: Vec<Option<usize>> = Vec::new();

    for (i, t) in schedule.targets.iter().enumerate() {
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
        let group_name = format!("{target_id}::group_{g}");
        names.insert(group_name);
        costs.push(group_costs[g]);
        symbol_counts.push(group_sym_counts[g]);
    }

    let total = names.len();
    let expected_targets = schedule.targets.len() - 1 + group_count;
    assert_eq!(
        total,
        expected_targets,
        "shatter of {target_id}: expected {expected_targets} targets \
         (original {} - 1 + {group_count} groups), got {total}",
        schedule.targets.len()
    );
    let original_syms: usize = schedule.targets[target_idx].symbol_count;
    let group_syms: usize = group_sym_counts.iter().sum();
    assert_eq!(
        group_syms, original_syms,
        "shatter of {target_id}: group symbols ({group_syms}) != \
         original symbol count ({original_syms})"
    );
    let mut graph = DiGraph::<usize, ()>::with_capacity(total, 0);
    for i in 0..total {
        graph.add_node(i);
    }

    // Reconstruct edges between non-shattered targets. When a target
    // depended on the shattered target, wire it to the last group as a
    // conservative fallback (it must wait for the whole target to finish
    // unless symbol-level analysis later proves it only needs earlier
    // groups). Without this, dependents with no direct symbol references
    // into the shattered target would lose their constraint entirely and
    // start too early.
    let last_group_idx = group_base + group_count - 1;
    for (i, t) in schedule.targets.iter().enumerate() {
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
                // dep_idx is the shattered target — wire to last group.
                graph.add_edge(
                    NodeIndex::new(last_group_idx),
                    NodeIndex::new(new_i),
                    (),
                );
            }
        }
    }

    // Inter-group edges from the SCC DAG.
    wire_inter_group_edges(intra, scc_to_group, group_base, &mut graph);

    // Wire group external deps.
    wire_group_external_deps(group_ext_deps, group_base, &names, &mut graph);

    // Rewire dependents to groups.
    wire_dependent_to_groups(
        symbol_graph,
        target_id,
        target_prefix,
        sym_to_group,
        group_base,
        &names,
        &mut graph,
    );

    Some(TargetGraph {
        names,
        costs,
        symbol_counts,
        graph,
    })
}

/// Groups SCCs by their effective horizon value.
///
/// SCCs with the same horizon are placed in the same group. Duration
/// implements `Eq`, so exact equality is used (no epsilon needed).
/// Returns a mapping from SCC id to group id, plus the total number
/// of groups. Groups are numbered in ascending horizon order (group 0
/// has the lowest horizon).
fn group_sccs_by_horizon(horizons: &[Duration]) -> (Vec<usize>, usize) {
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
            distinct
                .iter()
                .position(|&d| d == h)
                .expect("horizon must match a distinct value")
        })
        .collect();

    let group_count = distinct.len();
    (scc_to_group, group_count)
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
#[expect(
    clippy::too_many_arguments,
    reason = "helper mirrors shatter inputs; bundling into a struct would obscure call sites"
)]
fn build_shattered_symbol_graph(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    target_prefix: &str,
    intra: &IntraTargetGraph,
    scc_to_group: &[usize],
    group_count: usize,
    group_costs: &[Duration],
    group_event_times: &[HashMap<String, f64>],
) -> SymbolGraph {
    use crate::export::{
        insert_symbol_into_module, parse_symbol_path, remove_symbol_from_module,
    };

    let mut graph = symbol_graph.clone();

    let Some((pkg_name, target_key)) = target_id.split_once('/') else {
        return graph;
    };

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

    for node in &intra.nodes {
        let group = scc_to_group[node.id];
        for sym_path in &node.symbols {
            let Some((mod_segments, sym_name)) =
                parse_symbol_path(sym_path, target_prefix)
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

        let group_target = Target {
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

/// Adds inter-group edges derived from the SCC DAG.
///
/// If any SCC in group B depends on an SCC in group A (where A != B),
/// an edge is added from group A's target to group B's target. Duplicate
/// edges are avoided via a seen set.
fn wire_inter_group_edges(
    intra: &IntraTargetGraph,
    scc_to_group: &[usize],
    group_base: usize,
    graph: &mut petgraph::graph::DiGraph<usize, ()>,
) {
    use std::collections::HashSet;

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
fn wire_group_external_deps(
    group_ext_deps: &[std::collections::HashSet<String>],
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

/// Collects per-symbol external target dependencies from a module tree.
///
/// Returns a map from full symbol path to the set of external target
/// names that symbol depends on. Only dependencies NOT starting with
/// `target_prefix` are included (cross-target deps).
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

/// Finds which groups a dependent target references via symbol-level deps.
///
/// Walks the dependent's module tree, finds symbol dependencies that
/// point into the shattered target (start with `target_prefix`), and
/// maps those symbol paths to group ids via `sym_to_group`.
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
    use std::collections::HashSet;

    use tarjanize_schemas::*;

    use super::*;
    use crate::data::*;

    /// Builds a `SymbolGraph` with "dep-a/lib" as an external dependency
    /// and "test-pkg/lib" with the given symbols. Returns `(sg, schedule)`
    /// where dep-a finishes at `dep_finish_ms`.
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

    #[test]
    fn horizon_all_same_when_no_external_deps() {
        let (sg, schedule) = make_graph_with_external_dep(
            &[("a", 10.0, &[]), ("b", 20.0, &[])],
            50.0,
        );
        let intra = condense_target(&sg, "test-pkg/lib").unwrap();
        let horizons =
            compute_effective_horizons(&intra, &sg, "test-pkg/lib", &schedule);
        assert_eq!(horizons.len(), intra.nodes.len());
        for h in &horizons {
            assert!(
                h.is_zero(),
                "no external deps means horizon = Duration::ZERO"
            );
        }
    }

    #[test]
    fn horizon_reflects_external_dep_finish_time() {
        let (sg, schedule) = make_graph_with_external_dep(
            &[("a", 10.0, &["[dep-a/lib]::foo"]), ("b", 20.0, &[])],
            50.0,
        );
        let intra = condense_target(&sg, "test-pkg/lib").unwrap();
        let horizons =
            compute_effective_horizons(&intra, &sg, "test-pkg/lib", &schedule);
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
        let intra = condense_target(&sg, "test-pkg/lib").unwrap();
        let horizons =
            compute_effective_horizons(&intra, &sg, "test-pkg/lib", &schedule);
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

    #[test]
    fn extract_target_from_path_works() {
        assert_eq!(extract_target_from_path("[pkg/lib]::foo"), Some("pkg/lib"));
        assert_eq!(
            extract_target_from_path("[a/test]::mod::bar"),
            Some("a/test")
        );
        assert_eq!(extract_target_from_path("no_brackets"), None);
    }

    #[test]
    fn shatter_uses_cost_model() {
        // Setup: 1 target, 1 symbol, with significant metadata overhead.
        let mut symbols = HashMap::new();
        symbols.insert(
            "MySym".to_string(),
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
        let target = Target {
            timings: TargetTimings {
                wall_time: Duration::from_millis(100),
                event_times_ms: HashMap::from([
                    ("metadata_decode_entry".to_string(), 40.0),
                    ("incr_comp_persist".to_string(), 50.0),
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
        let (new_schedule, _) =
            shatter_target(&sg, "my-pkg/lib", &schedule, Some(&model)).unwrap();
        let group_target = new_schedule
            .targets
            .iter()
            .find(|t| t.name.contains("::group_"))
            .unwrap();

        // attr (10) + meta (40) + other (50) = 100ms
        assert_eq!(group_target.cost, Duration::from_millis(100));

        // Test fallback (no cost model).
        let (schedule_fallback, _) =
            shatter_target(&sg, "my-pkg/lib", &schedule, None).unwrap();
        let group_fallback = schedule_fallback
            .targets
            .iter()
            .find(|t| t.name.contains("::group_"))
            .unwrap();

        // attr (10) = 10ms
        assert_eq!(group_fallback.cost, Duration::from_millis(10));
    }
}
