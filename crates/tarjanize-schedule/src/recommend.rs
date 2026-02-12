//! Split recommendation engine: horizon analysis and candidate evaluation.
//!
//! Combines two phases:
//!
//! 1. **Effective horizon computation** — each SCC in a target's intra-target
//!    DAG gets an "effective horizon": the earliest wall-clock time at which
//!    it could begin compiling, given its external dependencies and the
//!    horizons of its predecessors.
//!
//! 2. **Split candidate generation** — finds threshold cuts where a
//!    significant jump in effective horizon indicates a good split boundary.
//!    For each candidate, computes both the local improvement (within the
//!    target) and the global improvement (full schedule recomputation).

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tarjanize_schemas::{
    CostModel, Module, SymbolGraph, Target, TargetTimings, serde_duration,
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
    // making all splits appear detrimental.
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

/// Ranked split candidates for a single target.
///
/// Contains the target's current cost and a list of candidate threshold
/// cuts, sorted by global improvement (most beneficial first). An empty
/// `candidates` list means the target has no beneficial split points.
#[derive(Debug, Deserialize, Serialize)]
pub struct SplitRecommendation {
    /// Target identifier in `{package}/{target}` format.
    pub target: String,
    /// Current predicted compilation cost.
    #[serde(rename = "current_cost_ms", with = "serde_duration")]
    pub current_cost: Duration,
    /// Candidate splits, sorted by `global_improvement` descending.
    pub candidates: Vec<SplitCandidate>,
}

/// A single candidate split at a specific horizon threshold.
///
/// Represents cutting the SCC DAG into a "downset" (SCCs with effective
/// horizon at or below the threshold) and an "upset" (remaining SCCs).
/// The downset becomes a new dependency crate that can begin compiling
/// earlier, while the upset keeps the original target name.
#[derive(Debug, Deserialize, Serialize)]
pub struct SplitCandidate {
    /// The effective horizon threshold for this cut.
    #[serde(rename = "threshold_ms", with = "serde_duration")]
    pub threshold: Duration,
    /// Reduction in the target's own compilation time from parallelism.
    #[serde(rename = "local_improvement_ms", with = "serde_duration")]
    pub local_improvement: Duration,
    /// Reduction in the full build's critical path length.
    #[serde(rename = "global_improvement_ms", with = "serde_duration")]
    pub global_improvement: Duration,
    /// Number of SCCs in the downset (new dependency crate).
    pub downset_scc_count: usize,
    /// Predicted cost of the downset crate.
    #[serde(rename = "downset_cost_ms", with = "serde_duration")]
    pub downset_cost: Duration,
    /// Number of SCCs in the upset (residual crate).
    pub upset_scc_count: usize,
    /// Predicted cost of the upset crate.
    #[serde(rename = "upset_cost_ms", with = "serde_duration")]
    pub upset_cost: Duration,
    /// Module paths in the downset.
    pub downset_modules: Vec<String>,
    /// Module paths in the upset.
    pub upset_modules: Vec<String>,
    /// All distinct module paths touched by this split.
    pub split_modules: Vec<String>,
}

/// Generates ranked split candidates for a target.
///
/// Analyzes the target's intra-target SCC DAG to find threshold cuts
/// where differing effective horizons indicate a beneficial split. For
/// each candidate, computes both local improvement (within-target
/// parallelism gain) and global improvement (full schedule critical
/// path reduction).
///
/// Returns a `SplitRecommendation` with candidates sorted by
/// `global_improvement` descending. Candidates with negative local
/// improvement are filtered out (they would make things worse).
pub fn compute_split_recommendations(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    schedule: &ScheduleData,
    cost_model: Option<&CostModel>,
) -> SplitRecommendation {
    let empty = SplitRecommendation {
        target: target_id.to_string(),
        current_cost: Duration::ZERO,
        candidates: Vec::new(),
    };

    // Look up the target's current cost from the schedule.
    let Some(target_data) =
        schedule.targets.iter().find(|t| t.name == target_id)
    else {
        return empty;
    };
    let current_cost = target_data.cost;

    // Condense the target into its SCC DAG.
    let Some(intra) = condense_target(symbol_graph, target_id) else {
        return SplitRecommendation {
            target: target_id.to_string(),
            current_cost,
            candidates: Vec::new(),
        };
    };

    // Compute effective horizons for each SCC.
    let horizons =
        compute_effective_horizons(&intra, symbol_graph, target_id, schedule);

    if horizons.is_empty() {
        return SplitRecommendation {
            target: target_id.to_string(),
            current_cost,
            candidates: Vec::new(),
        };
    }

    // Find the maximum horizon — thresholds must be strictly less than
    // this to produce a non-trivial split.
    let max_horizon = horizons.iter().copied().max().unwrap_or(Duration::ZERO);

    // Collect distinct threshold values strictly less than max_horizon.
    // Each unique horizon value is a potential cut point.
    let mut thresholds: Vec<Duration> = horizons
        .iter()
        .copied()
        .filter(|&h| h < max_horizon)
        .collect();
    thresholds.sort_unstable();
    thresholds.dedup();

    // Look up target-level timing data for cost model predictions.
    // Each half inherits the full target's meta/other overhead.
    let (meta, other) = extract_meta_other(symbol_graph, target_id);

    let ctx = ThresholdContext {
        intra: &intra,
        horizons: &horizons,
        max_horizon,
        target_data,
        current_cost,
        meta,
        other,
        cost_model,
        schedule,
        target_id,
    };

    let candidates: Vec<SplitCandidate> = thresholds
        .iter()
        .filter_map(|&threshold| evaluate_threshold(&ctx, threshold))
        .collect();

    let mut sorted = candidates;
    sorted.sort_by_key(|c| std::cmp::Reverse(c.global_improvement));

    SplitRecommendation {
        target: target_id.to_string(),
        current_cost,
        candidates: sorted,
    }
}

/// Context for evaluating threshold candidates within a single target.
///
/// Groups the per-target state needed by `evaluate_threshold` to keep
/// the parameter count within clippy's limit.
struct ThresholdContext<'a> {
    intra: &'a IntraTargetGraph,
    horizons: &'a [Duration],
    max_horizon: Duration,
    target_data: &'a crate::data::TargetData,
    current_cost: Duration,
    meta: f64,
    other: f64,
    cost_model: Option<&'a CostModel>,
    schedule: &'a ScheduleData,
    target_id: &'a str,
}

/// Evaluates a single threshold cut, returning a `SplitCandidate` if
/// the split produces a positive local improvement, or `None` otherwise.
///
/// This is extracted from `compute_split_recommendations` to keep the
/// main function under the line-count lint threshold.
fn evaluate_threshold(
    ctx: &ThresholdContext<'_>,
    threshold: Duration,
) -> Option<SplitCandidate> {
    // Partition SCCs into downset (horizon <= threshold) and upset.
    let downset_indices: Vec<usize> = (0..ctx.intra.nodes.len())
        .filter(|&i| ctx.horizons[i] <= threshold)
        .collect();
    let upset_indices: Vec<usize> = (0..ctx.intra.nodes.len())
        .filter(|&i| ctx.horizons[i] > threshold)
        .collect();

    if downset_indices.is_empty() || upset_indices.is_empty() {
        return None;
    }

    // Compute raw attr costs (sum of SCC cost values) as f64 ms for the
    // cost model, which operates in f64 millisecond space.
    let downset_attr: f64 = downset_indices
        .iter()
        .map(|&i| ctx.intra.nodes[i].cost.as_secs_f64() * 1000.0)
        .sum();
    let upset_attr: f64 = upset_indices
        .iter()
        .map(|&i| ctx.intra.nodes[i].cost.as_secs_f64() * 1000.0)
        .sum();

    // Predict costs for each half. With a cost model, use the predict
    // function (each half inherits the full target's meta/other).
    // Without a cost model, use raw attr sums. The cost model operates
    // in f64 milliseconds, so we convert to Duration at the boundary.
    let (downset_cost, upset_cost) = if let Some(model) = ctx.cost_model {
        (
            Duration::from_secs_f64(
                model.predict(downset_attr, ctx.meta, ctx.other) / 1000.0,
            ),
            Duration::from_secs_f64(
                model.predict(upset_attr, ctx.meta, ctx.other) / 1000.0,
            ),
        )
    } else {
        (
            Duration::from_secs_f64(downset_attr / 1000.0),
            Duration::from_secs_f64(upset_attr / 1000.0),
        )
    };

    // Compute local improvement: how much the target's finish time
    // improves from splitting. The downset starts at `threshold`,
    // the upset starts at `max(threshold + downset_cost, max_horizon)`.
    let downset_finish = threshold + downset_cost;
    let upset_start = (threshold + downset_cost).max(ctx.max_horizon);
    let upset_finish = upset_start + upset_cost;
    let split_finish = downset_finish.max(upset_finish);
    let original_finish = ctx.target_data.start + ctx.current_cost;
    let local_improvement = original_finish.saturating_sub(split_finish);

    // Filter out splits with zero or negative local improvement.
    if local_improvement.is_zero() {
        return None;
    }

    // Compute global improvement by recomputing the schedule with
    // the target replaced by two halves. The downset only needs deps
    // that finish at or before the threshold — that's the whole point
    // of the split (the downset can start compiling earlier).
    let global_improvement = compute_global_improvement(
        ctx.schedule,
        ctx.target_id,
        downset_cost,
        upset_cost,
        threshold,
    );

    let downset_modules = collect_modules(ctx.intra, &downset_indices);
    let upset_modules = collect_modules(ctx.intra, &upset_indices);
    let all_indices: Vec<usize> = (0..ctx.intra.nodes.len()).collect();
    let split_modules = collect_modules(ctx.intra, &all_indices);

    Some(SplitCandidate {
        threshold,
        local_improvement,
        global_improvement,
        downset_scc_count: downset_indices.len(),
        downset_cost,
        upset_scc_count: upset_indices.len(),
        upset_cost,
        downset_modules,
        upset_modules,
        split_modules,
    })
}

/// Extracts meta and other timing components from target-level timings.
///
/// Meta = sum of `metadata_decode_*` events. Other = sum of non-metadata
/// events. These are used by the cost model to predict compilation time
/// for synthetic crate halves. Each half inherits the full target's
/// meta/other overhead (conservative estimate).
fn extract_meta_other(
    symbol_graph: &SymbolGraph,
    target_id: &str,
) -> (f64, f64) {
    let Some((pkg, tgt)) = target_id.split_once('/') else {
        return (0.0, 0.0);
    };
    let Some(package) = symbol_graph.packages.get(pkg) else {
        return (0.0, 0.0);
    };
    let Some(target) = package.targets.get(tgt) else {
        return (0.0, 0.0);
    };

    let meta: f64 = target
        .timings
        .event_times_ms
        .iter()
        .filter(|(k, _)| k.starts_with("metadata_decode_"))
        .map(|(_, v)| v)
        .sum();
    let other: f64 = target
        .timings
        .event_times_ms
        .iter()
        .filter(|(k, _)| !k.starts_with("metadata_decode_"))
        .map(|(_, v)| v)
        .sum();

    (meta, other)
}

/// Collects distinct module paths for a set of SCC indices.
///
/// Returns sorted, deduplicated module paths from the specified SCC
/// nodes. Used to populate the module breakdown fields in
/// `SplitCandidate`.
fn collect_modules(
    intra: &IntraTargetGraph,
    scc_indices: &[usize],
) -> Vec<String> {
    let mut modules: Vec<String> = scc_indices
        .iter()
        .map(|&i| intra.nodes[i].module_path.clone())
        .collect();
    modules.sort();
    modules.dedup();
    modules
}

/// Shatters a target by horizon, grouping SCCs by effective horizon.
///
/// Computes per-SCC effective horizons, then merges all SCCs that share
/// the same horizon into a single target. This produces one target per
/// distinct horizon value: SCCs that can all start at the same time are
/// batched together, giving realistic parallelism gains without the
/// overhead of one-target-per-SCC.
///
/// Each group target uses raw attr cost (sum of member SCC costs, no
/// per-target overhead). Dependencies are wired precisely:
/// - Each group inherits the union of external deps from its SCCs.
/// - Inter-group edges are derived from the intra-target SCC DAG.
/// - Dependents are rewired to the specific groups they reference.
///
/// Returns `None` if the target doesn't exist or has no symbols.
pub fn shatter_target(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    schedule: &ScheduleData,
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

    // Aggregate per-group cost and symbol count from member SCCs.
    let mut group_costs = vec![Duration::ZERO; group_count];
    let mut group_sym_counts = vec![0_usize; group_count];
    for node in &intra.nodes {
        let g = scc_to_group[node.id];
        group_costs[g] += node.cost;
        group_sym_counts[g] += node.symbols.len();
    }

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
    wire_group_external_deps(
        symbol_graph,
        target_id,
        target_prefix,
        intra,
        scc_to_group,
        group_count,
        group_base,
        &names,
        &mut graph,
    );

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
/// tree using the `export` helpers. The original target is removed.
///
/// This ensures `/api/tree/{pkg}/{target}::group_N` resolves correctly
/// after shattering.
fn build_shattered_symbol_graph(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    target_prefix: &str,
    intra: &IntraTargetGraph,
    scc_to_group: &[usize],
    group_count: usize,
    group_costs: &[Duration],
) -> SymbolGraph {
    use crate::export::{
        insert_symbol_into_module, parse_symbol_path,
        remove_symbol_from_module,
    };

    let mut graph = symbol_graph.clone();

    let Some((pkg_name, target_key)) = target_id.split_once('/') else {
        return graph;
    };

    // Create empty group targets up front.
    let mut group_roots: Vec<Module> = (0..group_count)
        .map(|_| Module::default())
        .collect();

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

    // Remove the original target and insert group targets.
    let original_deps = source_target.dependencies.clone();
    pkg.targets.remove(target_key);

    for (g, root) in group_roots.into_iter().enumerate() {
        let group_key = format!("{target_key}::group_{g}");
        let group_target = Target {
            timings: TargetTimings {
                wall_time: group_costs[g],
                event_times_ms: HashMap::new(),
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
/// Walks the shattered target's module tree to find per-symbol external
/// deps, aggregates them by group, then adds edges from those external
/// targets to the group targets.
#[expect(
    clippy::too_many_arguments,
    reason = "wiring helper needs symbol graph, SCC mapping, and graph builder context"
)]
fn wire_group_external_deps(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    target_prefix: &str,
    intra: &IntraTargetGraph,
    scc_to_group: &[usize],
    group_count: usize,
    group_base: usize,
    names: &indexmap::IndexSet<String>,
    graph: &mut petgraph::graph::DiGraph<usize, ()>,
) {
    use std::collections::HashSet;

    use petgraph::graph::NodeIndex;

    let Some((pkg_name, tgt_key)) = target_id.split_once('/') else {
        return;
    };
    let target_obj = &symbol_graph.packages[pkg_name].targets[tgt_key];
    let dep_set = &target_obj.dependencies;
    let root = &target_obj.root;
    let all_sym_deps = collect_symbol_external_targets(root, target_prefix, "");

    // Collect external targets per group from member SCCs' symbols.
    // Only include targets that are actual dependencies of the shattered
    // target. Symbol paths can reference non-dependency targets (via
    // re-exports, test/bench targets in the same package, etc.). Adding
    // edges from those targets would create cycles: a dependent of the
    // shattered target could get an incoming edge from a group AND an
    // outgoing edge to a group (via the last-group fallback), forming a
    // loop through the inter-group chain.
    let mut group_ext: Vec<HashSet<&str>> = vec![HashSet::new(); group_count];
    for node in &intra.nodes {
        let g = scc_to_group[node.id];
        for sym in &node.symbols {
            if let Some(targets) = all_sym_deps.get(sym.as_str()) {
                for t in targets {
                    if dep_set.contains(t.as_str()) {
                        group_ext[g].insert(t.as_str());
                    }
                }
            }
        }
    }

    for (g, ext_targets) in group_ext.iter().enumerate() {
        let group_idx = group_base + g;
        for &ext_target in ext_targets {
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
fn collect_symbol_external_targets(
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

/// Computes the global critical path improvement from splitting a target.
///
/// Builds a modified `TargetGraph` where the original target is replaced
/// by two halves (downset + upset), recomputes the schedule, and returns
/// the reduction in critical path length.
///
/// The downset half gets synthetic name `"{target_id}::downset"` and
/// the upset keeps the original target name. Wiring:
/// - Dependencies finishing at or before `threshold` -> downset only
/// - All original dependencies -> upset
/// - Downset -> upset edge (upset depends on downset)
/// - Both halves -> original target's dependents
///
/// The downset only needs the subset of dependencies whose finish times
/// are at or below the threshold — that's the scheduling advantage of the
/// split. The upset still needs all original deps (plus the downset).
fn compute_global_improvement(
    schedule: &ScheduleData,
    target_id: &str,
    downset_cost: Duration,
    upset_cost: Duration,
    threshold: Duration,
) -> Duration {
    use indexmap::IndexSet;
    use petgraph::graph::{DiGraph, NodeIndex};

    let original_cp = schedule.summary.critical_path;

    // Find the target index in the schedule.
    let Some(target_idx) =
        schedule.targets.iter().position(|t| t.name == target_id)
    else {
        return Duration::ZERO;
    };

    let target_data = &schedule.targets[target_idx];
    let mut names = IndexSet::new();
    let mut costs = Vec::new();
    let mut symbol_counts = Vec::new();

    // Add all existing targets, substituting costs for the split target.
    for (i, t) in schedule.targets.iter().enumerate() {
        if i == target_idx {
            // The upset keeps the original target name but with the
            // upset's cost.
            names.insert(t.name.clone());
            costs.push(upset_cost);
            symbol_counts.push(t.symbol_count);
        } else {
            names.insert(t.name.clone());
            costs.push(t.cost);
            symbol_counts.push(t.symbol_count);
        }
    }

    // Add the downset as a new target.
    let downset_name = format!("{target_id}::downset");
    names.insert(downset_name);
    costs.push(downset_cost);
    // Symbol count for synthetic half is approximate; doesn't affect
    // scheduling.
    symbol_counts.push(0);

    let total = names.len();
    let downset_idx = total - 1;

    // Build the dependency graph.
    let mut graph = DiGraph::<usize, ()>::with_capacity(total, 0);
    for i in 0..total {
        graph.add_node(i);
    }

    // Reconstruct original edges, excluding edges to/from the split target.
    for (i, t) in schedule.targets.iter().enumerate() {
        for &dep_idx in &t.deps {
            if i == target_idx || dep_idx == target_idx {
                continue; // Handle separately below.
            }
            graph.add_edge(NodeIndex::new(dep_idx), NodeIndex::new(i), ());
        }
    }

    // Wire the split target's dependencies. The upset gets ALL original
    // deps. The downset only gets deps that finish at or before the
    // threshold — the whole point of the split is that the downset can
    // start earlier because it doesn't need late-finishing deps.
    for &dep_idx in &target_data.deps {
        let dep_finish = schedule.targets[dep_idx].finish;

        // All deps -> upset (original position).
        graph.add_edge(NodeIndex::new(dep_idx), NodeIndex::new(target_idx), ());

        // Only early-finishing deps -> downset.
        if dep_finish <= threshold {
            graph.add_edge(
                NodeIndex::new(dep_idx),
                NodeIndex::new(downset_idx),
                (),
            );
        }
    }

    // Wire downset -> upset (upset depends on downset).
    graph.add_edge(NodeIndex::new(downset_idx), NodeIndex::new(target_idx), ());

    // Wire both halves -> original target's dependents.
    for &dep_idx in &target_data.dependents {
        // upset -> dependent (already handled via original position)
        graph.add_edge(NodeIndex::new(target_idx), NodeIndex::new(dep_idx), ());
        // downset -> dependent (downset is also a dep of dependents)
        graph.add_edge(
            NodeIndex::new(downset_idx),
            NodeIndex::new(dep_idx),
            (),
        );
    }

    let modified_tg = TargetGraph {
        names,
        costs,
        symbol_counts,
        graph,
    };

    let new_schedule = compute_schedule(&modified_tg);
    original_cp.saturating_sub(new_schedule.summary.critical_path)
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

    // =================================================================
    // Split recommendation tests
    // =================================================================

    #[test]
    fn recommend_produces_candidate_for_split_horizon() {
        // "a" depends on external dep finishing at 50ms, "b" has no
        // external deps. One candidate at threshold 0.0 separating
        // "b" (downset) from "a" (upset).
        let (sg, schedule) = make_graph_with_external_dep(
            &[("a", 10.0, &["[dep-a/lib]::foo"]), ("b", 20.0, &[])],
            50.0,
        );
        let result =
            compute_split_recommendations(&sg, "test-pkg/lib", &schedule, None);
        assert_eq!(result.target, "test-pkg/lib");
        assert_eq!(result.candidates.len(), 1);
        let c = &result.candidates[0];
        assert!(
            c.threshold.is_zero(),
            "threshold should be Duration::ZERO, got {:?}",
            c.threshold
        );
        assert_eq!(c.downset_scc_count, 1);
        assert_eq!(c.upset_scc_count, 1);
    }

    #[test]
    fn recommend_global_improvement_nonzero_for_critical_path() {
        // "b" has no external deps (horizon 0), "a" depends on dep-a
        // (horizon 50). The split downset ("b") can start at T=0
        // instead of waiting for dep-a. Since test-pkg/lib is on the
        // critical path, this should produce a global improvement.
        let (sg, schedule) = make_graph_with_external_dep(
            &[("a", 10.0, &["[dep-a/lib]::foo"]), ("b", 50.0, &[])],
            50.0,
        );
        let result =
            compute_split_recommendations(&sg, "test-pkg/lib", &schedule, None);
        assert_eq!(result.candidates.len(), 1);
        let c = &result.candidates[0];
        assert!(
            !c.global_improvement.is_zero(),
            "critical path split should have positive global improvement, \
             got {:?}",
            c.global_improvement
        );
        assert!(
            !c.local_improvement.is_zero(),
            "split should have positive local improvement, got {:?}",
            c.local_improvement
        );
    }

    #[test]
    fn recommend_empty_when_all_horizons_equal() {
        // Two independent symbols, no external deps. All horizons are 0.
        let (sg, schedule) = make_graph_with_external_dep(
            &[("a", 10.0, &[]), ("b", 20.0, &[])],
            50.0,
        );
        let result =
            compute_split_recommendations(&sg, "test-pkg/lib", &schedule, None);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn recommend_local_improvement_is_positive() {
        let (sg, schedule) = make_graph_with_external_dep(
            &[("a", 40.0, &["[dep-a/lib]::foo"]), ("b", 30.0, &[])],
            50.0,
        );
        let result =
            compute_split_recommendations(&sg, "test-pkg/lib", &schedule, None);
        assert_eq!(result.candidates.len(), 1);
        assert!(!result.candidates[0].local_improvement.is_zero());
    }

    /// Like `make_graph_with_external_dep` but allows custom target
    /// timings for cost model testing.
    #[expect(
        clippy::too_many_lines,
        reason = "test fixture construction is inherently verbose"
    )]
    fn make_graph_with_timings(
        syms: &[(&str, f64, &[&str])],
        dep_finish_ms: f64,
        timings: TargetTimings,
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
            timings,
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
    fn recommend_negative_improvement_filtered_out() {
        // With a cost model that has very high per-target overhead,
        // splitting tiny symbols into two crates doubles the overhead,
        // making the split worse than the original.
        let timings = TargetTimings {
            wall_time: Duration::ZERO,
            event_times_ms: HashMap::from([
                ("metadata_decode_entry".to_string(), 5.0),
                ("incr_comp_persist".to_string(), 5.0),
            ]),
        };
        let (sg, schedule) = make_graph_with_timings(
            &[("a", 1.0, &["[dep-a/lib]::foo"]), ("b", 1.0, &[])],
            50.0,
            timings,
        );
        // Cost model: each half costs attr*1 + meta*100 + other*100.
        // Original: predict(2.0, 5.0, 5.0) = 2 + 500 + 500 = 1002
        // Each half: predict(1.0, 5.0, 5.0) = 1 + 500 + 500 = 1001
        // Splitting produces two ~1001ms halves, worse than one ~1002ms.
        let model = CostModel {
            coeff_attr: 1.0,
            coeff_meta: 100.0,
            coeff_other: 100.0,
            r_squared: 0.9,
            inlier_threshold: 1.0,
        };
        let result = compute_split_recommendations(
            &sg,
            "test-pkg/lib",
            &schedule,
            Some(&model),
        );
        assert!(
            result.candidates.is_empty(),
            "high-overhead split should be filtered, got {} candidates",
            result.candidates.len()
        );
    }

    #[test]
    fn recommend_missing_target_returns_empty() {
        let (sg, schedule) =
            make_graph_with_external_dep(&[("a", 10.0, &[])], 50.0);
        let result = compute_split_recommendations(
            &sg,
            "nonexistent/lib",
            &schedule,
            None,
        );
        assert!(result.candidates.is_empty());
    }
}
