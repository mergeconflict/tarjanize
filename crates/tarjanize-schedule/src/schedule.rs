//! Build schedule computation: forward/backward DP, slack, swim lanes.
//!
//! Takes a target dependency graph with costs and produces a complete
//! [`ScheduleData`] for visualization. The scheduling is split into:
//!
//! 1. **Forward pass** — computes earliest start/finish times via DP
//! 2. **Backward pass** — computes latest start times and slack
//! 3. **Swim lane packing** — assigns parallel lanes via greedy bin-packing
//! 4. **Critical path reconstruction** — follows predecessor chain
//!
//! Why: a single, deterministic scheduler keeps the CLI and UI aligned and
//! makes critical-path analysis reproducible across tools.

use std::time::Duration;

use indexmap::IndexSet;
use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};

use crate::data::{ScheduleData, Summary, TargetData};

/// Target graph used for scheduling.
///
/// Abstracts away the `SymbolGraph` so scheduling logic can be tested
/// with simple hand-built graphs. The `lib.rs` entry point converts
/// `SymbolGraph + CostModel` into this representation.
///
/// Why: schedule algorithms should not depend on extraction internals, so we
/// isolate the minimal graph data they need.
#[derive(Debug)]
pub struct TargetGraph {
    /// Target identifiers in `{package}/{target}` format.
    pub names: IndexSet<String>,
    /// Predicted compilation cost per target.
    pub costs: Vec<Duration>,
    /// Symbol count per target.
    pub symbol_counts: Vec<usize>,
    /// Dependency graph where edges point from dependency → dependent.
    pub graph: DiGraph<usize, ()>,
}

/// Result of the forward scheduling pass.
///
/// Why: forward pass produces intermediate values reused by the backward
/// pass and critical-path reconstruction.
struct ForwardResult {
    start: Vec<Duration>,
    finish: Vec<Duration>,
    forward_pred: Vec<Option<usize>>,
    critical_path: Duration,
}

/// Result of the backward scheduling pass.
///
/// Why: backward pass needs its own container so slack and successor chains
/// remain consistent with the forward pass outputs.
struct BackwardResult {
    slack: Vec<Duration>,
    on_critical_path: Vec<bool>,
    backward_succ: Vec<Option<usize>>,
}

/// Computes the full build schedule from a target graph.
///
/// Runs forward DP, backward DP, swim lane packing, and critical path
/// reconstruction. Returns a [`ScheduleData`] ready for JSON embedding.
///
/// Panics if the graph contains cycles (caller must ensure DAG).
///
/// Why: this is the single entry point for schedule computation used by
/// both the CLI and viz.
pub fn compute_schedule(tg: &TargetGraph) -> ScheduleData {
    let n = tg.names.len();
    if n == 0 {
        return ScheduleData {
            summary: Summary {
                critical_path: Duration::ZERO,
                total_cost: Duration::ZERO,
                parallelism_ratio: 0.0,
                target_count: 0,
                symbol_count: 0,
                lane_count: 0,
            },
            targets: Vec::new(),
            critical_path: Vec::new(),
        };
    }

    // Topological sort — the graph must be a DAG.
    let sorted = toposort(&tg.graph, None)
        .expect("target dependency graph contains a cycle");

    let fwd = forward_pass(tg, &sorted);
    let bwd = backward_pass(tg, &sorted, &fwd);
    let (lanes, lane_count) = pack_swim_lanes(
        n,
        &fwd.start,
        &fwd.finish,
        &fwd.forward_pred,
        &tg.graph,
    );

    // Reconstruct global critical path by finding the target with latest
    // finish time and walking backwards via forward_pred.
    let end_idx = fwd
        .finish
        .iter()
        .enumerate()
        .max_by_key(|(_, d)| *d)
        .map(|(i, _)| i)
        .expect("graph has at least one target after early return");

    let mut critical_path = Vec::new();
    let mut current = Some(end_idx);
    while let Some(idx) = current {
        critical_path.push(idx);
        current = fwd.forward_pred[idx];
    }
    critical_path.reverse();

    // Build deps/dependents adjacency lists for the JSON output.
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &node in &sorted {
        let idx = node.index();
        for dep in tg.graph.neighbors_directed(node, Direction::Incoming) {
            deps[idx].push(dep.index());
            dependents[dep.index()].push(idx);
        }
    }

    // Assemble final output.
    let total_cost: Duration = tg.costs.iter().sum();
    let symbol_count: usize = tg.symbol_counts.iter().sum();
    let parallelism_ratio = if fwd.critical_path.is_zero() {
        0.0
    } else {
        total_cost.as_secs_f64() / fwd.critical_path.as_secs_f64()
    };

    let targets: Vec<TargetData> = (0..n)
        .map(|i| TargetData {
            name: tg
                .names
                .get_index(i)
                .expect("index i is in 0..n which is within names bounds")
                .clone(),
            start: fwd.start[i],
            finish: fwd.finish[i],
            cost: tg.costs[i],
            slack: bwd.slack[i],
            lane: lanes[i],
            symbol_count: tg.symbol_counts[i],
            deps: deps[i].clone(),
            dependents: dependents[i].clone(),
            on_critical_path: bwd.on_critical_path[i],
            forward_pred: fwd.forward_pred[i],
            backward_succ: bwd.backward_succ[i],
        })
        .collect();

    ScheduleData {
        summary: Summary {
            critical_path: fwd.critical_path,
            total_cost,
            parallelism_ratio,
            target_count: n,
            symbol_count,
            lane_count,
        },
        targets,
        critical_path,
    }
}

/// Forward scheduling pass: computes earliest start/finish times.
///
/// For each target t in topo order:
///
/// ```text
/// start[t]        = max(finish[dep] for dep in incoming), or 0
/// finish[t]       = start[t] + cost[t]
/// forward_pred[t] = argmax(finish[dep])
/// ```
///
/// Why: earliest-start times define the critical path and feed slack
/// computation in the backward pass.
fn forward_pass(tg: &TargetGraph, sorted: &[NodeIndex]) -> ForwardResult {
    let n = tg.names.len();
    let mut start = vec![Duration::ZERO; n];
    let mut finish = vec![Duration::ZERO; n];
    let mut forward_pred: Vec<Option<usize>> = vec![None; n];

    for &node in sorted {
        let idx = node.index();
        let mut max_dep_finish = Duration::ZERO;
        let mut max_dep: Option<usize> = None;

        for dep in tg.graph.neighbors_directed(node, Direction::Incoming) {
            let dep_idx = dep.index();
            if finish[dep_idx] > max_dep_finish {
                max_dep_finish = finish[dep_idx];
                max_dep = Some(dep_idx);
            }
        }

        start[idx] = max_dep_finish;
        finish[idx] = max_dep_finish + tg.costs[idx];
        forward_pred[idx] = max_dep;
    }

    let critical_path = finish.iter().copied().max().unwrap_or(Duration::ZERO);

    ForwardResult {
        start,
        finish,
        forward_pred,
        critical_path,
    }
}

/// Backward scheduling pass: computes slack and backward successor chain.
///
/// For each target t in reverse topo order:
///
/// ```text
/// longest_from[t] = max(cost[dep] + longest_from[dep]) for outgoing
/// backward_succ[t] = argmax of that
/// slack[t]         = critical_path_ms - (start[t] + cost[t] + longest_from[t])
/// ```
///
/// Why: slack and backward successors drive critical-path highlighting and
/// schedule sensitivity in the UI.
fn backward_pass(
    tg: &TargetGraph,
    sorted: &[NodeIndex],
    fwd: &ForwardResult,
) -> BackwardResult {
    let n = tg.names.len();
    let mut longest_from = vec![Duration::ZERO; n];
    let mut backward_succ: Vec<Option<usize>> = vec![None; n];

    for &node in sorted.iter().rev() {
        let idx = node.index();
        let mut max_path = Duration::ZERO;
        let mut max_succ: Option<usize> = None;

        for dependent in tg.graph.neighbors_directed(node, Direction::Outgoing)
        {
            let dep_idx = dependent.index();
            let path_len = tg.costs[dep_idx] + longest_from[dep_idx];
            if path_len > max_path {
                max_path = path_len;
                max_succ = Some(dep_idx);
            }
        }

        longest_from[idx] = max_path;
        backward_succ[idx] = max_succ;
    }

    // Slack: how much a target can shift without affecting critical path.
    // Duration uses exact integer nanoseconds, so no epsilon needed.
    let slack: Vec<Duration> = (0..n)
        .map(|i| {
            fwd.critical_path
                .saturating_sub(fwd.start[i] + tg.costs[i] + longest_from[i])
        })
        .collect();
    let on_critical_path: Vec<bool> =
        slack.iter().map(Duration::is_zero).collect();

    BackwardResult {
        slack,
        on_critical_path,
        backward_succ,
    }
}

/// Assigns swim lanes via dependency-aware greedy bin-packing.
///
/// Like classic interval-scheduling, targets are sorted by start time
/// and reuse lanes whose last occupant has finished. The difference:
/// instead of arbitrarily picking the earliest-finishing lane, we pick
/// the available lane closest to the forward predecessor's lane. This
/// keeps dependency edges short without changing the lane count — the
/// count is determined by peak parallelism (how often zero lanes are
/// available), which is the same regardless of which available lane we
/// pick.
///
/// As a soft preference, targets avoid sharing a lane with any direct
/// dependency so that individual targets remain visually distinct.
/// This never increases lane count — if all available lanes belong to
/// direct dependencies, we pick the best one anyway.
///
/// Why: lane assignment is purely for visualization, so we optimize for
/// readability without changing overall parallelism.
fn pack_swim_lanes(
    n: usize,
    start: &[Duration],
    finish: &[Duration],
    forward_pred: &[Option<usize>],
    graph: &DiGraph<usize, ()>,
) -> (Vec<usize>, usize) {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        start[a].cmp(&start[b]).then(finish[a].cmp(&finish[b]))
    });

    let mut lanes: Vec<usize> = vec![0; n];
    // Last finish time per lane. A lane is available when its last
    // finish <= the target's start time.
    let mut lane_last_finish: Vec<Duration> = Vec::new();

    for &idx in &order {
        // Ideal lane: lane of the dependency that gates our start time.
        // Root targets (no deps) prefer lane 0.
        let ideal = forward_pred[idx].map_or(0, |pred| lanes[pred]);

        // Lanes used by direct dependencies. We prefer to avoid these
        // so each target is visually distinct, but we won't allocate
        // extra lanes to enforce it — lane count = peak parallelism.
        let forbidden: Vec<usize> = graph
            .neighbors_directed(NodeIndex::new(idx), Direction::Incoming)
            .map(|dep| lanes[dep.index()])
            .collect();

        // Find the available lane closest to ideal. Prefer non-forbidden
        // lanes, but fall back to forbidden ones over allocating.
        let mut best_lane = None;
        let mut best_dist = usize::MAX;
        let mut best_is_forbidden = true;

        for (lane, &lf) in lane_last_finish.iter().enumerate() {
            if lf <= start[idx] {
                let is_forbidden = forbidden.contains(&lane);
                let dist = lane.abs_diff(ideal);
                // Prefer non-forbidden over forbidden, then closer
                // to ideal, then lower lane number.
                let dominated = match (is_forbidden, best_is_forbidden) {
                    (false, true) => true,
                    (true, false) => false,
                    _ => {
                        dist < best_dist
                            || (dist == best_dist
                                && lane < best_lane.unwrap_or(usize::MAX))
                    }
                };
                if dominated {
                    best_lane = Some(lane);
                    best_dist = dist;
                    best_is_forbidden = is_forbidden;
                }
            }
        }

        let assigned = best_lane.unwrap_or_else(|| {
            // No available lane at all — allocate a new one.
            let new_lane = lane_last_finish.len();
            lane_last_finish.push(Duration::ZERO);
            new_lane
        });

        lanes[idx] = assigned;
        lane_last_finish[assigned] = finish[idx];
    }

    (lanes, lane_last_finish.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Converts f64 milliseconds to Duration for test convenience.
    ///
    /// Why: keeps test inputs readable while matching production units.
    fn ms(val: f64) -> Duration {
        Duration::from_secs_f64(val / 1000.0)
    }

    /// Builds a `TargetGraph` from target triples and edge pairs.
    /// Costs are specified in milliseconds for readability.
    ///
    /// Why: provides compact fixtures for schedule tests without
    /// depending on the full `SymbolGraph` pipeline.
    fn make_target_graph(
        targets: &[(&str, f64, usize)],
        edges: &[(usize, usize)],
    ) -> TargetGraph {
        let mut names = IndexSet::new();
        let mut costs = Vec::new();
        let mut symbol_counts = Vec::new();
        let mut graph = DiGraph::new();

        for (name, cost, syms) in targets {
            names.insert(name.to_string());
            costs.push(ms(*cost));
            symbol_counts.push(*syms);
            graph.add_node(names.len() - 1);
        }

        for &(from, to) in edges {
            graph.add_edge(NodeIndex::new(from), NodeIndex::new(to), ());
        }

        TargetGraph {
            names,
            costs,
            symbol_counts,
            graph,
        }
    }

    // =====================================================================
    // Empty graph
    // =====================================================================

    /// Empty graphs should produce empty schedules.
    ///
    /// Why: the scheduler must gracefully handle degenerate input.
    #[test]
    fn empty_graph_produces_empty_schedule() {
        let tg = make_target_graph(&[], &[]);
        let result = compute_schedule(&tg);

        assert_eq!(result.targets.len(), 0);
        assert_eq!(result.critical_path.len(), 0);
        assert_eq!(result.summary.critical_path, Duration::ZERO);
        assert_eq!(result.summary.target_count, 0);
        assert_eq!(result.summary.lane_count, 0);
    }

    // =====================================================================
    // Single target
    // =====================================================================

    /// Single targets should start at zero with no slack.
    ///
    /// Why: baseline behavior anchors later multi-target expectations.
    #[test]
    fn single_target_starts_at_zero_with_no_slack() {
        let tg = make_target_graph(&[("a/lib", 100.0, 5)], &[]);
        let result = compute_schedule(&tg);

        assert_eq!(result.targets.len(), 1);
        let t = &result.targets[0];

        assert_eq!(t.start, Duration::ZERO);
        assert_eq!(t.finish, ms(100.0));
        assert_eq!(t.cost, ms(100.0));
        assert_eq!(t.slack, Duration::ZERO);
        assert_eq!(t.lane, 0);
        assert!(t.on_critical_path);
        assert_eq!(t.forward_pred, None);
        assert_eq!(t.backward_succ, None);
        assert_eq!(t.symbol_count, 5);

        assert_eq!(result.critical_path, vec![0]);
        assert_eq!(result.summary.critical_path, ms(100.0));
        assert_eq!(result.summary.lane_count, 1);
    }

    // =====================================================================
    // Chain: A(50) → B(30) → C(20), all on critical path
    // =====================================================================

    /// A linear chain should schedule sequentially with no slack.
    ///
    /// Why: verifies critical-path propagation in the simplest DAG.
    #[test]
    fn chain_all_critical_sequential_starts() {
        // A(50) → B(30) → C(20)
        // Edges: A→B, B→C (dependency → dependent)
        let tg = make_target_graph(
            &[("a/lib", 50.0, 3), ("b/lib", 30.0, 2), ("c/lib", 20.0, 1)],
            &[(0, 1), (1, 2)],
        );
        let result = compute_schedule(&tg);

        // A: start=0, finish=50
        assert_eq!(result.targets[0].start, Duration::ZERO);
        assert_eq!(result.targets[0].finish, ms(50.0));

        // B: start=50, finish=80
        assert_eq!(result.targets[1].start, ms(50.0));
        assert_eq!(result.targets[1].finish, ms(80.0));

        // C: start=80, finish=100
        assert_eq!(result.targets[2].start, ms(80.0));
        assert_eq!(result.targets[2].finish, ms(100.0));

        // All on critical path, zero slack.
        for t in &result.targets {
            assert!(t.on_critical_path, "{} should be critical", t.name);
            assert!(t.slack.is_zero(), "{} should have zero slack", t.name);
        }

        // forward_pred chain: C←B←A
        assert_eq!(result.targets[0].forward_pred, None);
        assert_eq!(result.targets[1].forward_pred, Some(0));
        assert_eq!(result.targets[2].forward_pred, Some(1));

        // backward_succ chain: A→B→C
        assert_eq!(result.targets[0].backward_succ, Some(1));
        assert_eq!(result.targets[1].backward_succ, Some(2));
        assert_eq!(result.targets[2].backward_succ, None);

        assert_eq!(result.summary.critical_path, ms(100.0));
        assert_eq!(result.critical_path, vec![0, 1, 2]);
    }

    // =====================================================================
    // Fork: A(100) → B(30), A(100) → C(10)
    // B has less slack than C; both depend on A.
    // =====================================================================

    /// Parallel forks should allocate lanes and compute slack correctly.
    ///
    /// Why: ensures slack and lane assignment reflect parallel branches.
    #[test]
    fn fork_parallel_lanes_correct_slack() {
        // A(100) is the root. B(30) and C(10) depend on A.
        let tg = make_target_graph(
            &[("a/lib", 100.0, 5), ("b/lib", 30.0, 3), ("c/lib", 10.0, 1)],
            &[(0, 1), (0, 2)],
        );
        let result = compute_schedule(&tg);

        // A: start=0, finish=100
        assert_eq!(result.targets[0].start, Duration::ZERO);
        assert_eq!(result.targets[0].finish, ms(100.0));

        // B: start=100, finish=130 — on critical path
        assert_eq!(result.targets[1].start, ms(100.0));
        assert_eq!(result.targets[1].finish, ms(130.0));
        assert!(result.targets[1].on_critical_path);
        assert!(result.targets[1].slack.is_zero());

        // C: start=100, finish=110 — NOT on critical path
        assert_eq!(result.targets[2].start, ms(100.0));
        assert_eq!(result.targets[2].finish, ms(110.0));
        // slack = 130 - (100 + 10 + 0) = 20
        assert_eq!(result.targets[2].slack, ms(20.0));
        assert!(!result.targets[2].on_critical_path);

        // Critical path = A → B
        assert_eq!(result.summary.critical_path, ms(130.0));

        // B and C run in parallel after A, so they should get different
        // lanes (they start at the same time).
        assert_ne!(
            result.targets[1].lane, result.targets[2].lane,
            "B and C overlap in time, must be in different lanes"
        );
    }

    // =====================================================================
    // Diamond: A→B, A→C, B→D, C→D
    // Tests forward_pred/backward_succ through a join point.
    // =====================================================================

    /// Diamond shapes should preserve predecessor/successor chains.
    ///
    /// Why: validates join handling for critical-path reconstruction.
    #[test]
    fn diamond_correct_pred_succ() {
        // A(10) → B(100), A(10) → C(50), B→D(20), C→D(20)
        // Critical: A(10) → B(100) → D(20) = 130
        // C has slack: 130 - (10 + 50 + 20) = 50
        let tg = make_target_graph(
            &[
                ("a/lib", 10.0, 1),
                ("b/lib", 100.0, 5),
                ("c/lib", 50.0, 3),
                ("d/lib", 20.0, 2),
            ],
            &[(0, 1), (0, 2), (1, 3), (2, 3)],
        );
        let result = compute_schedule(&tg);

        // A: start=0, finish=10
        assert_eq!(result.targets[0].finish, ms(10.0));

        // B: start=10, finish=110
        assert_eq!(result.targets[1].start, ms(10.0));
        assert_eq!(result.targets[1].finish, ms(110.0));

        // C: start=10, finish=60
        assert_eq!(result.targets[2].start, ms(10.0));
        assert_eq!(result.targets[2].finish, ms(60.0));

        // D: start=110 (waits for B), finish=130
        assert_eq!(result.targets[3].start, ms(110.0));
        assert_eq!(result.targets[3].finish, ms(130.0));

        // D's forward_pred is B (B finishes later than C).
        assert_eq!(result.targets[3].forward_pred, Some(1));

        // A's backward_succ is B (longer path through B than C).
        assert_eq!(result.targets[0].backward_succ, Some(1));

        // B's backward_succ is D.
        assert_eq!(result.targets[1].backward_succ, Some(3));

        // C's backward_succ is D (only dependent).
        assert_eq!(result.targets[2].backward_succ, Some(3));

        // C has slack: critical(130) - (start(10) + cost(50) + from(20))
        assert_eq!(result.targets[2].slack, ms(50.0));

        // A, B, D on critical path; C is not.
        assert!(result.targets[0].on_critical_path);
        assert!(result.targets[1].on_critical_path);
        assert!(!result.targets[2].on_critical_path);
        assert!(result.targets[3].on_critical_path);

        assert_eq!(result.summary.critical_path, ms(130.0));
    }

    // =====================================================================
    // Swim lane packing: peak parallelism = lane count
    // =====================================================================

    /// Lane count should match peak parallelism for independent targets.
    ///
    /// Why: verifies that lane packing reflects true parallel capacity.
    #[test]
    fn swim_lanes_equal_peak_parallelism() {
        // Three independent targets — peak parallelism = 3.
        let tg = make_target_graph(
            &[
                ("a/lib", 100.0, 1),
                ("b/lib", 100.0, 1),
                ("c/lib", 100.0, 1),
            ],
            &[],
        );
        let result = compute_schedule(&tg);

        assert_eq!(
            result.summary.lane_count, 3,
            "3 independent targets = 3 lanes"
        );

        // All three should have distinct lanes.
        let mut lane_set: Vec<usize> =
            result.targets.iter().map(|t| t.lane).collect();
        lane_set.sort_unstable();
        lane_set.dedup();
        assert_eq!(lane_set.len(), 3);
    }

    /// Sequential chains should reuse a single lane.
    ///
    /// Why: ensures lane packing does not inflate lane count.
    #[test]
    fn swim_lanes_reuse_when_sequential() {
        // A(100) → B(100): sequential, only 1 lane needed.
        let tg = make_target_graph(
            &[("a/lib", 100.0, 1), ("b/lib", 100.0, 1)],
            &[(0, 1)],
        );
        let result = compute_schedule(&tg);

        assert_eq!(
            result.summary.lane_count, 1,
            "sequential chain needs only 1 lane"
        );
        assert_eq!(result.targets[0].lane, result.targets[1].lane);
    }

    /// Mixed parallelism should reuse lanes after joins.
    ///
    /// Why: validates that lanes reflect peak concurrency, not total targets.
    #[test]
    fn swim_lanes_mixed_parallelism() {
        // A(100) → C(50), B(100) → C(50)
        // A and B are parallel (2 lanes), C follows both (reuses a lane).
        let tg = make_target_graph(
            &[("a/lib", 100.0, 1), ("b/lib", 100.0, 1), ("c/lib", 50.0, 1)],
            &[(0, 2), (1, 2)],
        );
        let result = compute_schedule(&tg);

        assert_eq!(
            result.summary.lane_count, 2,
            "peak parallelism is 2 (A and B), C reuses a lane"
        );
    }

    // =====================================================================
    // Summary statistics
    // =====================================================================

    /// Summary stats should reflect total cost and critical path.
    ///
    /// Why: the UI and reports rely on these aggregates for accuracy.
    #[test]
    fn summary_statistics_correct() {
        // A(100), B(50), independent. Total=150, critical=100, ratio=1.5.
        let tg =
            make_target_graph(&[("a/lib", 100.0, 10), ("b/lib", 50.0, 5)], &[]);
        let result = compute_schedule(&tg);

        assert_eq!(result.summary.total_cost, ms(150.0));
        assert_eq!(result.summary.critical_path, ms(100.0));
        assert!((result.summary.parallelism_ratio - 1.5).abs() < 1e-6);
        assert_eq!(result.summary.target_count, 2);
        assert_eq!(result.summary.symbol_count, 15);
    }

    // =====================================================================
    // Deps and dependents lists
    // =====================================================================

    /// Dep/dependent lists should be populated from the graph.
    ///
    /// Why: downstream UI uses these lists for navigation and highlighting.
    #[test]
    fn deps_and_dependents_populated() {
        // A → B, A → C
        let tg = make_target_graph(
            &[("a/lib", 10.0, 1), ("b/lib", 20.0, 1), ("c/lib", 30.0, 1)],
            &[(0, 1), (0, 2)],
        );
        let result = compute_schedule(&tg);

        // A has no deps, two dependents.
        assert!(result.targets[0].deps.is_empty());
        assert_eq!(result.targets[0].dependents.len(), 2);

        // B depends on A.
        assert_eq!(result.targets[1].deps, vec![0]);

        // C depends on A.
        assert_eq!(result.targets[2].deps, vec![0]);
    }
}
