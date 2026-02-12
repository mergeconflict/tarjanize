//! Heat map computation for the intra-target SCC DAG.
//!
//! Runs forward/backward DP on the intra-target SCC DAG (analogous to
//! the target-level schedule in `schedule.rs`) to compute per-SCC slack.
//! Zero-slack SCCs are on the internal critical path -- splitting them
//! would reduce the target's compilation time. High-slack SCCs have room
//! to spare and are not bottlenecks.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tarjanize_schemas::serde_duration;

use crate::target_graph::IntraTargetGraph;

/// Heat map data for a single SCC node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SccHeat {
    /// SCC node index (matches `SccNode.id`).
    pub id: usize,
    /// Earliest start time within the intra-target schedule.
    #[serde(rename = "start_ms", with = "serde_duration")]
    pub start: Duration,
    /// Earliest finish time (start + cost).
    #[serde(rename = "finish_ms", with = "serde_duration")]
    pub finish: Duration,
    /// Slack: how much this SCC could be delayed without extending the
    /// target's internal critical path. Zero means on critical path.
    #[serde(rename = "slack_ms", with = "serde_duration")]
    pub slack: Duration,
    /// Whether this SCC is on the internal critical path.
    pub on_critical_path: bool,
    /// Estimated improvement if a downset split is made at this SCC.
    /// Only computed for critical-path SCCs; `None` for others.
    ///
    /// The design doc specifies this as: "for each SCC on the internal
    /// critical path, the backend tentatively evaluates splitting at
    /// that point (computing the downset as a new crate, recomputing
    /// the global schedule)."
    #[serde(
        rename = "improvement_ms",
        with = "serde_duration::option",
        skip_serializing_if = "Option::is_none"
    )]
    pub improvement: Option<Duration>,
}

/// Heat map data for an entire target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    /// Per-SCC heat data, indexed by SCC id.
    pub sccs: Vec<SccHeat>,
    /// The internal critical path length.
    #[serde(rename = "critical_path_ms", with = "serde_duration")]
    pub critical_path: Duration,
}

/// Computes the heat map for the intra-target SCC DAG.
///
/// Runs forward/backward DP identical to the target-level schedule
/// to assign earliest start/finish times and slack to each SCC node.
/// Uses exact `Duration` arithmetic — no epsilon-based comparisons.
pub fn compute_heatmap(graph: &IntraTargetGraph) -> HeatmapData {
    let n = graph.nodes.len();
    if n == 0 {
        return HeatmapData {
            sccs: Vec::new(),
            critical_path: Duration::ZERO,
        };
    }

    // Build adjacency lists. Edges are (from, to) where `from` is the
    // dependency and `to` is the dependent. We need both directions:
    // `dependents[i]` lists nodes that depend on i (outgoing edges),
    // `dependencies[i]` lists nodes that i depends on (incoming edges).
    let mut dependents: Vec<Vec<usize>> = vec![vec![]; n];
    let mut dependencies: Vec<Vec<usize>> = vec![vec![]; n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for &(from, to) in &graph.edges {
        dependents[from].push(to);
        dependencies[to].push(from);
        in_degree[to] += 1;
    }

    // Forward pass: Kahn's topological sort computing earliest
    // start/finish times. Nodes with zero in-degree start at time 0.
    let mut start = vec![Duration::ZERO; n];
    let mut finish = vec![Duration::ZERO; n];
    let mut topo_order: Vec<usize> = Vec::with_capacity(n);

    let mut queue: std::collections::VecDeque<usize> =
        std::collections::VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    while let Some(node) = queue.pop_front() {
        topo_order.push(node);

        // Start time = max finish time of all dependencies.
        let mut max_dep_finish = Duration::ZERO;
        for &dep in &dependencies[node] {
            max_dep_finish = max_dep_finish.max(finish[dep]);
        }

        start[node] = max_dep_finish;
        finish[node] = max_dep_finish + graph.nodes[node].cost;

        // Decrement in-degree of dependents; enqueue when zero.
        for &dependent in &dependents[node] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }

    let critical_path = finish.iter().copied().max().unwrap_or(Duration::ZERO);

    // Backward pass: compute longest path from each node to any sink.
    // Process in reverse topological order so dependents are computed
    // before their dependencies.
    let mut longest_from = vec![Duration::ZERO; n];

    for &node in topo_order.iter().rev() {
        let mut max_path = Duration::ZERO;
        for &dependent in &dependents[node] {
            let path_len =
                graph.nodes[dependent].cost + longest_from[dependent];
            max_path = max_path.max(path_len);
        }
        longest_from[node] = max_path;
    }

    // Compute slack via saturating subtraction (can't go negative).
    // Duration uses exact integer nanoseconds — no epsilon needed
    // for the critical path check.
    let sccs: Vec<SccHeat> = (0..n)
        .map(|i| {
            let slack = critical_path.saturating_sub(
                start[i] + graph.nodes[i].cost + longest_from[i],
            );
            SccHeat {
                id: graph.nodes[i].id,
                start: start[i],
                finish: finish[i],
                slack,
                on_critical_path: slack.is_zero(),
                // TODO: Compute improvement for critical-path SCCs
                // by tentatively evaluating the downset split.
                improvement: None,
            }
        })
        .collect();

    HeatmapData {
        sccs,
        critical_path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target_graph::SccNode;

    /// Helper to build an `IntraTargetGraph` from (id, `cost_ms`) pairs
    /// and edges. Uses `Duration::from_millis` for exact millisecond
    /// values (avoids f64-to-Duration conversion precision issues).
    fn make_graph(
        nodes: &[(usize, u64)],
        edges: &[(usize, usize)],
    ) -> IntraTargetGraph {
        IntraTargetGraph {
            nodes: nodes
                .iter()
                .map(|&(id, cost_ms)| SccNode {
                    id,
                    symbols: vec![format!("sym_{id}")],
                    module_path: String::new(),
                    cost: Duration::from_millis(cost_ms),
                })
                .collect(),
            edges: edges.to_vec(),
            modules: vec![],
        }
    }

    /// Shorthand for `Duration::from_millis` in assertions.
    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    #[test]
    fn heatmap_single_node() {
        let g = make_graph(&[(0, 10)], &[]);
        let h = compute_heatmap(&g);
        assert_eq!(h.sccs.len(), 1);
        assert!(
            h.sccs[0].slack.is_zero(),
            "single node should have zero slack"
        );
        assert!(
            h.sccs[0].on_critical_path,
            "single node should be on critical path"
        );
        assert_eq!(
            h.critical_path,
            ms(10),
            "critical path should equal the single node's cost"
        );
    }

    #[test]
    fn heatmap_chain_all_on_critical_path() {
        // Chain: A(10) -> B(20) -> C(5)
        // All on critical path. CP = 35ms.
        let g = make_graph(&[(0, 10), (1, 20), (2, 5)], &[(0, 1), (1, 2)]);
        let h = compute_heatmap(&g);
        assert_eq!(h.sccs.len(), 3);
        for scc in &h.sccs {
            assert!(
                scc.on_critical_path,
                "SCC {} should be on critical path",
                scc.id
            );
            assert!(
                scc.slack.is_zero(),
                "SCC {} should have zero slack",
                scc.id
            );
        }
        assert_eq!(h.critical_path, ms(35), "critical path should be 35ms");
    }

    #[test]
    #[expect(
        clippy::many_single_char_names,
        reason = "a/b/c are SCC nodes in the test graph; concise and clear"
    )]
    fn heatmap_fork_has_slack() {
        // Fork: A(10) -> B(20), A(10) -> C(5)
        // B is on critical path (cost 20 > 5). C has slack = 15.
        // CP = 10 + 20 = 30ms.
        let g = make_graph(&[(0, 10), (1, 20), (2, 5)], &[(0, 1), (0, 2)]);
        let h = compute_heatmap(&g);
        assert_eq!(h.critical_path, ms(30), "critical path should be 30ms");

        // A and B should be on critical path.
        let a = h.sccs.iter().find(|s| s.id == 0).unwrap();
        let b = h.sccs.iter().find(|s| s.id == 1).unwrap();
        let c = h.sccs.iter().find(|s| s.id == 2).unwrap();

        assert!(a.on_critical_path, "A should be on critical path");
        assert!(b.on_critical_path, "B should be on critical path");
        assert!(!c.on_critical_path, "C should not be on critical path");
        assert_eq!(c.slack, ms(15), "C slack should be 15ms");
    }

    #[test]
    fn heatmap_diamond() {
        // Diamond: A(5) -> B(10), A(5) -> C(3), B -> D(2), C -> D(2)
        // CP = A(5) + B(10) + D(2) = 17ms
        // C has slack = 10 - 3 = 7ms
        let g = make_graph(
            &[(0, 5), (1, 10), (2, 3), (3, 2)],
            &[(0, 1), (0, 2), (1, 3), (2, 3)],
        );
        let h = compute_heatmap(&g);
        assert_eq!(h.critical_path, ms(17), "critical path should be 17ms");

        let c = h.sccs.iter().find(|s| s.id == 2).unwrap();
        assert!(!c.on_critical_path, "C should not be on critical path");
        assert_eq!(c.slack, ms(7), "C slack should be 7ms");
    }

    #[test]
    fn heatmap_independent_nodes() {
        // Two independent nodes: A(10), B(5)
        // CP = max(10, 5) = 10ms
        // B has slack = 5.
        let g = make_graph(&[(0, 10), (1, 5)], &[]);
        let h = compute_heatmap(&g);
        assert_eq!(h.critical_path, ms(10), "critical path should be 10ms");

        let a = h.sccs.iter().find(|s| s.id == 0).unwrap();
        let b = h.sccs.iter().find(|s| s.id == 1).unwrap();
        assert!(a.on_critical_path, "A should be on critical path");
        assert!(!b.on_critical_path, "B should not be on critical path");
        assert_eq!(b.slack, ms(5), "B slack should be 5ms");
    }

    #[test]
    fn heatmap_start_times_are_correct() {
        // Chain: A(10) -> B(20)
        // A starts at 0, finishes at 10.
        // B starts at 10, finishes at 30.
        let g = make_graph(&[(0, 10), (1, 20)], &[(0, 1)]);
        let h = compute_heatmap(&g);

        let a = h.sccs.iter().find(|s| s.id == 0).unwrap();
        let b = h.sccs.iter().find(|s| s.id == 1).unwrap();

        assert_eq!(a.start, Duration::ZERO, "A should start at 0");
        assert_eq!(a.finish, ms(10), "A should finish at 10");
        assert_eq!(b.start, ms(10), "B should start at 10");
        assert_eq!(b.finish, ms(30), "B should finish at 30");
    }
}
