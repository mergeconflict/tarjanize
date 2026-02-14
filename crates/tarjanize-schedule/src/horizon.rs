//! Per-SCC effective horizon computation.
//!
//! Computes the earliest wall-clock time at which each SCC inside a target
//! could begin compiling, accounting for external dependency finish times
//! and transitive constraints through the intra-target DAG.
//!
//! Why: effective horizons determine which SCCs can start early and form
//! the basis of shatter grouping in the split explorer.

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::time::Duration;

use tarjanize_schemas::{Module, SymbolGraph, TargetId};

use crate::data::ScheduleData;
use crate::target_graph::IntraTargetGraph;

/// Computes a topological ordering of nodes in a DAG defined by edges.
///
/// Uses Kahn's algorithm: repeatedly removes sources (in-degree 0) until
/// all reachable nodes are ordered. Returns node indices in dependency
/// order (predecessors before dependents).
///
/// Also returns predecessor adjacency lists for propagation.
///
/// Why: horizon propagation requires processing nodes in dependency order
/// so that every predecessor's value is finalized before its dependents.
fn topological_order_with_predecessors(
    node_count: usize,
    edges: &[(usize, usize)],
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    let mut in_degree = vec![0_usize; node_count];

    for &(from, to) in edges {
        predecessors[to].push(from);
        successors[from].push(to);
        in_degree[to] += 1;
    }

    let mut queue: Vec<usize> =
        (0..node_count).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(node_count);

    while let Some(node) = queue.pop() {
        order.push(node);
        for &succ in &successors[node] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    (order, predecessors)
}

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
///
/// Why: effective horizons determine which SCCs can start early and form
/// the basis of shatter grouping.
pub fn compute_effective_horizons(
    intra: &IntraTargetGraph,
    symbol_graph: &SymbolGraph,
    target_id: &TargetId,
    schedule: &ScheduleData,
) -> Vec<Duration> {
    let n = intra.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Build a target name -> finish time lookup from the schedule,
    // restricted to targets that are actual dependencies of this target.
    // Symbol-level dependencies can reference targets outside the
    // target-level dependency set (e.g. benchmark/test targets in the
    // same package, or re-exported paths from non-dependencies). Including
    // those would inflate max_horizon beyond the target's real start time,
    // making shatter groups look later than they really are.
    let target_prefix = format!("[{target_id}]::");
    let target_obj =
        &symbol_graph.packages[target_id.package()].targets[target_id.target()];
    let dep_set = &target_obj.dependencies;
    let finish_times: HashMap<String, Duration> = schedule
        .targets
        .iter()
        .filter(|t| dep_set.contains(&t.name))
        .map(|t| (t.name.clone(), t.finish))
        .collect();

    // Walk the symbol graph's module tree to collect per-symbol max
    // external finish times. We need the original module tree because
    // `IntraTargetGraph` only stores symbol paths, not the dependency sets.
    let symbol_finish = collect_external_max_finish(
        &target_obj.root,
        &target_prefix,
        "",
        &finish_times,
    );

    // For each SCC, compute the direct horizon as the max external finish
    // time across all member symbols.
    let mut direct_horizon = vec![Duration::ZERO; n];
    for node in &intra.nodes {
        for sym in &node.symbols {
            if let Some(&ft) = symbol_finish.get(sym) {
                direct_horizon[node.id] = direct_horizon[node.id].max(ft);
            }
        }
    }

    // Topological sort + propagate: each SCC's effective horizon is the
    // max of its own direct horizon and the effective horizons of all
    // its predecessors.
    let (topo_order, predecessors) =
        topological_order_with_predecessors(n, &intra.edges);

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
///
/// Why: external dependencies refer to fully qualified symbol paths, so we
/// need to recover target IDs for schedule lookups.
pub(crate) fn extract_target_from_path(path: &str) -> Option<&str> {
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
///
/// Why: per-symbol external finish times drive horizon computation without
/// over-counting internal dependencies.
fn collect_external_max_finish<S: BuildHasher>(
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
