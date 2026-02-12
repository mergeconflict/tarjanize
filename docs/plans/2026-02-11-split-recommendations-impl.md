# Split Recommendations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the horizon-threshold split recommendation algorithm and expose it via an API endpoint with interactive frontend.

**Architecture:** New `recommend.rs` module in `tarjanize-schedule` implements the core algorithm (effective horizon computation + candidate generation). A `GET /api/splits/{pkg}/{target}` endpoint in `tarjanize-viz` serves ranked candidates. Frontend migrates to TypeScript; the sidebar shows recommendations when a Gantt bar is clicked; clicking a candidate previews the modified schedule.

**Tech Stack:** Rust (petgraph, serde, axum), TypeScript, esbuild, PixiJS

**Skills:** Every task MUST use `/rust-development` for Rust code and `/verification-before-completion` before marking done. Backend tasks (1–4) MUST use `/test-driven-development`. Frontend tasks (5–7) should use manual verification via `cargo build` + browser testing.

---

### Task 1: Add anchor back-edges to `condense_target`

Port the anchor constraint logic from `tarjanize-condense/src/scc.rs` (step 2b, lines 281–315) into `tarjanize-schedule/src/target_graph.rs::condense_target()`. Without this, impl blocks and their trait/type anchors can land on opposite sides of a threshold cut, producing impossible splits.

**Files:**
- Modify: `crates/tarjanize-schedule/src/target_graph.rs`
- Test: same file, `#[cfg(test)] mod tests`

**Step 1: Write failing test — impl and anchor merge into same SCC**

Add a test that creates a target with an impl symbol whose anchor is a struct in the same target. Without back-edges, they'd be in separate SCCs. With back-edges, they must be in the same SCC.

```rust
#[test]
fn condense_target_merges_impl_with_anchor() {
    // Create a target with:
    //   - Struct "Foo" (no deps)
    //   - Impl "{{impl}}[0]" with anchor "Foo", depends on Foo
    // Without anchor back-edges: two SCCs (Foo, impl).
    // With anchor back-edges: one SCC containing both.
    let prefix = "[test-pkg/lib]::";
    let mut symbols = HashMap::new();
    symbols.insert(
        "Foo".to_string(),
        Symbol {
            file: "lib.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), 1.0)]),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Struct".to_string(),
                visibility: Visibility::Public,
            },
        },
    );
    symbols.insert(
        "{{impl}}[0]".to_string(),
        Symbol {
            file: "lib.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), 2.0)]),
            dependencies: HashSet::from([format!("{prefix}Foo")]),
            kind: SymbolKind::Impl {
                name: "Foo".to_string(),
                anchors: HashSet::from([format!("{prefix}Foo")]),
            },
        },
    );

    let root = Module {
        symbols,
        submodules: HashMap::new(),
    };
    let target = Target {
        timings: TargetTimings::default(),
        dependencies: HashSet::new(),
        root,
    };
    let mut targets = HashMap::new();
    targets.insert("lib".to_string(), target);
    let mut packages = HashMap::new();
    packages.insert("test-pkg".to_string(), Package { targets });
    let sg = SymbolGraph { packages };

    let graph = condense_target(&sg, "test-pkg/lib").unwrap();

    // Both symbols must be in the same SCC.
    assert_eq!(graph.nodes.len(), 1, "impl and anchor should be merged");
    assert_eq!(graph.nodes[0].symbols.len(), 2);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run -p tarjanize-schedule condense_target_merges_impl_with_anchor`
Expected: FAIL — currently 2 SCCs because no back-edge is added.

**Step 3: Add `collect_impl_anchors` helper and modify `condense_target`**

Add a new helper function that walks the module tree and collects `(symbol_path, anchors)` pairs for Impl symbols:

```rust
/// Collects impl anchor information from the module tree.
///
/// Returns `(impl_path, anchor_paths)` for each `Impl` symbol. Used to
/// add synthetic back-edges before SCC computation, ensuring impls and
/// their anchors land in the same SCC (orphan rule constraint).
fn collect_impl_anchors(
    module: &Module,
    target_prefix: &str,
    module_path: &str,
) -> Vec<(String, HashSet<String>)> {
    let mut result = Vec::new();

    for (name, symbol) in &module.symbols {
        if let SymbolKind::Impl { anchors, .. } = &symbol.kind {
            let full_path = if module_path.is_empty() {
                format!("{target_prefix}{name}")
            } else {
                format!("{target_prefix}{module_path}::{name}")
            };
            result.push((full_path, anchors.clone()));
        }
    }

    for (submod_name, submod) in &module.submodules {
        let child_path = if module_path.is_empty() {
            submod_name.clone()
        } else {
            format!("{module_path}::{submod_name}")
        };
        result.extend(collect_impl_anchors(submod, target_prefix, &child_path));
    }

    result
}
```

Then, in `condense_target`, after building the initial graph and before calling `condensation()`, add:

```rust
// Add synthetic back-edges for anchor constraints (orphan rule).
// For each impl, add a back-edge from its least-connected anchor to the
// impl. This creates a cycle that condensation collapses into a single
// SCC, ensuring impl and anchor can never be separated by a split.
// Ported from tarjanize-condense/src/scc.rs step 2b.
let impl_anchors_list =
    collect_impl_anchors(&target.root, &target_prefix, "");
for (impl_path, anchors) in &impl_anchors_list {
    let Some(&impl_idx) = path_to_idx.get(impl_path.as_str()) else {
        continue;
    };
    // Find valid anchors: those within this target (in the index).
    let valid: Vec<usize> = anchors
        .iter()
        .filter_map(|a| path_to_idx.get(a.as_str()).copied())
        .collect();
    if valid.is_empty() {
        continue;
    }
    // Choose anchor with minimum in-degree (rough proxy for "niche").
    let best = valid
        .iter()
        .copied()
        .min_by_key(|&idx| {
            graph
                .neighbors_directed(nodes[idx], petgraph::Direction::Incoming)
                .count()
        })
        .expect("valid is non-empty");
    // Back-edge: anchor → impl (creates cycle anchor → ... → impl → anchor).
    graph.add_edge(nodes[best], nodes[impl_idx], ());
}
```

This requires adding `use petgraph::Direction;` if not already imported.

**Step 4: Run test to verify it passes**

Run: `cargo nextest run -p tarjanize-schedule condense_target_merges_impl_with_anchor`
Expected: PASS

**Step 5: Write test — impl with external anchor is not affected**

Add a test where the impl's anchor is in a different target (external). The back-edge should NOT be added because the anchor isn't in `path_to_idx`.

```rust
#[test]
fn condense_target_external_anchor_no_back_edge() {
    // Impl with anchor pointing to another target — should NOT merge.
    let prefix = "[test-pkg/lib]::";
    let mut symbols = HashMap::new();
    symbols.insert(
        "Bar".to_string(),
        Symbol {
            file: "lib.rs".to_string(),
            event_times_ms: HashMap::new(),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Struct".to_string(),
                visibility: Visibility::Public,
            },
        },
    );
    symbols.insert(
        "{{impl}}[0]".to_string(),
        Symbol {
            file: "lib.rs".to_string(),
            event_times_ms: HashMap::new(),
            dependencies: HashSet::from([format!("{prefix}Bar")]),
            kind: SymbolKind::Impl {
                name: "ExternalTrait for Bar".to_string(),
                anchors: HashSet::from([
                    "[other-pkg/lib]::ExternalTrait".to_string(),
                    format!("{prefix}Bar"),
                ]),
            },
        },
    );

    let root = Module {
        symbols,
        submodules: HashMap::new(),
    };
    let target = Target {
        timings: TargetTimings::default(),
        dependencies: HashSet::new(),
        root,
    };
    let mut targets = HashMap::new();
    targets.insert("lib".to_string(), target);
    let mut packages = HashMap::new();
    packages.insert("test-pkg".to_string(), Package { targets });
    let sg = SymbolGraph { packages };

    let graph = condense_target(&sg, "test-pkg/lib").unwrap();

    // Bar is the only valid intra-target anchor. The impl depends on Bar,
    // and the back-edge anchor→impl creates a cycle, so they merge.
    assert_eq!(graph.nodes.len(), 1);
}
```

**Step 6: Run test**

Run: `cargo nextest run -p tarjanize-schedule condense_target_external_anchor`
Expected: PASS (Bar is a valid intra-target anchor, so it merges)

**Step 7: Run full test suite + clippy**

Run: `cargo nextest run -p tarjanize-schedule && cargo clippy -p tarjanize-schedule --all-targets`
Expected: All pass, no warnings.

**Step 8: Commit**

```bash
jj describe -m "Add anchor back-edges to condense_target for orphan rule compliance"
jj new
```

---

### Task 2: Effective horizon computation

Create `crates/tarjanize-schedule/src/recommend.rs` with the effective horizon algorithm. This module computes per-SCC effective horizons by resolving external dependencies to finish times and propagating through the SCC DAG.

**Files:**
- Create: `crates/tarjanize-schedule/src/recommend.rs`
- Modify: `crates/tarjanize-schedule/src/lib.rs` (add `pub mod recommend;`)

**Step 1: Register the new module**

Add `pub mod recommend;` to `crates/tarjanize-schedule/src/lib.rs` alongside the other module declarations.

**Step 2: Write failing test — effective horizons on a simple chain**

Create `recommend.rs` with types, the function signature, and tests. The function should compute effective horizons for each SCC in an intra-target graph.

```rust
use std::collections::{HashMap, HashSet};

use crate::data::ScheduleData;
use crate::target_graph::{condense_target, IntraTargetGraph, SccNode};

use serde::Serialize;
use tarjanize_schemas::symbol_graph::SymbolGraph;

/// Per-SCC effective horizon: the earliest time this SCC could start
/// compiling, given its external dependencies and predecessors.
///
/// Computed by propagating finish times through the SCC DAG in
/// topological order. See the design doc for the recurrence:
/// ```text
/// effective_horizon(s) = max(
///     max(finish(e) for e in s.external_deps),
///     max(effective_horizon(pred) for pred in s.predecessors)
/// )
/// ```
fn compute_effective_horizons(
    intra: &IntraTargetGraph,
    symbol_graph: &SymbolGraph,
    target_id: &str,
    schedule: &ScheduleData,
) -> Vec<f64> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use tarjanize_schemas::symbol_graph::*;
    use crate::data::*;

    /// Builds a SymbolGraph with two targets:
    /// - "dep-a/lib" (external dependency)
    /// - "test-pkg/lib" with the given symbols
    ///
    /// Returns (symbol_graph, schedule) where the schedule has dep-a
    /// finishing at `dep_finish_ms`.
    fn make_graph_with_external_dep(
        syms: &[(&str, f64, &[&str])],
        dep_finish_ms: f64,
    ) -> (SymbolGraph, ScheduleData) {
        let prefix = "[test-pkg/lib]::";
        let mut symbols = HashMap::new();
        for &(name, cost, deps) in syms {
            let dep_set: HashSet<String> =
                deps.iter().map(|d| d.to_string()).collect();
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

        // Build a minimal schedule with dep-a finishing at dep_finish_ms.
        let schedule = ScheduleData {
            summary: Summary {
                critical_path_ms: dep_finish_ms + 100.0,
                total_cost_ms: dep_finish_ms + 100.0,
                parallelism_ratio: 1.0,
                target_count: 2,
                symbol_count: syms.len(),
                lane_count: 1,
            },
            targets: vec![
                TargetData {
                    name: "dep-a/lib".to_string(),
                    start: 0.0,
                    finish: dep_finish_ms,
                    cost: dep_finish_ms,
                    slack: 0.0,
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
                    start: dep_finish_ms,
                    finish: dep_finish_ms + 100.0,
                    cost: 100.0,
                    slack: 0.0,
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
        // Two independent symbols, no external deps.
        // Both should have effective horizon 0.
        let (sg, schedule) = make_graph_with_external_dep(
            &[
                ("a", 10.0, &[]),
                ("b", 20.0, &[]),
            ],
            50.0,
        );
        let intra = condense_target(&sg, "test-pkg/lib").unwrap();
        let horizons =
            compute_effective_horizons(&intra, &sg, "test-pkg/lib", &schedule);
        assert_eq!(horizons.len(), intra.nodes.len());
        for h in &horizons {
            assert_eq!(*h, 0.0, "no external deps means horizon = 0");
        }
    }

    #[test]
    fn horizon_reflects_external_dep_finish_time() {
        // Symbol "a" depends on [dep-a/lib]::foo (external).
        // Symbol "b" has no external deps.
        // "a"'s effective horizon = dep-a's finish time (50.0).
        // "b"'s effective horizon = 0.0.
        let (sg, schedule) = make_graph_with_external_dep(
            &[
                ("a", 10.0, &["[dep-a/lib]::foo"]),
                ("b", 20.0, &[]),
            ],
            50.0,
        );
        let intra = condense_target(&sg, "test-pkg/lib").unwrap();
        let horizons =
            compute_effective_horizons(&intra, &sg, "test-pkg/lib", &schedule);

        // Find which SCC contains "a" and which contains "b".
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

        assert!(
            (horizons[a_scc] - 50.0).abs() < f64::EPSILON,
            "a depends on dep-a which finishes at 50"
        );
        assert!(
            horizons[b_scc].abs() < f64::EPSILON,
            "b has no external deps"
        );
    }

    #[test]
    fn horizon_propagates_through_predecessors() {
        // Chain: b depends on a. "a" depends on external [dep-a/lib]::foo.
        // a's horizon = 50.0 (external dep finish time).
        // b's horizon = max(b's own external deps, a's horizon) = 50.0.
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

        assert!(
            (horizons[a_scc] - 50.0).abs() < f64::EPSILON,
            "a directly depends on dep-a"
        );
        assert!(
            (horizons[b_scc] - 50.0).abs() < f64::EPSILON,
            "b inherits a's horizon through the predecessor edge"
        );
    }
}
```

**Step 3: Run tests to verify they fail**

Run: `cargo nextest run -p tarjanize-schedule horizon`
Expected: FAIL (todo!() panics)

**Step 4: Implement `compute_effective_horizons`**

```rust
fn compute_effective_horizons(
    intra: &IntraTargetGraph,
    symbol_graph: &SymbolGraph,
    target_id: &str,
    schedule: &ScheduleData,
) -> Vec<f64> {
    let n = intra.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Build a name → finish time lookup from the schedule.
    let finish_times: HashMap<&str, f64> = schedule
        .targets
        .iter()
        .map(|t| (t.name.as_str(), t.finish))
        .collect();

    // Build a symbol → external target finish times mapping.
    // Walk the symbol graph's module tree to find each symbol's deps,
    // filter to external deps (not starting with the target prefix),
    // and resolve them to finish times.
    let target_prefix = format!("[{target_id}]::");
    let (package_name, target_key) = target_id.split_once('/').unwrap();
    let target = &symbol_graph.packages[package_name].targets[target_key];

    // Map from symbol full path → set of external target finish times.
    let external_finishes: HashMap<String, f64> =
        collect_external_max_finish(
            &target.root,
            &target_prefix,
            "",
            &finish_times,
        );

    // For each SCC, compute the max external finish time across all its
    // member symbols.
    let mut scc_direct_horizon: Vec<f64> = vec![0.0; n];
    for (scc_id, node) in intra.nodes.iter().enumerate() {
        for sym in &node.symbols {
            if let Some(&finish) = external_finishes.get(sym) {
                scc_direct_horizon[scc_id] =
                    scc_direct_horizon[scc_id].max(finish);
            }
        }
    }

    // Build predecessor adjacency list. IntraTargetGraph edges go
    // dep → dependent, so for node `to`, its predecessors (things it
    // depends on) are the `from` nodes.
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];
    for &(from, to) in &intra.edges {
        predecessors[to].push(from);
        in_degree[to] += 1;
    }

    // Topological sort (Kahn's algorithm) for propagation order.
    let mut queue: std::collections::VecDeque<usize> =
        (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut topo_order = Vec::with_capacity(n);
    let mut remaining_in = in_degree.clone();
    while let Some(node) = queue.pop_front() {
        topo_order.push(node);
        // Find successors (nodes that depend on this node).
        for &(from, to) in &intra.edges {
            if from == node {
                remaining_in[to] -= 1;
                if remaining_in[to] == 0 {
                    queue.push_back(to);
                }
            }
        }
    }

    // Propagate effective horizons in topological order.
    let mut effective_horizon = scc_direct_horizon;
    for &node in &topo_order {
        for &pred in &predecessors[node] {
            effective_horizon[node] =
                effective_horizon[node].max(effective_horizon[pred]);
        }
    }

    effective_horizon
}

/// Walks the module tree and returns a map from symbol full path to the
/// maximum finish time across all its external dependencies.
fn collect_external_max_finish(
    module: &Module,
    target_prefix: &str,
    module_path: &str,
    finish_times: &HashMap<&str, f64>,
) -> HashMap<String, f64> {
    let mut result = HashMap::new();

    for (name, symbol) in &module.symbols {
        let full_path = if module_path.is_empty() {
            format!("{target_prefix}{name}")
        } else {
            format!("{target_prefix}{module_path}::{name}")
        };

        let mut max_finish = 0.0f64;
        for dep in &symbol.dependencies {
            // Skip intra-target deps.
            if dep.starts_with(target_prefix) {
                continue;
            }
            // Extract target name from path: "[pkg/target]::symbol" → "pkg/target"
            if let Some(target_name) = extract_target_from_path(dep) {
                if let Some(&finish) = finish_times.get(target_name) {
                    max_finish = max_finish.max(finish);
                }
            }
        }
        if max_finish > 0.0 {
            result.insert(full_path, max_finish);
        }
    }

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

/// Extracts the target identifier from a symbol path.
///
/// `[pkg/target]::module::symbol` → `pkg/target`
fn extract_target_from_path(path: &str) -> Option<&str> {
    let start = path.find('[')? + 1;
    let end = path.find(']')?;
    Some(&path[start..end])
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo nextest run -p tarjanize-schedule horizon`
Expected: PASS

**Step 6: Run clippy**

Run: `cargo clippy -p tarjanize-schedule --all-targets`
Expected: No warnings.

**Step 7: Commit**

```bash
jj describe -m "Add effective horizon computation for split recommendations"
jj new
```

---

### Task 3: Generate and evaluate split candidates

Add the candidate generation and evaluation logic to `recommend.rs`. For each distinct effective horizon threshold below the target's max horizon, compute the downset/upset partition and evaluate local + global improvement.

**Files:**
- Modify: `crates/tarjanize-schedule/src/recommend.rs`
- Modify: `crates/tarjanize-schedule/src/lib.rs` (re-export types if needed)

**Step 1: Define response types**

Add serializable types for the API response to `recommend.rs`:

```rust
/// Ranked split candidates for a single target.
#[derive(Debug, Serialize)]
pub struct SplitRecommendation {
    /// Target identifier (`{package}/{target}`).
    pub target: String,
    /// Current predicted cost of the target (ms).
    pub current_cost_ms: f64,
    /// Ranked candidates, best first (by global_improvement_ms desc).
    pub candidates: Vec<SplitCandidate>,
}

/// A single candidate split at a specific horizon threshold.
#[derive(Debug, Serialize)]
pub struct SplitCandidate {
    /// The effective horizon threshold defining this cut (ms).
    pub threshold_ms: f64,
    /// How much earlier the two halves finish vs. the original (ms).
    pub local_improvement_ms: f64,
    /// Reduction in critical path length (ms). Zero for off-critical-path
    /// targets.
    pub global_improvement_ms: f64,
    /// Number of SCCs in the extracted (early) crate.
    pub downset_scc_count: usize,
    /// Predicted cost of the extracted crate (ms).
    pub downset_cost_ms: f64,
    /// Number of SCCs remaining in the original crate.
    pub upset_scc_count: usize,
    /// Predicted cost of the remaining crate (ms).
    pub upset_cost_ms: f64,
    /// Modules with at least one SCC in the downset.
    pub downset_modules: Vec<String>,
    /// Modules with at least one SCC in the upset.
    pub upset_modules: Vec<String>,
    /// Modules that appear in both halves (require careful refactoring).
    pub split_modules: Vec<String>,
}
```

**Step 2: Write failing test — basic candidate generation**

```rust
#[test]
fn recommend_produces_candidate_for_split_horizon() {
    // Target with two SCCs: "a" depends on external dep finishing at 50ms,
    // "b" has no external deps (horizon 0). This creates one candidate
    // at threshold 0.0 that separates "b" (downset) from "a" (upset).
    let (sg, schedule) = make_graph_with_external_dep(
        &[
            ("a", 10.0, &["[dep-a/lib]::foo"]),
            ("b", 20.0, &[]),
        ],
        50.0,
    );

    let result = compute_split_recommendations(
        &sg,
        "test-pkg/lib",
        &schedule,
        None, // no cost model
    );

    assert_eq!(result.target, "test-pkg/lib");
    assert_eq!(result.candidates.len(), 1, "one threshold below max");
    let c = &result.candidates[0];
    assert!((c.threshold_ms - 0.0).abs() < f64::EPSILON);
    assert_eq!(c.downset_scc_count, 1, "b is in the downset");
    assert_eq!(c.upset_scc_count, 1, "a is in the upset");
}

#[test]
fn recommend_empty_when_all_horizons_equal() {
    // Two independent symbols, no external deps. All horizons are 0.
    // No threshold below max_horizon exists, so no candidates.
    let (sg, schedule) = make_graph_with_external_dep(
        &[
            ("a", 10.0, &[]),
            ("b", 20.0, &[]),
        ],
        50.0,
    );

    let result = compute_split_recommendations(
        &sg,
        "test-pkg/lib",
        &schedule,
        None,
    );

    assert!(
        result.candidates.is_empty(),
        "no split possible when all horizons are equal"
    );
}

#[test]
fn recommend_local_improvement_is_positive() {
    // Split should save time locally: the downset can start earlier.
    let (sg, schedule) = make_graph_with_external_dep(
        &[
            ("a", 40.0, &["[dep-a/lib]::foo"]),
            ("b", 30.0, &[]),
        ],
        50.0,
    );

    let result = compute_split_recommendations(
        &sg,
        "test-pkg/lib",
        &schedule,
        None,
    );

    assert_eq!(result.candidates.len(), 1);
    assert!(
        result.candidates[0].local_improvement_ms > 0.0,
        "splitting should improve local finish time"
    );
}

#[test]
fn recommend_negative_improvement_filtered_out() {
    // When overhead from splitting exceeds the parallelism gain,
    // the candidate has negative improvement and is filtered out.
    // With a cost model that has high per-target overhead coefficients,
    // a tiny split produces negative improvement.
    use tarjanize_schemas::cost_model::CostModel;

    let (sg, schedule) = make_graph_with_external_dep(
        &[
            ("a", 1.0, &["[dep-a/lib]::foo"]),
            ("b", 1.0, &[]),
        ],
        50.0,
    );

    // Cost model with very high overhead (meta + other coefficients).
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
        "high-overhead split should be filtered out"
    );
}
```

**Step 3: Run tests to verify they fail**

Run: `cargo nextest run -p tarjanize-schedule recommend`
Expected: FAIL (function not defined)

**Step 4: Implement `compute_split_recommendations`**

```rust
use tarjanize_schemas::cost_model::CostModel;
use tarjanize_schemas::symbol_graph::Module;

/// Computes ranked split recommendations for a target.
///
/// Returns `SplitRecommendation` with candidates sorted by
/// `global_improvement_ms` descending (best first). Candidates with
/// negative local improvement (where split overhead exceeds parallelism
/// gain) are filtered out.
pub fn compute_split_recommendations(
    symbol_graph: &SymbolGraph,
    target_id: &str,
    schedule: &ScheduleData,
    cost_model: Option<&CostModel>,
) -> SplitRecommendation {
    // Look up the target's current cost from the schedule.
    let current_cost_ms = schedule
        .targets
        .iter()
        .find(|t| t.name == target_id)
        .map_or(0.0, |t| t.cost);

    let empty = SplitRecommendation {
        target: target_id.to_string(),
        current_cost_ms,
        candidates: Vec::new(),
    };

    // Condense the target into its SCC DAG.
    let Some(intra) = condense_target(symbol_graph, target_id) else {
        return empty;
    };

    if intra.nodes.is_empty() {
        return empty;
    }

    // Compute effective horizons.
    let horizons =
        compute_effective_horizons(&intra, symbol_graph, target_id, schedule);

    // Find max_horizon (the target's current start time equivalent).
    let max_horizon = horizons
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0);

    // Find distinct thresholds below max_horizon.
    let mut thresholds: Vec<f64> = horizons
        .iter()
        .copied()
        .filter(|&h| h < max_horizon - f64::EPSILON)
        .collect::<std::collections::BTreeSet<u64>>()
        // Use bit representation for dedup since f64 doesn't impl Ord.
        // Alternative: collect into Vec, sort, dedup.
        ;

    // Actually, let's use a simpler approach for dedup:
    let mut threshold_set: Vec<f64> = horizons
        .iter()
        .copied()
        .filter(|&h| h < max_horizon - f64::EPSILON)
        .collect();
    threshold_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
    threshold_set.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

    if threshold_set.is_empty() {
        return empty;
    }

    // Look up the target's meta and other costs for cost model predictions.
    let (meta, other) = if cost_model.is_some() {
        let (pkg, tgt) = target_id.split_once('/').unwrap();
        let target = &symbol_graph.packages[pkg].targets[tgt];
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
    } else {
        (0.0, 0.0)
    };

    // The target's current start time in the schedule.
    let target_start = schedule
        .targets
        .iter()
        .find(|t| t.name == target_id)
        .map_or(0.0, |t| t.start);

    let original_finish = target_start + current_cost_ms;

    // Generate candidates for each threshold.
    let mut candidates = Vec::new();
    for &tau in &threshold_set {
        // Downset: SCCs with effective_horizon <= tau.
        // Upset: remaining SCCs.
        let downset: Vec<usize> = (0..intra.nodes.len())
            .filter(|&i| horizons[i] <= tau + f64::EPSILON)
            .collect();
        let upset: Vec<usize> = (0..intra.nodes.len())
            .filter(|&i| horizons[i] > tau + f64::EPSILON)
            .collect();

        if downset.is_empty() || upset.is_empty() {
            continue;
        }

        // Compute costs for each half.
        let downset_attr: f64 =
            downset.iter().map(|&i| intra.nodes[i].cost_ms).sum();
        let upset_attr: f64 =
            upset.iter().map(|&i| intra.nodes[i].cost_ms).sum();

        let (downset_cost, upset_cost) = if let Some(model) = cost_model {
            // Each half inherits the full target's meta/other overhead.
            (
                model.predict(downset_attr, meta, other),
                model.predict(upset_attr, meta, other),
            )
        } else {
            // No cost model: use raw attr sums.
            (downset_attr, upset_attr)
        };

        // Evaluate local improvement.
        // Downset starts at tau, finishes at tau + downset_cost.
        // Upset starts at max(tau + downset_cost, max_horizon), finishes
        // at start(U) + upset_cost.
        let downset_finish = tau + downset_cost;
        let upset_start = (tau + downset_cost).max(max_horizon);
        let upset_finish = upset_start + upset_cost;
        let split_finish = downset_finish.max(upset_finish);
        let local_improvement = original_finish - split_finish;

        // Filter out detrimental splits.
        if local_improvement <= 0.0 {
            continue;
        }

        // Compute global improvement by evaluating the modified schedule.
        let global_improvement = compute_global_improvement(
            schedule,
            target_id,
            downset_cost,
            upset_cost,
            tau,
            max_horizon,
        );

        // Compute module breakdowns.
        let downset_modules = collect_modules(&intra, &downset);
        let upset_modules = collect_modules(&intra, &upset);
        let split_modules: Vec<String> = downset_modules
            .iter()
            .filter(|m| upset_modules.contains(m))
            .cloned()
            .collect();

        candidates.push(SplitCandidate {
            threshold_ms: tau,
            local_improvement_ms: local_improvement,
            global_improvement_ms: global_improvement,
            downset_scc_count: downset.len(),
            downset_cost_ms: downset_cost,
            upset_scc_count: upset.len(),
            upset_cost_ms: upset_cost,
            downset_modules,
            upset_modules,
            split_modules,
        });
    }

    // Sort by global improvement descending.
    candidates.sort_by(|a, b| {
        b.global_improvement_ms
            .partial_cmp(&a.global_improvement_ms)
            .unwrap()
    });

    SplitRecommendation {
        target: target_id.to_string(),
        current_cost_ms,
        candidates,
    }
}

/// Collects distinct module paths for a set of SCC indices.
fn collect_modules(
    intra: &IntraTargetGraph,
    scc_indices: &[usize],
) -> Vec<String> {
    let mut mods: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    for &i in scc_indices {
        mods.insert(intra.nodes[i].module_path.clone());
    }
    mods.into_iter().collect()
}

/// Computes global improvement by rebuilding the schedule with the target
/// replaced by two halves.
///
/// Returns the reduction in critical path length (ms). Negative values
/// are clamped to 0.
fn compute_global_improvement(
    schedule: &ScheduleData,
    target_id: &str,
    downset_cost: f64,
    upset_cost: f64,
    threshold: f64,
    max_horizon: f64,
) -> f64 {
    use crate::schedule::{compute_schedule, TargetGraph};
    use indexmap::IndexSet;
    use petgraph::graph::{DiGraph, NodeIndex};

    // Build a modified TargetGraph with the target replaced by two halves.
    let target_idx = schedule
        .targets
        .iter()
        .position(|t| t.name == target_id);
    let Some(target_idx) = target_idx else {
        return 0.0;
    };

    let n = schedule.targets.len();
    let mut names = IndexSet::new();
    let mut costs = Vec::new();
    let mut symbol_counts = Vec::new();
    let mut graph = DiGraph::<usize, ()>::new();

    // Map from original target index to new index.
    let mut old_to_new: Vec<Option<usize>> = vec![None; n];
    let downset_name = format!("{target_id}::downset");
    let upset_name = target_id.to_string();

    // Add all targets except the split one. Add the two halves instead.
    for (i, t) in schedule.targets.iter().enumerate() {
        if i == target_idx {
            // Add downset half.
            let di = names.len();
            names.insert(downset_name.clone());
            costs.push(downset_cost);
            symbol_counts.push(0);
            graph.add_node(di);

            // Add upset half (keeps the original name).
            let ui = names.len();
            names.insert(upset_name.clone());
            costs.push(upset_cost);
            symbol_counts.push(t.symbol_count);
            graph.add_node(ui);

            old_to_new[i] = Some(ui); // upset keeps the identity
        } else {
            let ni = names.len();
            old_to_new[i] = Some(ni);
            names.insert(t.name.clone());
            costs.push(t.cost);
            symbol_counts.push(t.symbol_count);
            graph.add_node(ni);
        }
    }

    let downset_idx = names.get_index_of(&downset_name).unwrap();
    let upset_idx = names.get_index_of(&upset_name).unwrap();

    // Wire edges.
    for (i, t) in schedule.targets.iter().enumerate() {
        if i == target_idx {
            continue;
        }
        let ni = old_to_new[i].unwrap();

        for &dep in &t.deps {
            if dep == target_idx {
                // This target depended on the split target.
                // Now depends on both halves.
                graph.add_edge(
                    NodeIndex::new(downset_idx),
                    NodeIndex::new(ni),
                    (),
                );
                graph.add_edge(
                    NodeIndex::new(upset_idx),
                    NodeIndex::new(ni),
                    (),
                );
            } else if let Some(di) = old_to_new[dep] {
                graph.add_edge(NodeIndex::new(di), NodeIndex::new(ni), ());
            }
        }
    }

    // Wire the split target's original deps to both halves.
    let orig = &schedule.targets[target_idx];
    for &dep in &orig.deps {
        if let Some(di) = old_to_new[dep] {
            graph.add_edge(NodeIndex::new(di), NodeIndex::new(downset_idx), ());
            graph.add_edge(NodeIndex::new(di), NodeIndex::new(upset_idx), ());
        }
    }

    // Add edge: downset → upset (upset depends on downset completing,
    // because upset SCCs may use downset SCCs).
    graph.add_edge(
        NodeIndex::new(downset_idx),
        NodeIndex::new(upset_idx),
        (),
    );

    let tg = TargetGraph {
        names,
        costs,
        symbol_counts,
        graph,
    };

    let new_schedule = compute_schedule(&tg);
    let improvement =
        schedule.summary.critical_path_ms - new_schedule.summary.critical_path_ms;
    improvement.max(0.0)
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo nextest run -p tarjanize-schedule recommend`
Expected: PASS

**Step 6: Run full suite + clippy**

Run: `cargo nextest run -p tarjanize-schedule && cargo clippy -p tarjanize-schedule --all-targets`
Expected: All pass, no warnings.

**Step 7: Commit**

```bash
jj describe -m "Add split candidate generation with local and global improvement evaluation"
jj new
```

---

### Task 4: API endpoint `GET /api/splits/{package}/{target}`

Add the API endpoint to the viz server that returns split recommendations for a target.

**Files:**
- Modify: `crates/tarjanize-viz/src/server.rs`

**Step 1: Write failing test**

Add a test in `server.rs`'s test module:

```rust
#[tokio::test]
async fn api_splits_returns_recommendations() {
    let state = test_state();
    let app = build_router(state.clone());

    // Pick a target from the schedule to query.
    let schedule = state.schedule.read().await;
    let target_name = &schedule.targets[0].name;
    let url = format!("/api/splits/{target_name}");
    drop(schedule);

    let response = app
        .oneshot(
            http::Request::builder()
                .uri(&url)
                .body(http_body_util::Empty::<bytes::Bytes>::new())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), http::StatusCode::OK);

    let body = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    let rec: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(rec.get("target").is_some());
    assert!(rec.get("candidates").is_some());
    assert!(rec.get("current_cost_ms").is_some());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run -p tarjanize-viz api_splits`
Expected: FAIL (404 — route doesn't exist)

**Step 3: Implement the endpoint**

Add to `server.rs`:

```rust
use tarjanize_schedule::recommend::compute_split_recommendations;

// In build_router(), add the route:
.route(
    "/api/splits/{package}/{target}",
    axum::routing::get(splits_handler),
)

// Handler:
/// Returns ranked split recommendations for a target.
async fn splits_handler(
    State(state): State<Arc<AppState>>,
    Path((package, target)): Path<(String, String)>,
) -> impl IntoResponse {
    let target_id = format!("{package}/{target}");
    let schedule = state.schedule.read().await;
    let rec = compute_split_recommendations(
        &state.symbol_graph,
        &target_id,
        &schedule,
        state.cost_model.as_ref(),
    );
    Json(rec)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo nextest run -p tarjanize-viz api_splits`
Expected: PASS

**Step 5: Run full suite + clippy**

Run: `cargo nextest run -p tarjanize-viz && cargo clippy -p tarjanize-viz --all-targets`
Expected: All pass, no warnings.

**Step 6: Commit**

```bash
jj describe -m "Add GET /api/splits/{package}/{target} endpoint"
jj new
```

---

### Task 5: TypeScript migration

Rename frontend `.js` files to `.ts`, add `tsconfig.json`, and update the build pipeline. esbuild handles TypeScript natively so no additional tooling is needed.

**Files:**
- Create: `tsconfig.json` (project root)
- Rename: `crates/tarjanize-viz/templates/logic.js` → `logic.ts`
- Rename: `crates/tarjanize-viz/templates/renderer.js` → `renderer.ts`
- Rename: `crates/tarjanize-viz/templates/dag.js` → `dag.ts`
- Modify: `crates/tarjanize-viz/build.rs` (update entry points)

**Step 1: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "baseUrl": ".",
    "paths": {
      "pixi.js": ["node_modules/pixi.js"]
    }
  },
  "include": ["crates/tarjanize-viz/templates/**/*.ts"]
}
```

**Step 2: Rename files**

```bash
mv crates/tarjanize-viz/templates/logic.js crates/tarjanize-viz/templates/logic.ts
mv crates/tarjanize-viz/templates/renderer.js crates/tarjanize-viz/templates/renderer.ts
mv crates/tarjanize-viz/templates/dag.js crates/tarjanize-viz/templates/dag.ts
```

**Step 3: Update build.rs entry points and rerun triggers**

Change `templates/renderer.js` → `templates/renderer.ts`, `templates/dag.js` → `templates/dag.ts`, `templates/logic.js` → `templates/logic.ts` in `build.rs`. Both the entry point paths and the `cargo:rerun-if-changed` lines need updating.

**Step 4: Update imports in renderer.ts**

Change `import { ... } from './logic.js'` to `import { ... } from './logic.ts'` (or remove the extension — esbuild resolves both).

**Step 5: Verify build**

Run: `cargo build -p tarjanize-viz`
Expected: Compiles successfully (esbuild handles `.ts` natively).

**Step 6: Run full test suite**

Run: `cargo nextest run -p tarjanize-viz`
Expected: All pass.

**Step 7: Commit**

```bash
jj describe -m "Migrate frontend from JavaScript to TypeScript"
jj new
```

---

### Task 6: Frontend — sidebar split recommendations

Replace the left-panel DAG + manual split UI with a sidebar recommendations panel. When the user clicks a Gantt bar, the sidebar fetches and displays ranked split candidates. Clicking a candidate previews the modified schedule.

**Files:**
- Modify: `crates/tarjanize-viz/templates/app.html`
- Modify: `crates/tarjanize-viz/templates/renderer.ts`
- Modify: `crates/tarjanize-viz/templates/style.css`
- Remove: `crates/tarjanize-viz/templates/dag.ts` (or empty it — handled in Task 7)

**Step 1: Update app.html — replace left panel with recommendations sidebar**

Replace the left panel (DAG header, SVG canvas, split controls) with a recommendations panel inside the sidebar. The sidebar should have two states:

1. **No selection**: shows overall schedule stats (current behavior).
2. **Target selected**: shows target name, cost, and ranked candidates.

Add an inline script that:
- Listens for `target-click` events from the Gantt chart.
- Fetches `GET /api/splits/{package}/{target}` for the clicked target.
- Renders the candidates list in the sidebar.
- Clicking a candidate fetches a preview schedule and re-renders the Gantt.

Specific UI elements per candidate:
```html
<div class="split-candidate" data-index="0">
  <div class="improvement">saves 203ms critical path, 1,400ms local</div>
  <div class="downset-info">43 SCCs (1,100ms) can start at T=850ms</div>
  <div class="split-warning">splits db::schema across both crates</div>
</div>
```

**Step 2: Update renderer.ts — add click handler for split preview**

Add a function that accepts a new `ScheduleData` and re-renders the Gantt chart with it. This is used when the user clicks a candidate to preview the split.

```typescript
export function updateSchedule(newData: ScheduleData): void {
    // Rebuild bars, labels, edges from newData.
    // Reuse the existing PixiJS app — just clear and rebuild containers.
}
```

**Step 3: Add CSS for recommendations panel**

Style the candidate cards, improvement numbers, and split warnings. Use the existing dark theme palette.

**Step 4: Add preview API endpoint**

Add `POST /api/preview-split` to `server.rs` that accepts `{ target_id: string, threshold_ms: number }`, runs the recommendation algorithm to find the matching candidate, applies the split via `apply_splits`, and returns the modified `ScheduleData`.

```rust
#[derive(Deserialize)]
struct PreviewSplitRequest {
    target_id: String,
    threshold_ms: f64,
}

async fn preview_split_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PreviewSplitRequest>,
) -> impl IntoResponse {
    // 1. Get recommendations to find the downset SCCs for this threshold.
    // 2. Apply the split using apply_splits.
    // 3. Compute new schedule.
    // 4. Return ScheduleData.
}
```

**Step 5: Verify in browser**

Run: `cargo run -p tarjanize -- viz <test-data-path>`
Expected: Server starts. Click a target → sidebar shows candidates. Click a candidate → Gantt re-renders with the split applied. Click away → Gantt returns to original.

**Step 6: Commit**

```bash
jj describe -m "Add sidebar split recommendations UI with Gantt preview"
jj new
```

---

### Task 7: Remove old DAG/split UI and clean up

Remove the d3-force DAG visualization, manual split workflow, and related dead code.

**Files:**
- Delete: `crates/tarjanize-viz/templates/dag.ts`
- Modify: `crates/tarjanize-viz/build.rs` (remove dag bundle)
- Modify: `crates/tarjanize-viz/src/server.rs` (remove dag/split/unsplit/heatmap endpoints, remove `DAG_BUNDLE_JS`)
- Modify: `crates/tarjanize-viz/templates/app.html` (remove dag script tag, split controls references)
- Modify: `crates/tarjanize-viz/templates/style.css` (remove DAG/split styles)

**Step 1: Remove dag.ts and dag bundle from build.rs**

Delete `crates/tarjanize-viz/templates/dag.ts`. In `build.rs`, remove the second esbuild invocation that produces `dag_bundle.js`. Remove the `cargo:rerun-if-changed=templates/dag.ts` line.

**Step 2: Remove DAG_BUNDLE_JS from server.rs**

Remove the `DAG_BUNDLE_JS` constant and the `dag_bundle_js` field from `AppTemplate`. Remove the related routes:
- `GET /api/target/{package}/{target}/graph` (target_graph_handler)
- `GET /api/target/{package}/{target}/heatmap` (heatmap_handler)
- `POST /api/split` (split_handler)
- `POST /api/unsplit` (unsplit_handler)

Keep `GET /api/export` (still useful for saving the modified SymbolGraph).

**Step 3: Remove split state from AppState**

Remove `splits: RwLock<Vec<SplitOperation>>` from `AppState` if no longer needed (the preview endpoint would use ephemeral state, not persistent splits).

**Step 4: Clean up CSS**

Remove styles for `.dag-header`, `.split-controls`, `#dag-tooltip`, `.left-panel`, and related selectors. Update the layout to single-panel (sidebar + Gantt only).

**Step 5: Remove obsolete tests**

Remove tests for removed endpoints: `api_target_graph_returns_scc_dag`, `api_target_graph_404_for_missing`, `api_heatmap_returns_slack_data`, `api_split_returns_result_and_schedule`, `api_split_then_unsplit_restores_schedule`.

**Step 6: Verify**

Run: `cargo nextest run -p tarjanize-viz && cargo clippy -p tarjanize-viz --all-targets`
Expected: All pass, no warnings.

Run: `cargo build -p tarjanize-viz`
Expected: Compiles. No dead code warnings.

**Step 7: Commit**

```bash
jj describe -m "Remove old DAG/split UI, clean up dead code and endpoints"
jj new
```

---

## Execution Notes

- Tasks 1–4 are sequential (each builds on the previous).
- Task 5 (TS migration) is independent of Tasks 1–4 and can be done in parallel.
- Tasks 6–7 depend on both Task 4 (API) and Task 5 (TS migration).
- The `improvement_ms` field in `SccHeat` (heatmap.rs) and its test `improvement_ms_populated_for_critical_path_sccs` are superseded by this design. That test should either be deleted or updated to match the new approach. Handle this during Task 7 cleanup.
- The `compute_global_improvement` function builds a modified `TargetGraph` by hand rather than using `apply_splits`, because `apply_splits` requires `SplitOperation` with specific SCC IDs and a `new_crate_name` — machinery we don't need for evaluation-only purposes.
