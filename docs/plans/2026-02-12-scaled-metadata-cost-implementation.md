# Scaled Metadata Cost Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Improve cost prediction accuracy for shattered crates by scaling the metadata cost component based on the number of dependencies each group actually uses.

**Architecture:** Modify `shatter_target` to compute the number of external and internal dependencies per group, calculate the ratio relative to the original crate's dependencies, and scale the `meta` input to the cost model accordingly.

**Tech Stack:** Rust, petgraph

---

### Task 1: Add Regression Test for Dependency Scaling

**Files:**
- Create: `crates/tarjanize-schedule/tests/shatter_scaling.rs`

**Step 1: Create failing test `shatter_scales_metadata_cost`**

This test sets up a scenario where a shattered group uses only a subset of dependencies, and asserts that its cost is LOWER than the full metadata cost. Use `superpowers:test-driven-development`.

```rust
#[test]
fn shatter_scales_metadata_cost() {
    // Setup:
    // Original target depends on Dep1, Dep2.
    // It has two symbols:
    // - SymA depends on Dep1
    // - SymB depends on nothing (but is in the same crate)
    //
    // Shatter into GroupA (SymA) and GroupB (SymB).
    //
    // Original Deps = 2 (Dep1, Dep2).
    // GroupA Deps = 1 (Dep1) + 0 (internal) = 1. Ratio = 0.5.
    // GroupB Deps = 0 (external) + 0 (internal) = 0. Ratio = 0.0.
    //
    // Cost Model: meta=100.
    // GroupA meta cost should be 50.
    // GroupB meta cost should be 0.

    // ... implementation details ...
    // Verify assert_eq!(group_a.cost, expected_scaled_cost);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --test shatter_scaling shatter_scales_metadata_cost`
Expected: FAIL (currently it predicts full cost 100 for both).

**Step 3: Commit**

```bash
git add crates/tarjanize-schedule/tests/shatter_scaling.rs
git commit -m "test: add regression test for metadata cost scaling"
```

---

### Task 2: Implement Dependency Counting Logic

**Files:**
- Modify: `crates/tarjanize-schedule/src/recommend.rs`

**Step 1: Implement `count_internal_dependencies`**

Add a helper function to count unique group-to-group dependencies from the `intra` SCC DAG.

```rust
fn count_internal_dependencies(
    intra: &IntraTargetGraph,
    scc_to_group: &[usize],
    group_count: usize,
) -> Vec<usize> {
    // ...
}
```

**Step 2: Update `shatter_target` to use dependency counts**

1.  Get `original_deps_count` from `schedule.targets[target_idx].deps.len()`.
2.  Get `external_deps_count` from `group_ext[g].len()`.
3.  Get `internal_deps_count` from new helper.
4.  Calculate `total_deps = external + internal`.
5.  Calculate `ratio`. Handle division by zero.
6.  Scale `meta` before passing to `model.predict()`.

**Step 3: Run tests to verify pass**

Run: `cargo nextest run --test shatter_scaling`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/tarjanize-schedule/src/recommend.rs
git commit -m "feat: scale metadata cost by dependency count in shatter_target"
```

---

### Task 3: Verify Zero-Dependency Behavior

**Files:**
- Modify: `crates/tarjanize-schedule/tests/shatter_scaling.rs`

**Step 1: Add/Update test for zero dependencies**

Ensure the test case covers:
1.  Original target has 0 deps -> Ratio 0.
2.  Shattered group has 0 deps -> Ratio 0.

**Step 2: Run tests**

Run: `cargo nextest run --test shatter_scaling`
Expected: PASS

**Step 3: Verification**

Use `superpowers:verification-before-completion` to run the full test suite.

**Step 4: Commit**

```bash
git add crates/tarjanize-schedule/tests/
git commit -m "test: verify zero-dependency cost scaling behavior"
```
