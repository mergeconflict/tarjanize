# Structural Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the structural smells identified by the codebase audit — primitive obsession, data clumps, manual traversal duplication, boolean parameters, mixed-concern modules, and long functions.

**Architecture:** Six phases, each independently committable. Phase 1 introduces foundation types (`TargetId`, `CrateIdentity`) that later phases use. Phase 2 adds a module visitor to reduce traversal duplication. Phase 3 replaces boolean params with enums. Phases 4-5 split mixed-concern modules and refactor long functions. Each phase must leave all 449 tests green and clippy clean.

**Tech Stack:** Rust, cargo clippy, cargo nextest, cargo fmt

**Inviolable constraint:** No test may be deleted or have its behavioral contract weakened. Tests may be updated to use new types (e.g., `TargetId` instead of `&str`) but must assert the same conditions.

---

## Phase 1: Foundation Types

### Task 1: `TargetId` newtype in `tarjanize-schemas`

The `"pkg/target"` string format is parsed with `split_once('/')` in 5 call sites and constructed with `format!("{pkg}/{tgt}")` in 8+ call sites. A newtype centralizes this.

**Files:**
- Modify: `crates/tarjanize-schemas/src/symbol_graph.rs`
- Modify: `crates/tarjanize-schemas/src/lib.rs` (re-export)

**Design:**

```rust
/// Identifies a compilation target within a workspace as `"package/target"`.
///
/// The package name uses Cargo's hyphenated form (e.g., `my-pkg`), and the
/// target key is one of `lib`, `test`, `bench`, or a binary name.
///
/// Why: centralizes the `split_once('/')` parsing that was repeated in 5
/// call sites and the `format!("{pkg}/{tgt}")` construction in 8+ sites.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TargetId(String);

impl TargetId {
    /// Creates a `TargetId` from package name and target key.
    pub fn new(package: &str, target: &str) -> Self {
        Self(format!("{package}/{target}"))
    }

    /// Parses a `"package/target"` string, returning `None` if no `/` is present.
    pub fn parse(s: &str) -> Option<Self> {
        s.contains('/').then(|| Self(s.to_owned()))
    }

    /// Returns the package name (before the `/`).
    pub fn package(&self) -> &str {
        self.0.split_once('/').expect("TargetId always contains '/'").0
    }

    /// Returns the target key (after the `/`).
    pub fn target(&self) -> &str {
        self.0.split_once('/').expect("TargetId always contains '/'").1
    }

    /// Returns the full `"package/target"` string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TargetId { /* delegate to self.0 */ }
impl AsRef<str> for TargetId { /* delegate to self.0 */ }
impl std::borrow::Borrow<str> for TargetId { /* delegate to self.0 — enables HashMap<TargetId, _>.get("pkg/lib") */ }
```

**Step 1:** Write tests for `TargetId` — construction, parsing, accessors, Display, Borrow.

**Step 2:** Implement `TargetId` in `symbol_graph.rs`, re-export from `lib.rs`.

**Step 3:** Run `cargo nextest run -p tarjanize-schemas` — all pass.

**Step 4:** Commit.

### Task 2: Adopt `TargetId` at call sites

Replace `split_once('/')` parsing and `format!("{pkg}/{tgt}")` construction across 4 crates.

**Files to modify (parsing sites — replace `split_once('/')`):**
- `crates/tarjanize-schedule/src/target_graph.rs:80`
- `crates/tarjanize-schedule/src/recommend.rs:58-60`
- `crates/tarjanize-schedule/src/split.rs:802-805, 1060-1062`
- `crates/tarjanize-schedule/src/export.rs:74-77`

**Files to modify (construction sites — replace `format!`):**
- `crates/tarjanize-schedule/src/lib.rs:86, 138-139, 174`
- `crates/tarjanize-condense/src/scc.rs:151-152`

**Strategy:** Most code currently uses `String` for target IDs. We don't need to change every `String` to `TargetId` — focus on the sites that parse or construct the format. Internal variables can stay as `String` / `&str` where `TargetId::as_str()` bridges the gap. The `Borrow<str>` impl lets `HashMap<TargetId, _>` be looked up with `&str` keys.

For functions whose signature currently takes `target_id: &str` and immediately calls `split_once('/')`, change the parameter to `target_id: &TargetId` (or accept `TargetId` by value where ownership is needed).

For `IndexSet<String>` / `HashMap<String, _>` that store target IDs as keys: evaluate case by case. If the set/map is only used internally, it can stay as `String` with `TargetId::as_str()` at insertion. If it crosses API boundaries, consider `IndexSet<TargetId>`.

**Step 1:** Update each file, one crate at a time. After each crate, run `cargo check` to catch type errors early.

**Step 2:** Run `cargo clippy --all-targets && cargo nextest run` — all 449 tests pass.

**Step 3:** Commit.

### Task 3: `CrateIdentity` struct in `driver.rs`

Eight functions in `driver.rs` take `(crate_name, package_name, target_key)` as three separate `&str` parameters. Bundle them.

**Files:**
- Modify: `crates/cargo-tarjanize/src/driver.rs`

**Design:**

```rust
/// Identifies a crate being processed by the driver.
///
/// Bundles the three names that are always passed together through the
/// profile-processing pipeline: the rustc crate name (underscores),
/// the Cargo package name (hyphens), and the target key (lib/test/etc.).
///
/// Why: these three values form a data clump passed to 8 functions.
struct CrateIdentity<'a> {
    /// The rustc crate name (e.g., `my_package`). Uses underscores.
    crate_name: &'a str,
    /// The Cargo package name (e.g., `my-package`). Uses hyphens.
    package_name: &'a str,
    /// The target key within the package (e.g., `lib`, `test`).
    target_key: &'a str,
}
```

Note: `CrateIdentity` is private to `driver.rs` — it doesn't need to go in schemas. The `target_key` + `package_name` overlap with `TargetId`, but `CrateIdentity` additionally carries `crate_name` (the rustc name), which `TargetId` doesn't model.

**Functions to update** (change 3 params → `&CrateIdentity`):
- `process_and_write_crate` (line 943)
- `collect_profile_paths` (line 1070)
- `load_profile_data_with_span` (line 1104)
- `roll_up_unmatched_with_span` (line 1124)
- `apply_event_times_with_span` (line 1151)
- `append_unmatched_with_span` (line 1172)
- `append_unmatched_paths` (line 1240)
- `log_rustc_invocation` (line 422)

**Step 1:** Add `CrateIdentity` struct. Update function signatures and bodies. Construct `CrateIdentity` at the call site in `TarjanizeCallbacks::after_analysis` where all three values are known.

**Step 2:** Run `cargo clippy --all-targets -p cargo-tarjanize && cargo nextest run -p cargo-tarjanize` — all pass.

**Step 3:** Commit.

---

## Phase 2: Module Visitor

### Task 4: `Module::for_each_symbol` traversal method

15+ manual traversals walk `.symbols` then `.submodules` recursively. A shared method reduces duplication and makes the traversal pattern a single source of truth.

**Files:**
- Modify: `crates/tarjanize-schemas/src/symbol_graph.rs` (add methods to `impl Module`)

**Design:**

```rust
impl Module {
    /// Calls `f(module_path, symbol_name, symbol)` for every symbol in
    /// this module tree, recursing into submodules.
    ///
    /// The `path` parameter is the module path prefix for this module
    /// (empty string for the root).
    ///
    /// Why: 15+ call sites manually walk `.symbols` + `.submodules`.
    /// Centralizing the traversal eliminates the duplication and ensures
    /// consistent ordering.
    pub fn for_each_symbol(
        &self,
        path: &str,
        f: &mut impl FnMut(&str, &str, &Symbol),
    ) {
        for (name, symbol) in &self.symbols {
            f(path, name, symbol);
        }
        for (submod_name, submod) in &self.submodules {
            let sub_path = if path.is_empty() {
                submod_name.clone()
            } else {
                format!("{path}::{submod_name}")
            };
            submod.for_each_symbol(&sub_path, f);
        }
    }

    /// Calls `f(module_path, module)` for this module and every
    /// descendant module.
    ///
    /// Why: some traversals need module-level processing (e.g.,
    /// collecting module paths), not just symbol-level.
    pub fn for_each_module(
        &self,
        path: &str,
        f: &mut impl FnMut(&str, &Module),
    ) {
        f(path, self);
        for (submod_name, submod) in &self.submodules {
            let sub_path = if path.is_empty() {
                submod_name.clone()
            } else {
                format!("{path}::{submod_name}")
            };
            submod.for_each_module(&sub_path, f);
        }
    }
}
```

**Step 1:** Write tests for both methods — verify they visit all symbols/modules with correct paths.

**Step 2:** Implement both methods.

**Step 3:** Run `cargo nextest run -p tarjanize-schemas` — pass.

**Step 4:** Adopt at call sites where it simplifies code. Not every manual traversal needs to switch — only those where the closure-based API is a net readability win. Skip traversals that need mutable access to the module tree or that accumulate results with complex control flow (early returns, etc.).

Good candidates to convert (accumulate into a collection via closure):
- `collect_symbol_paths` / `collect_module_paths` in `driver.rs` (lines 1288-1324)
- `collect_symbols` in `target_graph.rs` (lines 342-375)
- `populate_costs` in `server.rs` (lines 248-293)
- `count_impls_by_group` in `recommend.rs` (lines 320-336)

Skip (need mutable module access or complex return types):
- `collect_impl_anchors` in `target_graph.rs` — returns structured anchor data
- `collect_dependencies` in `target_graph.rs` — returns dependency pairs
- `wire_dependent_to_groups` in `recommend.rs` — mutates graph

**Step 5:** Run full `cargo clippy --all-targets && cargo nextest run` — all 449 pass.

**Step 6:** Commit.

---

## Phase 3: Boolean Parameter Enums

### Task 5: `IsIntegrationTest` enum in `driver.rs`

Two functions take `is_integration_test: bool` where call sites pass bare `false` literals (especially in tests at lines 1334, 1344, 1352, 1360).

**Files:**
- Modify: `crates/cargo-tarjanize/src/driver.rs`

**Design:**

```rust
/// Whether a target is an integration test (lives in `tests/`).
///
/// Why: bare `bool` parameters at call sites like
/// `determine_target_key(&args, "name", false)` don't communicate intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IsIntegrationTest {
    Yes,
    No,
}
```

**Functions to update:**
- `log_rustc_invocation` (line 422) — `is_test: bool, is_integration_test: bool` → keep `is_test` as bool (it's always passed as a named variable), convert `is_integration_test`
- `determine_target_key` (line 756) — `is_integration_test: bool` → `IsIntegrationTest`
- `resolve_workspace_package` return type (line 325) — currently returns `Option<(String, bool)>`, change to `Option<(String, IsIntegrationTest)>`

**Step 1:** Add enum, update signatures, update all call sites. Test call sites at lines 1334, 1344, 1352, 1360 become `IsIntegrationTest::No`.

**Step 2:** Run `cargo clippy --all-targets -p cargo-tarjanize && cargo nextest run -p cargo-tarjanize`.

**Step 3:** Commit.

---

## Phase 4: Module Splits

Each split is a pure file reorganization — move functions to a new file, add `mod` declaration, re-export public items. No logic changes. Existing tests must pass unchanged.

### Task 6: Split `scc.rs` → extract `rewrite.rs`

Move path-mapping and module-tree-building functions out of `scc.rs` into a new `rewrite.rs` module.

**Files:**
- Create: `crates/tarjanize-condense/src/rewrite.rs`
- Modify: `crates/tarjanize-condense/src/scc.rs` (remove moved functions, add `use crate::rewrite::*`)
- Modify: `crates/tarjanize-condense/src/lib.rs` (add `mod rewrite;`)

**Functions to move** (~250 lines):
- `compute_path_mapping` (lines 842-933)
- `SymbolPathKey` and `SymbolOccurrences` type aliases (lines 943-951)
- `build_module_tree` (lines 957-1034)
- `rewrite_symbol` (lines 1039-1069)
- `get_or_create_module` (lines 1074-1090)

**What remains in `scc.rs`:** SCC computation (`condense_and_partition`), graph building (`build_output_graph`), symbol indexing, anchor parsing, all tests.

**Step 1:** Create `rewrite.rs` with `//!` module doc, move functions, add necessary `use` imports.

**Step 2:** In `scc.rs`, replace moved functions with `use crate::rewrite::{compute_path_mapping, build_module_tree, rewrite_symbol};`.

**Step 3:** Run `cargo nextest run -p tarjanize-condense` — all pass.

**Step 4:** Commit.

### Task 7: Split `recommend.rs` → extract `horizon.rs` and `grouping.rs`

**Files:**
- Create: `crates/tarjanize-schedule/src/horizon.rs`
- Create: `crates/tarjanize-schedule/src/grouping.rs`
- Modify: `crates/tarjanize-schedule/src/recommend.rs`
- Modify: `crates/tarjanize-schedule/src/lib.rs` (add `mod horizon; mod grouping;`)

**Move to `horizon.rs`** (~188 lines):
- `compute_effective_horizons` (lines 36-133)
- `extract_target_from_path` (lines 149-154)
- `collect_external_max_finish` (lines 168-223)

**Move to `grouping.rs`** (~415 lines):
- `GroupRatios` struct (line 231)
- All ratio computation helpers: `normalized_ratios_from_usize`, `normalized_ratios_from_f64`, etc. (lines 248-307)
- `count_impls_by_group` (lines 315-334)
- `distribute_event_times` (lines 363-436)
- `build_shatter_groups` (lines 449-468)
- `GroupStats`, `ShatteredTargetLists` structs (lines 473-498)
- `compute_dependency_ratios` (lines 503-525)
- `compute_group_stats` (lines 530-626)
- `adjust_group_symbol_counts` (lines 631-646)
- `group_sccs_by_horizon` (lines 1102-1128)

**What remains in `recommend.rs`:** `ShatterBuilder` and all builder methods, wiring helpers, public `shatter_target` API, tests.

**Step 1:** Create both files with module docs, move functions, add imports.

**Step 2:** Update `recommend.rs` to import from new modules.

**Step 3:** Run `cargo nextest run -p tarjanize-schedule` — all pass.

**Step 4:** Commit.

### Task 8: Split `extract.rs` → extract `anchors.rs`

**Files:**
- Create: `crates/cargo-tarjanize/src/anchors.rs`
- Modify: `crates/cargo-tarjanize/src/extract.rs`
- Modify: `crates/cargo-tarjanize/src/main.rs` or appropriate `mod` declaration site

**Move to `anchors.rs`** (~115 lines):
- `extract_impl_anchors` (lines 1669-1695)
- `collect_anchors_from_type` (lines 1704-1783)

**Step 1-4:** Same pattern as above. Run `cargo nextest run -p cargo-tarjanize`.

### Task 9: Split `orchestrator.rs` → extract `path_transform.rs`

**Files:**
- Create: `crates/cargo-tarjanize/src/path_transform.rs`
- Modify: `crates/cargo-tarjanize/src/orchestrator.rs`

**Move to `path_transform.rs`** (~290 lines):
- `transform_symbol_paths` (lines 758-803)
- `transform_module_paths` (lines 813-862)
- `transform_path` (lines 878-916)
- `find_symbol_target` (lines 709-757)
- `module_contains_symbol` (lines 625-676)
- `module_contains_submodule` (lines 653-676)
- `truncate_impl_child` (lines 678-707)

**Step 1-4:** Same pattern. Run `cargo nextest run -p cargo-tarjanize`.

### Task 10: Split `driver.rs` → extract `profile_processing.rs`

**Files:**
- Create: `crates/cargo-tarjanize/src/profile_processing.rs`
- Modify: `crates/cargo-tarjanize/src/driver.rs`

**Move to `profile_processing.rs`** (~255 lines):
- `collect_profile_paths` (line 1070)
- `load_profile_data_with_span` (line 1104)
- `roll_up_unmatched_with_span` (line 1124)
- `apply_event_times_with_span` (line 1151)
- `append_unmatched_with_span` (line 1172)
- `apply_event_times` (line 1208)
- `unmatched_output_path` (line 1229)
- `append_unmatched_paths` (line 1240)
- `collect_symbol_paths` (line 1288)
- `collect_module_paths` (line 1309)

These functions will reference `CrateIdentity` from Task 3, so Task 3 must be complete first.

**Step 1-4:** Same pattern. Run `cargo nextest run -p cargo-tarjanize`.

---

## Phase 5: Long Function Refactoring

### Task 11: Refactor `compute_effective_horizons` (now in `horizon.rs`)

97 lines with 6 numbered sequential steps. Extract each step into a named helper.

**File:** `crates/tarjanize-schedule/src/horizon.rs` (after Task 7 moves it)

**Strategy:** The function currently has 6 labeled steps (dependency lookup, symbol graph walk, horizon computation, predecessor building, topological sort, propagation). Extract steps 3-6 into helper functions that the main function calls in sequence. Keep the main function as a thin orchestrator.

**Step 1:** Read the function body, identify step boundaries. Extract helpers. Run tests.

**Step 2:** Commit.

### Task 12: Refactor `aggregate_profile`

140 lines with high complexity — per-thread event stacks, self-time calculation, wall-clock tracking.

**File:** `crates/cargo-tarjanize/src/profile.rs`

**Strategy:** Extract thread-stack finalization (the logic that pops remaining stack entries at profile end) and the per-event recording loop body into helper methods on the `ProfileData` struct.

**Step 1:** Read the function, identify extractable blocks. Refactor. Run tests.

**Step 2:** Commit.

### Task 13: Refactor `build_output_graph`

217 lines with `#[expect(clippy::too_many_lines)]`. Three sections: crate naming, path mapping computation, and graph construction with synthetic target synthesis.

**File:** `crates/tarjanize-condense/src/scc.rs`

**Strategy:** Extract the cost model prediction block and the synthetic target creation into separate helper functions. The goal is to remove the `#[expect]` lint suppression.

**Step 1:** Read the function, extract helpers. Remove `#[expect(clippy::too_many_lines)]`. Run tests and clippy.

**Step 2:** Commit.

### Task 14: Refactor `collect_type_deps`

97 lines matching on 20+ `ty::TyKind` variants.

**File:** `crates/cargo-tarjanize/src/extract.rs`

**Strategy:** This function walks rustc type trees. The match arms are mostly single-line dispatches. Only refactor if natural groupings emerge (e.g., "container types" vs "scalar types" vs "special types"). If the match is flat and each arm is 1-3 lines, leave it — a flat match is more readable than scattered helper functions.

**Step 1:** Read the function. Refactor if beneficial, skip if the flat match is clearer.

---

## Phase 6: Verification

### Task 15: Final verification

**Commands:**
```bash
cargo fmt -- --check
cargo clippy --all-targets          # 0 warnings
cargo nextest run                   # 449 tests pass
cargo test --doc                    # doc tests pass
```

**Check:** No new `#[expect]` suppressions added without `reason = "..."`.

**Commit** the final state.

---

## Execution Notes

**Parallelism:** Tasks 6-10 (module splits) are independent and can be dispatched in parallel. Tasks 1-3 are sequential (Task 2 depends on Task 1; Task 3 is independent but logically grouped). Tasks 11-14 depend on Tasks 6-10 (the files they refactor may have moved).

**Risk:** The module splits (Tasks 6-10) are the highest-risk items because they touch many `use` paths and `pub` visibility boundaries. Each should be tested immediately after the move.

**Scope control:** If any task proves more disruptive than expected, it can be skipped — each phase is independently committable.
