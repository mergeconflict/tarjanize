# Cost Model

> **Current status**: The cost model uses `cargo check` for extraction (frontend-only profiling) and a simple scheduling formula: `finish[t] = start[t] + frontend_wall_ms`. Event-level predictions achieve R²=0.91-0.995 and per-symbol attribution achieves R²=0.82-0.997 for lib targets across both tokio and omicron.

## 1. Overview

This document covers tarjanize's cost model: how rustc compilation works, how we predict build times, how we validated the model, and what structural properties of Rust code drive compilation cost.

### The Build Time Problem

Large Rust workspaces suffer from serialized dependency chains. In Omicron (~160 crates), the crates above `nexus-db-model` form a linear chain:

```
nexus-db-model → nexus-auth → nexus-db-queries → nexus-reconfigurator-execution → omicron-nexus
```

Each crate must wait for the previous one, so the last part of the build is completely serialized. Tarjanize identifies opportunities to break up these crates into smaller, parallelizable units.

### What the Model Achieves

- **R² = 0.856** for library targets (86% variance explained)
- **5.32x parallelism ratio** vs 5.05x actual (within 5%)
- Correctly identifies critical path bottlenecks
- Per-symbol attribution R²=0.82-0.997 for lib targets
- Single-event prediction R²=0.91-0.995 across workspaces

The model operates on `cargo check` data, tracking only frontend compilation costs. Backend (LLVM/codegen) runs in parallel via CGUs and doesn't meaningfully affect crate-level scheduling.

---

## 2. Rustc Compilation Reference

### Compilation Phases

Rustc compilation proceeds through multiple phases with different parallelism characteristics:

| Phase | Description | Parallelism |
|-------|-------------|-------------|
| **Frontend** | Parsing, type checking, borrow checking, MIR generation | Largely serial within a crate |
| **Backend** | LLVM optimization and codegen | Parallel across Codegen Units (CGUs) |
| **Linking** | Combining outputs | Serial |

Frontend events have **DefPath** in their `additional_data`, allowing direct attribution to symbols:
```
typeck
  additional_data: ["my_crate::module::Type::{{impl}}::method"]
  duration: 1.23ms
```

Backend events have **CGU names** (not DefPaths):
```
LLVM_module_codegen_emit_obj
  additional_data: ["my_crate.ab33bbc52732c0f1-cgu.01"]
  duration: 45.67ms
```

### Self-Profile Data

Enable self-profiling with:
```bash
RUSTFLAGS="-Zself-profile=/path/to/output" cargo +nightly check
```

This produces `.mm_profdata` files. When loading with `analyzeme`, pass the path **without** the extension:
```rust
ProfilingData::new(Path::new("/tmp/output/tarjanize_schemas-0065639"))
```

Control what's recorded with `-Zself-profile-events=...`:

| Category | What it records | In `default`? |
|----------|-----------------|---------------|
| `query-provider` | Each internal query invocation | Yes |
| `generic-activity` | Non-query compiler work | Yes |
| `query-keys` | Query arguments (DefPaths) | No |
| `function-args` | Generic activity arguments | No |
| `args` | Alias for `query-keys` + `function-args` | No |
| `llvm` | LLVM passes and codegen | No |

**Critical**: To get useful data for cost attribution, use:
```bash
RUSTFLAGS="-Zself-profile=/path -Zself-profile-events=default,args"
```

The `args` flag is essential — without it, `additional_data` is empty and you cannot attribute costs to symbols.

Each event has:
```rust
struct Event {
    event_kind: String,      // "Query", "GenericActivity", etc.
    label: String,           // "typeck", "LLVM_module_codegen_emit_obj", etc.
    additional_data: Vec<String>,  // DefPath or CGU name (if args enabled)
    duration: Option<Duration>,
    thread_id: u32,
}
```

### DefPath Formats and Normalization

DefPaths from self-profile use rustc's internal naming conventions:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `{{impl}}` | An impl block | `my_crate::Type::{{impl}}` |
| `{{impl}}[N]` | Nth impl block on same type | `my_crate::Type::{{impl}}[1]` |
| `{{closure}}` | A closure | `my_crate::foo::{{closure}}` |
| `{{closure}}[N]` | Nth closure in same scope | `my_crate::foo::{{closure}}[0]` |
| `_[N]` | Anonymous/derive-generated item | `my_crate::module::_[7]::{{impl}}` |

**Normalization for symbol matching**:
1. Truncate after `{{impl}}` or `{{impl}}[N]` — impl methods aggregate to impl block
2. Truncate at first `{{closure}}` — closures aggregate to containing function
3. Filter out `std::`, `core::`, `alloc::` paths — external crate costs

Examples:
```
my_crate::Type::{{impl}}::method       → my_crate::Type::{{impl}}
my_crate::foo::{{closure}}::{{closure}} → my_crate::foo
my_crate::module::_[7]::{{impl}}::deserialize → my_crate::module::_[7]::{{impl}}
```

### Event Categorization

```rust
fn categorize_event(label: &str) -> Category {
    let backend_prefixes = ["LLVM", "codegen_module", "link_", "target_machine"];
    let overhead_prefixes = ["incr_comp_", "self_profile", "metadata_", "copy_all_cgu"];

    if backend_prefixes.iter().any(|p| label.starts_with(p)) {
        Category::Backend
    } else if overhead_prefixes.iter().any(|p| label.starts_with(p)) {
        Category::Overhead
    } else {
        Category::Frontend
    }
}
```

**Warning**: `codegen_fn_attrs` starts with "codegen" but is a **frontend query**, not backend work. Use `codegen_module` specifically for actual codegen.

---

## 3. The Cost Model

### Formula

The model predicts target build time using only frontend wall-clock time:

```
finish[t] = start[t] + frontend_wall_ms
start[t]  = max(finish[dep] for dep in dependencies)
```

For test targets, we add the lib target's frontend cost since `cargo test` recompiles the lib with `--test`.

### Target-Level Analysis

The model operates at the **target** level (lib, test, bin), not package level:
- Each target is a separate node in the dependency graph
- Test targets depend on their lib target explicitly
- This naturally resolves dev-dependency "cycles"

### Why Lib Targets Determine the Critical Path

Test and binary targets depend on lib targets, but nothing depends on them. They're leaf nodes in the dependency graph. The critical path — the longest chain that determines minimum build time — always goes through lib targets only. This is why tarjanize focuses on splitting lib targets to improve parallelism.

### Why Frontend Matters

Analysis of 151 lib targets in Omicron:

| Package | Total | Frontend | Codegen | FE % |
|---------|-------|----------|---------|------|
| omicron-nexus | 166.5s | 114.7s | 51.8s | 69% |
| nexus-db-queries | 33.9s | 29.8s | 4.1s | 88% |
| nexus-db-model | 33.5s | 31.1s | 2.4s | 93% |
| **TOTAL (151 libs)** | **596.0s** | **418.9s** | **177.0s** | **70%** |

Frontend is 70% of lib compilation time. The critical path bottlenecks are especially frontend-heavy (88-93%).

### Component Contribution

| Component | Contribution to LIB R² |
|-----------|------------------------|
| Frontend + Backend | 0.752 (baseline) |
| + Metadata | **0.856** (+10.4%) |
| + Linking | 0.756 (+0.4%) |

**Metadata is essential** (+10% R²). **Linking is negligible** (<1% of lib build time).

| Aspect | Linking | Metadata |
|--------|---------|----------|
| Contribution to LIB R² | +0.002 | +0.104 |
| Best predictor R² | 0.42 | 0.71 |
| % of LIB build time | <1% | ~5-15% |

### Metadata Estimation

For synthetic crates (from SCC merging), metadata is estimated from frontend cost.

**Finding: Metadata scales with the cube root of frontend cost.**

The ratio `metadata/frontend` is not constant — smaller crates have disproportionately higher metadata costs. Testing different curve fits on tokio data:

| Model | Formula | R² |
|-------|---------|-----|
| Linear | `0.12*fe + 460` | 0.916 |
| **Power law** | `69 * fe^0.33` | **0.956** |
| Square root | `13*√fe + 230` | 0.961 |

The power law model `metadata = k * frontend^0.33` fits well with **no intercept**, which is important because fixed overhead varies across build environments.

---

## 4. Validation

### Per-Target Accuracy

| Target Type | Count | R² | Interpretation |
|-------------|-------|-----|----------------|
| LIB | 149 | **0.856** | Good fit — this is what matters for critical path |
| TEST | 161 | 0.724 | Lower fit, different scaling |
| LIB + TEST merged | 160 | **0.917** | Best fit when averaging across target types |

Test targets had a different scaling factor (10x higher time per model unit) because `cargo test` recompiles lib code. We now add lib costs to test targets in the model, which improves the parallelism ratio match.

### Parallelism Validation

Comparison of simulated parallel execution against actual `cargo build --timings`:

| Metric | Actual | Model | Match |
|--------|--------|-------|-------|
| Avg parallelism | 5.05x | 5.32x | within 5% |
| Peak concurrency | 141 | 169 | similar |
| Critical path fraction | 71% | 84% | similar |

The absolute times differ by ~5x (constant factor), but proportions match.

### Bottleneck Identification

The model correctly identifies the most expensive lib targets:

| Rank | Target | Model Cost | Actual Time | Scaling |
|------|--------|------------|-------------|---------|
| 1 | omicron-nexus/lib | 877s | 166s | 5.3x |
| 2 | nexus-db-queries/lib | 498s | 34s | 14.6x |
| 3 | nexus-db-schema/lib | 285s | 20s | 14.3x |

Same bottlenecks, same ordering. These three account for 84% of the critical path time.

### Per-Symbol Cost Distribution

Per-symbol cost attribution is **essential** because costs are extremely skewed.

**Omicron (127k symbols):**

| Metric | Frontend | Backend |
|--------|----------|---------|
| Max/min ratio | **1,116,715x** | 8,816x |
| Top 1% share | **75.3%** | 21.6% |
| Top 10% share | 92.8% | 47.7% |
| Most expensive | 225.6 seconds | 2.9 seconds |

**Tokio (2.7k symbols):**

| Metric | Frontend | Backend |
|--------|----------|---------|
| Max/min ratio | 5,091x | 687x |
| Top 1% share | 34.4% | 16.8% |
| Top 10% share | 68.8% | 50.7% |

The most expensive symbols are `{{impl}}` blocks for complex types:
- HTTP entrypoint handlers (Dropshot macros) — 225s each in Omicron
- I/O types (TcpStream, UdpSocket, File) — 2s each in Tokio

**Why this matters for crate splitting**: If we used average cost per symbol, we'd have massive errors. In Omicron, median is 1.3ms, P99 is 407ms (300x higher), max is 225,563ms (170,000x higher than median). Per-symbol attribution ensures accurate cost prediction for any crate split.

### Known Limitations

**Constant Factor (Not a Problem)**: Model costs are ~5x higher than actual build times. This doesn't affect usefulness because relative comparisons are accurate, critical path identification works, and parallelism ratios match. The factor likely comes from profiling overhead and differences between profiled vs actual compilation.

**Residual Analysis**:

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| Backend cost | r = -0.358 | Higher backend → overestimate |
| Symbol count | r = -0.264 | More symbols → overestimate |
| Actual build time | r = +0.408 | Bigger crates → underestimate |
| Number of dependencies | r = +0.066 | No significant effect |
| Linking time | r = -0.009 | No significant effect |

**Outlier Characteristics**: Overestimated crates are DB-heavy (nexus_db_schema with 79k symbols, high frontend costs). Underestimated crates are thin wrappers (few symbols, high linking/metadata overhead relative to frontend).

**Linking Time**: Intentionally excluded. Best predictor achieves only R²=0.42, contributes <1% of build time for LIB targets, and improves model R² by only 0.002.

**Test/Bin Targets**: Tests recompile lib code (modeled by adding lib costs). Linking is significant for tests/bins (10-63%) but poorly predicted. Small binary sample (26) limits validation.

---

## 5. What Drives Compilation Cost

### Cost Breakdown by Category

From `cargo check` profiling across two workspaces:

| Category | Tokio (270 targets) | Omicron (428 targets*) | Key events |
|----------|--------------------:|-----------------------:|------------|
| Trait system | 15.7% | **33.9%** | `impl_trait_header`, `implementations_of_trait`, `specialization_graph_of`, `compare_impl_item` |
| Function body analysis | **26.9%** | 26.4% | `typeck`, `mir_borrowck`, `mir_built`, `mir_pass_*`, `evaluate_obligation`, `check_*` |
| Per-item queries | 17.9% | 8.2% | `predicates_of`, `generics_of`, `param_env`, `type_of`, `fn_sig` |
| Metadata decode | 15.1% | 14.2% | `metadata_decode_entry_*`, `metadata_register_crate` |
| Crate-level & resolution | 10.0% | 6.9% | `expand_crate`, `hir_crate`, `lint_mod`, `late_resolve_crate` |
| Type system queries | 4.3% | 2.0% | `is_copy_raw`, `needs_drop_raw`, `layout_of`, `dropck_outlives` |
| Profiler overhead | 2.9% | 4.3% | `self_profile_alloc_query_strings` |
| Per-crate fixed cost | 1.4% | 1.1% | `defined_lang_items`, `crate_name`, `all_diagnostic_items` |
| Backend residual | 0.4% | 0.4% | `codegen_crate`, `codegen_select_candidate` |
| **Total categorized** | **94.6%** | **97.6%** | |

*\*Omicron excludes nexus-db-queries/lib outlier — see Section 5.7.*

**Workspace-dependent cost profiles**: Trait system costs dominate omicron (34%) but are modest in tokio (16%). Function body analysis is consistent across both (~27%). Metadata decode is similar in both (~15%). Per-item queries are proportionally larger in tokio (18% vs 8%) because tokio has more items relative to its trait complexity. Crate-level resolution is a non-trivial fixed-per-target cost (7-10%) that doesn't scale with code volume.

### Per-Item Queries (8-18% of wall)

These run once per `DefId` (item definition) in the crate. Their total cost scales linearly with item count, modulated by per-item complexity.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `predicates_of` | Where-clauses and trait bounds for an item | Item count x bounds per item |
| `generics_of` | Generic parameter list (type, lifetime, const params) | Item count x params per item |
| `type_of` | The type of an item (return type, field type, etc.) | Item count |
| `explicit_predicates_of` | Explicit where-clauses (vs inferred) | Same as `predicates_of` |
| `param_env` | All bounds in scope, normalized | Item count x inherited bound depth (supertraits) |
| `constness` | Whether an item is const | Pure item count (trivial per-call) |
| `codegen_fn_attrs` | `#[inline]`, `#[cold]`, `#[target_feature]`, etc. | Function count |
| `intrinsic_raw` | Whether a function is a compiler intrinsic | Function count |
| `lookup_stability` | `#[stable]` / `#[unstable]` attributes | Item count |
| `inferred_outlives_of` | Inferred lifetime outlives bounds for ADTs | Struct/enum count x field count |
| `associated_item_def_ids` | Lists associated items of a trait or impl | Trait/impl count x items per trait |
| `associated_item` | Looks up a single associated item | Trait/impl item references |
| `associated_items` | All associated items for an impl/trait | Trait/impl count |
| `inherent_impls` | Inherent impl blocks for a type | Type count |
| `fn_sig` | Function signature | Function count |
| `adt_def` | ADT definition (struct/enum layout) | Struct/enum count |
| `def_span` | Source span for a DefId | Pure item count (trivial) |
| `def_kind` | What kind of item a DefId is (fn, struct, etc.) | Pure item count (trivial) |
| `is_doc_hidden` | Whether an item has `#[doc(hidden)]` | Item count |

**Why these predict so well**: They all scale with "number of things in the crate." A crate with 500 items takes ~5x longer on these queries than one with 100 items.

### Function Body Analysis (26-27% of wall)

Per function body, cost scales with body size/complexity. This is the largest category in both workspaces when all sub-events are included.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `typeck` | Type-checks a function body | Function count x body complexity |
| `mir_borrowck` | Borrow checking | Function count x CFG complexity |
| `mir_built` | Builds MIR from THIR | Function count x expression count |
| `evaluate_obligation` | Evaluates a single trait obligation | Total trait constraints during typeck |
| `check_well_formed` | Validates item well-formedness | Item count x bound complexity |
| `check_mod_privacy` | Privacy/visibility checking per module | Module count x item count |
| `check_liveness` | Dead code / unused variable detection | Function count x variable count |
| `check_match` | Match exhaustiveness and usefulness | Function count x match arms x pattern complexity |
| `thir_body` | Builds THIR from HIR for a function body | Function count x expression count |
| `mir_promoted` | Extracts promoted constants from MIR | Function count x const expression count |
| `mir_drops_elaborated_and_const_checked` | Drop elaboration + const checking | Function count x drop complexity |
| `mir_for_ctfe` | MIR for compile-time function evaluation | Const fn count x body size |
| `optimized_mir` | Final optimized MIR | Function count x MIR size |
| `mir_pass_*` | MIR optimization passes (~20 individual passes) | Function count x MIR size |
| `normalize_canonicalized_projection` | Normalize associated type projections | Generic code volume x associated type usage |

**What drives these**: Two factors multiply: (1) number of function bodies, and (2) size of each body. MIR passes are particularly sensitive to total MIR size (basic blocks x statements). The `mir_pass_*` family alone contains ~20 individual passes.

### Trait System (16-34% of wall)

These scale with the trait implementation landscape, including upstream crates. The most workspace-dependent category: 34% in omicron (heavy serde/diesel usage) vs 16% in tokio.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `implementations_of_trait` | All impls of a given trait across all crates | Traits used x total impls in ecosystem |
| `specialization_graph_of` | Specialization DAG for a trait | Traits with multiple overlapping impls |
| `impl_trait_header` | Trait and self-type for an impl block | Total visible trait impls (local + upstream) |
| `impl_parent` | Parent trait of a specializing impl | Trait specialization depth |
| `trait_impls_of` | All impls of a trait within a crate | Trait count x local impl density |
| `compare_impl_item` | Validate impl items match trait definition | Trait impl count x items per impl |
| `coherent_trait` | Coherence checking for a trait | Trait count x overlapping impl potential |

**What drives these**: Partly the crate's own code (how many trait impls it defines), but largely the *dependency ecosystem*. A crate that uses traits with thousands of impls across its dependencies (e.g., `serde::Serialize`) pays for searching all of them. The top four events account for 80% of the category in omicron.

### Metadata Decoding (14-15% of wall)

Reading upstream crate `.rmeta` files. This is the cost of having dependencies. Consistent across both workspaces at ~15%.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `metadata_decode_entry_impl_trait_header` | Trait impl headers from upstream | Upstream trait impl count |
| `metadata_decode_entry_implementations_of_trait` | Finding impls across upstream crates | Traits used x upstream impl count |
| `metadata_decode_entry_impl_parent` | Parent trait info from upstream | Upstream specialization depth |
| `metadata_decode_entry_module_children` | Items exported by upstream modules | Upstream crate count x exported items |
| `metadata_decode_entry_type_of` | Type info from upstream items | Upstream items referenced |
| `metadata_decode_entry_generics_of` | Generic params from upstream items | Upstream items referenced |
| `metadata_register_crate` | Initial metadata load for a dependency | One-time per upstream crate |

**What drives these**: Two factors: (1) number of direct + transitive dependencies, and (2) how much of each dependency's API surface is touched. Note that `metadata_decode_entry_*` events mirror the per-item and trait system queries — they're the same queries, but answered by reading `.rmeta` files instead of computing locally.

### Crate-Level & Resolution (7-10% of wall)

Per-crate overhead: macro expansion, name resolution, HIR construction, linting, metadata generation. Closer to fixed costs than the categories above — when splitting a crate, both halves pay this overhead.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `expand_crate` | Macro expansion for the entire crate | Macro invocation count + proc macro count |
| `late_resolve_crate` | Name resolution (post-expansion) | Item count x import complexity |
| `hir_crate` | Build HIR for the entire crate | Crate size (one-time) |
| `generate_crate_metadata` | Write `.rmeta` for downstream crates | Exported item count |
| `lint_mod` | Run lint passes per module | Module count x item count |
| `visible_parent_map` | Map items to their visible parent module | Module tree depth |
| `module_children` | Children of each module | Module count x items per module |

### Cross-Workspace Stepwise Regression

Forward stepwise regression across 13 workspaces, finding the minimum set of self-profile event labels that predict `frontend_wall_ms` per target. Each model selects features greedily by adjusted R², stopping at adj R² > 0.999 or 10 features.

| Workspace | Targets | Wall (s) | Features | adj R² | Top univariate predictor | Univ R² |
|-----------|--------:|---------:|---------:|-------:|--------------------------|--------:|
| tokio | 270 | 240 | 4 | 0.9991 | `predicates_of` | 0.995 |
| rust-analyzer | 114 | 546 | 5 | 0.9993 | `predicates_of` | 0.993 |
| opte | 21 | 85 | 5 | 0.9991 | `inferred_outlives_of` | 0.993 |
| sqlx | 9 | 13 | 3 | 0.9995 | `metadata_decode_entry_generics_of` | 0.994 |
| progenitor | 26 | 171 | 1 | 0.9993 | `generics_of` | 0.999 |
| ruff | 148 | 734 | 6 | 0.9994 | `explicit_predicates_of` | 0.987 |
| propolis | 46 | 232 | 5 | 0.9992 | `type_of` | 0.981 |
| uv | 136 | 773 | 5 | 0.9992 | `inferred_outlives_of` | 0.980 |
| meilisearch | 71 | 529 | 5 | 0.9992 | `check_mod_privacy` | 0.974 |
| nushell | 30 | 160 | 6 | 0.9995 | `inferred_outlives_of` | 0.967 |
| bevy | 403 | 1838 | 10 | 0.9989 | `inhabited_predicate_type` | 0.854 |
| helix | 17 | 297 | 3 | 0.9997 | `crate_inherent_impls` | 0.988 |
| omicron* | 428 | 3770 | 8 | 0.9992 | `check_liveness` | 0.913 |

*\*Omicron excludes nexus-db-queries/lib outlier.*

**All 13 workspaces achieve adj R² > 0.999** with 1-10 features. A small number of rustc query self-times explain >99.9% of variance in compilation wall time across diverse Rust codebases.

#### Feature Frequency Across Models

43 unique features were selected across all 13 models. Features appearing in 2+ models:

| Feature | Category | Models |
|---------|----------|-------:|
| `metadata_decode_entry_impl_trait_header` | metadata | 6 |
| `predicates_of` | per_item | 4 |
| `inferred_outlives_of` | per_item | 4 |
| `def_span` | per_item | 3 |
| `specialization_graph_of` | trait_system | 3 |
| `check_mod_privacy` | fn_body | 2 |
| `mir_borrowck` | fn_body | 2 |
| `generics_of` | per_item | 2 |
| `typeck` | fn_body | 2 |
| `lib_features` | other | 2 |
| `self_profile_alloc_query_strings` | profiler_overhead | 2 |
| `metadata_decode_entry_implementations_of_trait` | metadata | 2 |

**Key findings**:

1. **Per-item events dominate** as top univariate predictors (10 of 13 workspaces). The `predicates_of` / `inferred_outlives_of` / `explicit_predicates_of` / `generics_of` family is the strongest universal signal.
2. **`metadata_decode_entry_impl_trait_header`** is the most universal secondary feature (6 models), capturing the cost of reading upstream trait impl metadata.
3. **Bevy is the hardest to fit** — needs all 10 features and has the weakest top univariate predictor (R²=0.854). Its trait system dominates at 44.3% of wall time, reflecting heavy generics/ECS usage.
4. **Omicron's outlier is real** — excluding `nexus-db-queries/lib` brings R² from 0.952 to 0.999.

#### Universal Pooled Model (All Workspaces Combined)

Forward stepwise regression on all 1719 targets pooled together (419 candidate features present in >= 10% of targets).

| Step | adj R² | Feature | Category |
|------|-------:|---------|----------|
| 1 | 0.783 | `needs_drop_raw` | type_system |
| 2 | 0.900 | `self_profile_alloc_query_strings` | profiler_overhead |
| 3 | 0.939 | `check_well_formed` | fn_body |
| 4 | 0.976 | `metadata_decode_entry_implementations_of_trait` | metadata |
| 5 | 0.986 | `impl_self_is_guaranteed_unsized` | trait_system |
| 6 | 0.990 | `specialization_graph_of` | trait_system |
| 7 | 0.995 | `mir_borrowck` | fn_body |
| 8 | 0.997 | `impl_super_outlives` | trait_system |
| 9 | 0.998 | `maybe_building_test_harness` | other |
| 10 | 0.998 | `is_mir_available` | other |
| 11 | 0.999 | `typeck` | fn_body |
| 12 | 0.999 | `metadata_decode_entry_mir_coroutine_witnesses` | metadata |
| 13 | 0.999 | `predicates_of` | per_item |

**Final: R²=0.9992, adj R²=0.9992** with 13 features across 1719 targets.

Per-workspace fit using the universal model:

| Workspace | Targets | R² |
|-----------|--------:|---:|
| helix | 17 | 0.99998 |
| omicron* | 428 | 0.99962 |
| meilisearch | 71 | 0.99927 |
| uv | 136 | 0.99926 |
| rust-analyzer | 114 | 0.99798 |
| ruff | 148 | 0.99785 |
| propolis | 46 | 0.99728 |
| progenitor | 26 | 0.99648 |
| bevy | 403 | 0.99496 |
| tokio | 270 | 0.99383 |
| nushell | 30 | 0.99315 |
| opte | 21 | 0.98927 |
| sqlx | 9 | 0.96850 |

**Key differences from per-workspace models**:

1. **Different features win universally.** `needs_drop_raw` (type_system, univariate R²=0.783) is the best pooled predictor despite never winning per-workspace. Per-item events like `predicates_of` (R²=0.995 in tokio) drop to R²=0.66 when pooled because their absolute scale shifts between workspaces.
2. **The model draws from all categories**: type_system, profiler_overhead, fn_body, metadata, trait_system, other, per_item. No single category dominates.
3. **Per-workspace models still win** — every workspace achieves adj R²>0.999 with its own model, while the universal model ranges from 0.969 to 0.999.

### Cross-Workspace Comparison

| Metric | Tokio (`check`) | Omicron (`check`*) |
|--------|----------------:|-----------------:|
| Targets with profiling data | 270 | 428 |
| Lib targets | 6 | 136 |
| Unique event labels | 421 | 511 |
| Best single-event R² (all) | 0.995 (`predicates_of`) | 0.913 (`check_liveness`) |
| Per-symbol R² (all) | 0.814 | 0.375 |
| Per-symbol R² (lib) | 0.997 (n=6) | 0.822 |
| Per-symbol mean % of wall | 3.3% | 3.0% |
| Total events / wall ratio | 0.98x | 0.99x |
| Top cost category | Metadata decode / trait queries | Trait system (40%) + metadata (23%) |

*\*Omicron excludes nexus-db-queries/lib outlier.*

Key patterns across both workspaces:
1. Single event labels are excellent wall-time predictors (R²=0.91-0.995)
2. Per-symbol attribution captures only 3% of wall time but predicts well for lib targets (R²=0.82-0.997)
3. Trait resolution and metadata decoding are consistently the largest frontend costs
4. With `cargo check`, event costs match wall time almost exactly (0.98-0.99x)
5. Different workspaces have different top predictors — tokio favors per-item queries, omicron favors MIR passes and macro resolution

### Impact of `cargo check` vs `cargo build`

| Metric | `cargo build` | `cargo check` | Change |
|--------|-------------:|-------------:|-------:|
| Total wall (tokio) | 1,108,397 ms | 239,775 ms | **4.6x smaller** |
| Total event costs | 1,505,852 ms | 234,732 ms | 6.4x smaller |
| Events / wall ratio | 1.36x | 0.98x | **No inflation** |
| Event labels | 506 | 421 | 85 backend labels gone |
| Best single-event R² | 0.972 | **0.995** | +0.023 |
| Per-symbol R² | 0.674 | **0.814** | +0.140 |

The `cargo build` profiles were contaminated by backend events (LLVM, codegen, linking) that inflated `frontend_wall_ms`. With `cargo check`, `frontend_wall_ms` is a clean measurement of frontend compilation time.

### The nexus-db-queries Anomaly

`nexus-db-queries/lib` has 1.3M ms wall time but only 109K ms in event costs — 92% of wall time is invisible to the profiler. This single outlier drops the all-target best R² from 0.913 to 0.478. It is excluded from correlation analysis. This likely represents a profiler limitation or extreme trait-system behavior.

---

## 6. Structural Properties

### Properties That Matter

The compiler events above are driven by a small number of countable structural properties:

**1. Total Item Count** — The single most important driver. Not just module-level definitions, but everything with a `DefId`: closures, generic params, anonymous consts, opaque types. A function with 3 closures and 2 generic params produces 6+ DefIds. Our symbol count undercounts because we collapse associated items and don't count closures/generic params.

**2. Predicate / Bound Count** — Drives `predicates_of`, `explicit_predicates_of`, `param_env`, `evaluate_obligation`. The sum of trait bounds across all items.

**3. Generic Parameter Count** — Drives `generics_of` directly, and indirectly `predicates_of`.

**4. Function Body Size** — Drives `typeck`, `mir_built`, `mir_borrowck`, all `mir_pass_*`. Two possible measurements: THIR expression count (available during extraction) or MIR basic block / statement count.

**5. Trait Impl Count (Local)** — Drives `impl_trait_header`, `trait_impls_of`, `specialization_graph_of`.

**6. Upstream Dependency Surface** — Drives all `metadata_decode_*` events. This is the one property that's NOT about the crate's own code — it's about the dependency graph. When splitting a crate, both halves still need to decode the same upstream metadata.

### What We Could Extract

We already call most relevant `tcx` queries in `extract.rs` for dependency analysis. Adding structural counters would be cheap.

**Per-Symbol:**

| Property | How to get it | Which events it predicts |
|----------|--------------|------------------------|
| DefId count | Count descendant DefIds | All per-item queries |
| Predicate count | `tcx.predicates_of(def_id).predicates.len()` | `predicates_of`, `param_env`, `evaluate_obligation` |
| THIR expression count | Count during existing THIR walk (free) | `typeck`, `mir_built` |
| MIR statement count | `tcx.mir_built(def_id)` → sum statements | All `mir_pass_*`, `mir_borrowck` |

**Per-Target:**

| Property | How to get it | Which events it predicts |
|----------|--------------|------------------------|
| Total DefId count | Sum of per-symbol DefId counts | All per-item queries |
| Total predicate count | Sum of per-item predicate counts | `predicates_of` family |
| Total MIR size | Sum of basic blocks across all fns | `mir_pass_*` family |
| Trait impl count | Count DefKind::Impl { of_trait: true } | Trait system events |
| Dependency count | Already have `target.dependencies.len()` | `metadata_decode_*` |

### Note on Event-Based Model

The structural extraction approach described above has been superseded by the event-based cost model (see [docs/structural-extraction-plan.md](structural-extraction-plan.md)), which fits a regression model directly against event self-times. Structural properties remain useful for predicting costs of *synthetic* crates that have never been compiled.

---

## 7. Implications and Open Questions

### Why Crate Splitting Helps

Splitting crates reduces **frontend cost** per crate, enabling earlier downstream compilation. Since frontend is 70% of lib compilation and largely serial within a crate, splitting the linear chain (nexus-db-schema → nexus-db-queries → omicron-nexus) allows more parallel frontend work.

### What the Model Can Identify

1. **Critical path**: Which chain of lib dependencies determines minimum build time
2. **Bottleneck targets**: Which libs contribute most to the critical path
3. **Splitting opportunities**: Which SCCs could be split to reduce critical path

### Open Questions

1. **Does splitting reduce total frontend cost?** Or does cross-crate overhead increase it?
2. **What's the per-crate overhead?** More crates = more metadata files to read
3. **Where are the natural split points?** Symbol dependency analysis should reveal independent subsets
4. **Which structural features are additive?** For crate splitting, we need `cost(A ∪ B) ≈ cost(A) + cost(B)`. Per-item queries should be perfectly additive. Trait system queries might not be — moving an impl to a different crate changes what's visible.
5. **How do we handle metadata decode costs?** These depend on the dependency graph, not the crate's own code. Both halves still need to decode the same upstream metadata (~15% of wall in both workspaces).
6. **What's happening with nexus-db-queries/lib?** 92% of wall time invisible to the profiler. Needs dedicated investigation.
7. **Do we need workspace-specific models?** Function body analysis (~27%) and metadata decode (~15%) are consistent. The main divergence is trait system (34% omicron vs 16% tokio).
8. **What regression model for cost prediction?** Linear might suffice if costs are roughly additive, but metadata decode is a fixed overhead per target, and trait system costs scale non-linearly with ecosystem size.

---

## Appendix A: Backend Cost Tracking *[Historical]*

> tarjanize no longer tracks backend costs. This appendix is retained as reference for how rustc's backend works.

### Codegen Units (CGUs)

A Codegen Unit is a chunk of code that LLVM compiles independently. Multiple CGUs compile in parallel.

**Partitioning strategy**: Rustc creates two CGUs per source-level module (stable + volatile), up to 16 CGUs for release builds, 256 for debug. Controllable via `-Ccodegen-units=N`.

**CGU naming**: `{crate_name}.{hash}-cgu.{index}` (e.g., `tarjanize_schemas.ab33bbc52732c0f1-cgu.01`).

**Parallelism**: The slowest CGU is the bottleneck. More CGUs = better parallelism but potentially worse optimization (LLVM can't inline across CGU boundaries).

### Mono Items

A "mono item" is a monomorphized function or static that becomes machine code. Generic functions become multiple mono items (`foo::<i32>`, `foo::<String>`, etc.).

Enable with:
```bash
RUSTFLAGS="-Zprint-mono-items=yes" cargo +nightly build 2>&1 | tee mono_items.txt
```

Output format:
```
MONO_ITEM fn my_crate::foo::bar @@ my_crate.ab33bbc52732c0f1-cgu.01[External]
```

Mono item paths include special formats: generic instantiations (`<Type>`), closures (`{closure@src/lib.rs:131:59}`), shims (`- shim(vtable)`), and items in multiple CGUs.

### Joining Self-Profile with Mono Items

Frontend costs (from self-profile) have DefPath → directly attributable to symbols. Backend costs have CGU name → not directly attributable.

**Join process**: Parse self-profile for `CGU name → total LLVM duration`. Parse mono items for `CGU name → [mono item paths]`. Join on CGU name. Distribute each CGU's time across its mono items.

**Complications**: Mono items ≠ symbols (generics multiply), CGU names include a hash that changes between builds (must use same compilation), and closures/shims need special handling.

### Backend Cost Distribution to Symbols

To distribute CGU costs to symbols:
1. Parse mono-items for `CGU → [mono item paths]`
2. Each mono item gets `CGU_cost / N` (equal distribution)
3. Normalize: strip generic params, closure suffixes, map shims to source
4. Aggregate: sum costs for mono items mapping to the same symbol

### Schema Design for Cost Storage

The original design stored both frontend and backend costs per symbol:

```rust
struct Symbol {
    frontend_cost_ms: f64,  // from DefPath events
    backend_cost_ms: f64,   // distributed from CGU
}

struct Crate {
    linking_ms: f64,
    metadata_ms: f64,
    root: Module,
}
```

Wall-clock prediction:
```
wall_clock = frontend_time + backend_time + overhead
  frontend_time = Σ symbol.frontend_cost_ms    (serial)
  backend_time  = max(module.backend_cost_ms)  (parallel!)
  overhead      = linking_ms + metadata_ms
```

Backend tracking was removed because CGU→symbol attribution via mono items proved unreliable, and backend time doesn't meaningfully affect the critical path for crate splitting.

### Observed Ratios

| Crate Type | Example | Frontend | Backend | Overhead |
|------------|---------|----------|---------|----------|
| Type-heavy library | serde_core | 88% | 2.5% | 10% |
| Code-heavy library | syn | 67% | 30% | 4% |
| Small with derives | tarjanize-schemas | 40% | 47% | 13% |

---

## Appendix B: Investigation History

This section documents how we arrived at the current model through iterative validation.

**Initial validation (crate-level)**: Pearson r = 0.949, R² = 0.900, model ≈ 2.9x actual. Looked promising but masked target-type differences.

**Target-level breakdown**: LIB R²=0.856 with slope 0.15; TEST R²=0.724 with slope 1.59 (~10x different). Tests take 10x more time per unit of modeled cost because they recompile lib code.

**Component contribution testing**: Frontend+Backend+Metadata (full) R²=0.856; Frontend only R²=0.844; Backend only R²=0.035; Linking+Metadata alone R²=0.927 (surprisingly high, but not actionable since synthetic crates lack measured linking/metadata).

**The rmeta pipelining discovery** *[Historical]*: Test targets started almost immediately after their lib target started — not when the lib finished. `omicron-nexus/test` started 3.7s after lib began, though lib took 166s to complete. The answer: Cargo uses pipelined compilation. Downstream crates only need rmeta (type signatures), available after frontend — roughly 70% through. With the simplified frontend-only model, pipelining is implicit.

**Test cost correction**: Adding lib costs to test targets improved parallelism ratio from 3.31x to 5.32x (actual: 5.05x).

**Metadata estimation**: Frontend cost (all symbols) is the best predictor of metadata cost (R²=0.705). Public symbols predict metadata *worse* than all symbols — rustc metadata includes everything for downstream compilation.

---

## Appendix C: Methodology and Reproduction

### Data Collection

- **Model predictions**: `cargo tarjanize` with `CARGO_INCREMENTAL=0` and `-Zself-profile`
- **Actual build times**: `cargo build --timings` (default settings)
- **Scope**: Clean builds only (no incremental compilation)

### Test Data

- **Workspace**: Omicron (Oxide Computer Company)
- **Packages**: 161 workspace packages
- **Targets**: 336 (149 lib, 161 test, 26 bin)
- **Symbols**: 127,810 (127,200 with frontend costs, 63,094 with backend costs)
- **Cost distribution**: Top 1% of symbols account for 75% of frontend cost

### Statistical Validation

- **Pearson correlation (r)**: Measures straight-line relationship (-1 to +1). Our r = 0.925 for libs.
- **R²**: Correlation squared (0.925² = 0.856). Fraction of variation "explained" by the model.
- **p-value**: For r = 0.925, n = 149, the p-value is essentially zero (31 standard deviations from random chance).

### Reproducing the Validation

```bash
# 1. Generate symbol graph with profiling
cd /path/to/omicron
CARGO_INCREMENTAL=0 cargo tarjanize --profile > omicron.json

# 2. Run a timed build
cargo build --timings

# 3. Run cost analysis
tarjanize cost -i /path/to/omicron.json
```

---

## Appendix D: Validation Directory Structure

Cross-project validation data lives in `/home/debian/github/validation/`, separate from the tarjanize repo.

```
/home/debian/github/validation/
├── repos/                     # Cloned workspace repos (read-only reference)
│   ├── bevy/
│   ├── diesel/
│   ├── helix/
│   ├── meilisearch/
│   ├── nushell/
│   ├── ruff/
│   ├── rust-analyzer/
│   ├── sqlx/
│   ├── tokio/                 # Primary validation target
│   ├── uv/
│   └── zed/
│
└── data/                      # Extracted symbol graphs, logs, and analysis output
    ├── tokio/                 # Per-workspace data directory
    │   ├── symbol_graph_wall.json
    │   ├── cargo-timing-wall.html
    │   ├── tarjanize.log
    │   └── cost_output_new.txt
    │
    ├── omicron/               # ~160-crate workspace (primary R²=0.856 validation)
    │   ├── symbol_graph.json
    │   ├── cargo-timing.html
    │   ├── build.log
    │   └── tarjanize.log
    │
    └── helix/
        └── tarjanize.log
```

### Regenerating Data

```bash
# Extract symbol graph with profiling (from the workspace repo)
cd /home/debian/github/validation/repos/tokio
cargo-tarjanize -o /home/debian/github/validation/data/tokio/symbol_graph_wall.json -v \
    2>&1 | tee /home/debian/github/validation/data/tokio/tarjanize.log

# Run cost analysis (from the tarjanize repo)
cargo run --release -p tarjanize -- cost \
    -i /home/debian/github/validation/data/tokio/symbol_graph_wall.json

# Collect actual build times for comparison
cd /home/debian/github/validation/repos/tokio
CARGO_INCREMENTAL=0 cargo build --timings --release
cp target/cargo-timings/cargo-timing-*.html \
    /home/debian/github/validation/data/tokio/cargo-timing-wall.html
```

---

## References

- [Intro to rustc's self profiler](https://blog.rust-lang.org/inside-rust/2020/02/25/intro-rustc-self-profile/) - Inside Rust Blog
- [Parallel compilation - Rust Compiler Development Guide](https://rustc-dev-guide.rust-lang.org/parallel-rustc.html)
- [Monomorphization - Rust Compiler Development Guide](https://rustc-dev-guide.rust-lang.org/backend/monomorph.html)
- [measureme repository](https://github.com/rust-lang/measureme) - Profiling tools
- [Back-end parallelism in the Rust compiler](https://nnethercote.github.io/2023/07/11/back-end-parallelism-in-the-rust-compiler.html) - Nicholas Nethercote
- [Faster compilation with the parallel front-end](https://blog.rust-lang.org/2023/11/09/parallel-rustc/) - Rust Blog
