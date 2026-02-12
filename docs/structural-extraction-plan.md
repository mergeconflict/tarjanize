# Structural Extraction Plan

**Status: Implemented.** All implementation steps (1-6) below have been
completed. The event-based cost model is the production approach. This
document is retained as design rationale — it explains *why* the cost model
works and documents the validation results.

---

Extract per-symbol structural properties during `cargo tarjanize` that predict
compilation wall-clock time, enabling cost prediction for synthetic targets
produced by `tarjanize condense`.

## Problem

We can predict wall-clock compilation time with R²>0.999 per-workspace using
event self-times from `-Zself-profile`. But event self-times only exist for
targets that have been compiled. After `tarjanize condense` splits crates into
new synthetic targets, we need to predict their compilation times without
compiling them.

## Approach

1. **Record per-symbol event times** during `cargo tarjanize` by
   attributing DefPath-bearing event self-times from `-Zself-profile` onto
   each symbol's `event_times_ms` map, broken down by event label. These
   per-event costs travel with symbols through condense.
2. **Fit a per-workspace model** in `tarjanize cost`:
   `wall_time = a · Σ(symbol_attr) + b · Σ(metadata_decode_*)`
   where `Σ(symbol_attr)` is the sum of all `Symbol.event_times_ms` values
   across the target's symbols. Fitted using the original targets where we
   have both profiled costs and actual wall times.
3. **Predict synthetic target costs** by summing `event_times_ms` values
   across the synthetic target's symbols, estimating metadata decode cost,
   and applying the fitted model.

## Event Attribution

Rustc's `-Zself-profile` emits events with `additional_data[0]` that falls
into two useful categories (see Appendix A for the full 24-event breakdown):

1. **Symbol-attributable**: Events with a clean DefPath in `additional_data[0]`
   (e.g., `typeck` → `my_crate::foo::{{closure}}`). Any event with a DefPath
   should be attributed to its nearest ancestor symbol after descendant
   aggregation. This covers 14 of our 24 key events (~70-80% of profiled wall
   time) plus any other DefPath-bearing events the compiler emits.

2. **Metadata decode**: `GenericActivity` events with `metadata_decode_*`
   labels and no `additional_data`. These represent the cost of reading upstream
   `.rmeta` files — pure per-target overhead driven by the dependency graph, not
   attributable to any symbol. ~5-15% of wall time.

We don't care about the remaining events; these two categories above are already
both excellent predictors of total wall time for a target, and significant
components percentage-wise of total wall time. We record and sum the events
in TargetTimings.event_times_ms for the sake of debugging (summing all events
between a target and it symbols should be very close to the measured wall time),
but they don't factor into our cost model.

## Cost Model

### Validated model: two-term linear regression with no intercept

Cross-workspace validation (13 workspaces, 1719 targets) confirms that a
simple two-term model with no intercept predicts wall time with median
R²=0.9944:

```
wall_time = a · Σ(symbol_attr) + b · Σ(metadata_decode_*)
```

Where:
- `Σ(symbol_attr)` = sum of all `Symbol.event_times_ms` values across all
  symbols in the target (i.e., total attributed self-time)
- `Σ(metadata_decode_*)` = sum of all `metadata_decode_*` entries in
  `TargetTimings.event_times_ms`
- `a` ≈ 3.4x (median across workspaces, range 2.7-5.4)
- `b` ≈ 2.8x (median across workspaces, range 1.2-5.2)

Note: the `a` and `b` coefficients above were fitted against target-level
event sums (correct per-target totals). The descendant aggregation bug has
been fixed, so `Σ(symbol_attr)` now matches those sums.

No constant term is needed — a target with zero code and zero dependencies
has zero compilation time. The two coefficients `a` and `b` are fitted
per-workspace via ordinary least squares.

#### Validation results

| Workspace     | Targets | R²     | a (attr) | b (meta) |
|---------------|--------:|-------:|---------:|---------:|
| uv            |     136 | 0.9986 |    3.000 |    3.548 |
| tokio         |     270 | 0.9955 |    3.295 |    2.961 |
| progenitor    |      26 | 0.9954 |    2.964 |    5.182 |
| rust-analyzer |     114 | 0.9952 |    3.138 |    2.742 |
| nushell       |      30 | 0.9946 |    3.287 |    2.432 |
| propolis      |      46 | 0.9944 |    3.416 |    2.903 |
| meilisearch   |      71 | 0.9944 |    2.684 |    4.826 |
| omicron*      |     426 | 0.9917 |    3.480 |    2.722 |
| ruff          |     148 | 0.9900 |    3.401 |    2.301 |
| opte          |      21 | 0.9743 |    3.552 |    1.229 |
| bevy          |     403 | 0.9704 |    3.458 |    2.743 |

*omicron excludes nexus-db-queries (pathological: 91.8% of wall time is
uninstrumented, likely proc macro expansion from diesel). helix and sqlx
excluded (only 17 and 9 targets respectively; helix has a profiling artifact
where `self_profile_alloc_query_strings` is 95% of wall for helix-core).

#### Why this works

The 16 attributable events cover only ~20-28% of wall time, and the 4 metadata
events cover another ~5-15%. Yet together they predict ~99% of wall time
variance. This is because the remaining unattributable events
(`evaluate_obligation`, `type_op_prove_predicate`, `impl_trait_header`,
`impl_parent`, `trait_impls_of`, `incr_comp_encode_dep_graph`, etc.) scale
proportionally with the attributable events — they're not independent noise,
they're correlated overhead. The two coefficients `a` and `b` absorb this
proportional overhead.

Note: The validation above used target-level event sums (correct per-target
totals). The descendant aggregation bug has since been fixed, so per-symbol
`event_times_ms` attribution is now accurate.

### Schema

**`Symbol.event_times_ms`** (HashMap<String, f64>, replaces `frontend_cost_ms`):
Self-time breakdown by event label for events attributed to this symbol after
descendant aggregation. For example, a function `foo<T>` might have:
`{"typeck": 1.2, "mir_borrowck": 0.8, "predicates_of": 0.3, ...}`. Any event
the compiler emits with a valid DefPath in `additional_data[0]` gets
attributed — no filtering by event label.

The total attributed cost for a symbol is `Σ(event_times_ms.values())`. This
replaces the old scalar `frontend_cost_ms` field and provides per-event
visibility for regression testing: if a rustc update changes which events fire
or shifts time between events, the breakdown makes it immediately visible.
Serialization skips the field when the map is empty (same as other optional
fields).

**`TargetTimings.wall_time_ms`** (f64, renamed from `frontend_wall_ms`):
Wall-clock elapsed time of compilation (min start to max end across all
profiled events).

**`TargetTimings.event_times_ms`** (HashMap<String, f64>, renamed from
`event_costs`): Self-time breakdown by event label for events that were NOT
attributed to a symbol. This ensures no double-counting between
`Symbol.event_times_ms` and `TargetTimings.event_times_ms` — every
millisecond of profiled self-time lands in exactly one place. The metadata
decode cost for the regression model is derived by summing all
`metadata_decode_*` entries at query time — no separate field needed.

Note: `Symbol.event_times_ms` and `TargetTimings.event_times_ms` share the
same field name and type but are complementary: the symbol map contains
events attributed to that specific symbol (via DefPath), while the target
map contains unattributed remainder events. Together they account for all
profiled self-time.

### How predictions work

**Fitting** (in `tarjanize cost`): For each target in the original workspace,
compute `Σ(symbol_attr)` by summing all `Symbol.event_times_ms` values across
all symbols in the target, and compute `Σ(metadata_decode_*)` by summing all
`metadata_decode_*` entries in `TargetTimings.event_times_ms`. Fit the two
coefficients `a` and `b` via least-squares against `wall_time_ms`. Report R²
and per-target accuracy.

**Predicting** (in `tarjanize condense`): For each synthetic target produced by
condense:
1. Sum all `event_times_ms` values across all symbols assigned to that target.
2. Estimate metadata decode cost as the max `Σ(metadata_decode_*)` across all
   constituent original targets that were merged into this synthetic target.
   (Rationale: a synthetic target inherits the dependencies of all its
   constituent targets, so its metadata cost is at least as large as the
   largest constituent. Using max is conservative — the true cost may be
   slightly higher due to additional cross-piece dependencies, but max is a
   good lower bound.)
3. Apply: `predicted_wall_ms = a · Σ(symbol_attr) + b · metadata_decode_ms`.

### Shared implementation

The regression fitting function should live in a shared crate (likely
`tarjanize-cost` re-exported or a new utility module) so both `tarjanize cost`
and `tarjanize condense` can use it. The function takes a slice of
`(symbol_attr_sum, metadata_decode_sum, wall_time_ms)` triples and returns
`(a, b, r_squared)`.

## Implementation Order (all completed)

1. **Fix descendant DefPath aggregation** (`profile.rs`): Rewrite
   `normalize_frontend_path` to peel all internal DefPath segments (generic
   params, anonymous consts, opaque types, constructors, coroutine bodies) back
   to the nearest ancestor symbol — not just closures and impl methods. This is
   the critical bug fix: without it, per-item events for descendants are silently
   lost, and per-symbol attribution is far too low. After fixing, all
   DefPath-bearing events roll up correctly into `Symbol.event_times_ms` and
   are excluded from `TargetTimings.event_times_ms` to prevent double-counting.
   Add unit tests.
2. **Schema changes** (`tarjanize-schemas`): Replace `frontend_cost_ms: f64`
   with `event_times_ms: HashMap<String, f64>` on `Symbol`. Rename
   `frontend_wall_ms` → `wall_time_ms` on `TargetTimings`, `event_costs` →
   `event_times_ms` on `TargetTimings`.
3. **Regression function** (`tarjanize-cost`): Implement two-variable
   no-intercept least-squares: `wall = a·attr + b·meta`. Return `(a, b, r_squared)`.
   Add a TODO indicating that this should eventually be moved to a different crate
   for numerical stuff, or we should just use an existing crate from the ecosystem.
   Add unit tests for this function.
4. **Cost reporting** (`tarjanize cost`): Fit the model on original targets,
   report coefficients, R², and per-target predicted vs actual. Add tests.
5. **Condense prediction** (`tarjanize condense`): For each synthetic target,
   sum attributed costs and take max metadata cost from constituents, apply
   fitted model to predict wall time. Add tests.
6. **Validation**: Run on all 13 validation workspaces, verify R² matches the
   Python validation results above.

## Resolved Questions

Issues from earlier review that are now resolved by the event-based cost model.

### Structural properties deferred

The original plan proposed extracting 4 structural properties per symbol
(defid_count, mir_statements, predicate_count, trait_impl_count). The validated
event-based model (`wall = a·attr + b·meta`) achieves R²>0.99 using profiled
event self-times directly, without needing structural proxies. Structural
extraction is deferred — it would only be needed if we want to predict costs
for code that has never been compiled (e.g., after major refactoring). For the
condense use case, all symbols come from compiled targets with profiled costs.

### THIR expression count, MIR statement count, predicate count

Not needed for the event-based model. The 16 attributable event self-times
already capture the cost of typeck, mir_borrowck, predicates_of, etc. directly.
These structural properties would only add value as proxies if we didn't have
the event times — but we always do, since extraction requires compilation.

### Metadata decode handling

Resolved: derive metadata decode cost by summing `metadata_decode_*` entries
in `event_times_ms` at query time — no dedicated field needed. For synthetic
targets after condense, use the max across constituent targets. This replaces
the earlier `dep_count` proxy approach and is both simpler and more accurate.

### Descendant DefPath aggregation (fixed)

The entire cost model depends on `Symbol.event_times_ms` being the correct
attribution of DefPath-bearing event self-times per symbol.
`normalize_frontend_path` in
`profile.rs` is responsible for rolling descendant DefPaths up to their parent
symbol, but it currently only handles two cases:

- **Closures**: `foo::{{closure}}` → `foo` (handled)
- **Impl methods**: `foo::{{impl}}::method` → `foo::{{impl}}` (handled)

It does NOT handle these descendant types, which also fire per-item events:

- **Generic/lifetime params**: `foo::T`, `foo::'_`, `foo::'_[1]`
- **Anonymous consts**: `foo::_`, `foo::_[7]`
- **Opaque types**: `foo::{{opaque}}`, `foo::{{opaque}}[N]`
- **Constructors**: `foo::{{constructor}}`
- **Coroutine bodies**: `foo::{{coroutine}}`

Per-item events like `predicates_of`, `generics_of`, `type_of`, `def_span`
fire once per DefId including ALL descendants. A function `foo<T, U>` with 3
closures generates 6+ `predicates_of` events. The 3 closure events are
correctly aggregated to `foo`, but the generic param events (`foo::T`,
`foo::U`) are stored under keys that match nothing in the extracted symbol
map — they're silently lost.

This is why per-symbol attribution currently captures only ~3% of wall time.
The profile data is there; we're just failing to roll it up. Fixing this
should push per-symbol attribution from ~3% to 30-50%+ of wall time, making
`Symbol.event_times_ms` an accurate per-event breakdown of each symbol's
compilation cost.

**Fix**: Rewrite `normalize_frontend_path` to peel segments from the end of
the path until it reaches a segment that could be a real symbol name. Internal
segments to strip: `{{closure}}[N]`, `{{opaque}}[N]`, `{{constructor}}`,
`{{coroutine}}`, `_[N]` (anonymous consts), single-ident generic params (`T`,
`'_`), and anything after `{{impl}}[N]` (impl methods). These can nest and
combine (e.g., `foo::{{opaque}}::{{closure}}` must resolve to `foo`), so the
stripping must be iterative from the end.

## Testing

### Unit tests: `normalize_frontend_path` (`profile.rs`)

Existing tests cover closures and impl methods. Add tests for every new
descendant type being stripped:

- `my_crate::foo::T` → `my_crate::foo` (generic type param)
- `my_crate::foo::'_` → `my_crate::foo` (lifetime param)
- `my_crate::foo::'_[1]` → `my_crate::foo` (numbered lifetime param)
- `my_crate::foo::_` → `my_crate::foo` (anonymous const)
- `my_crate::foo::_[7]` → `my_crate::foo` (numbered anonymous const)
- `my_crate::foo::{{opaque}}` → `my_crate::foo` (opaque type)
- `my_crate::foo::{{opaque}}[2]` → `my_crate::foo` (numbered opaque)
- `my_crate::foo::{{constructor}}` → `my_crate::foo` (constructor)
- `my_crate::foo::{{coroutine}}` → `my_crate::foo` (coroutine body)
- Nesting: `my_crate::foo::{{opaque}}::{{closure}}` → `my_crate::foo`
- Impl + descendant: `my_crate::Type::{{impl}}::method::T` →
  `my_crate::Type::{{impl}}`

### Unit tests: no double-counting (`profile.rs`)

The existing `test_attributed_events_in_event_costs` test asserts that
symbol-attributed events appear in *both* `frontend_costs` and
`event_costs`. This must be updated to assert the opposite: attributed
events go ONLY to `frontend_costs` (→ `Symbol.event_times_ms`) and are
excluded from `TargetTimings.event_times_ms`. Add a dedicated test:

- Construct a `typeck` event with DefPath `my_crate::foo`. Call
  `record_event`.
- Assert `frontend_costs["my_crate::foo"]` has key `"typeck"` with
  correct self-time.
- Assert `TargetTimings.event_times_ms` does NOT contain `"typeck"`.
- Add a `mir_borrowck` event for the same symbol. Call `record_event`.
- Assert `frontend_costs["my_crate::foo"]` now has both `"typeck"` and
  `"mir_borrowck"` keys with correct individual self-times.
- Construct a `GenericActivity` event with no DefPath. Call
  `record_event`.
- Assert it appears in `TargetTimings.event_times_ms` but NOT in
  `frontend_costs`.

Also update `test_event_costs_complete_accounting` to reflect the new
invariant: `Σ(Symbol.event_times_ms) + Σ(TargetTimings.event_times_ms)
= total self-time`.

### Unit tests: two-variable regression (`tarjanize-cost`)

The existing `linear_fit` tests cover single-variable OLS. Add tests for
the new two-variable no-intercept regression `wall = a·attr + b·meta`:

- Perfect fit: `wall = 3·attr + 2·meta` with 5+ data points → R²=1.0,
  a=3.0, b=2.0.
- Zero metadata: all metadata values zero → degenerates to
  `wall = a·attr`, b irrelevant.
- Zero attribution: all attr values zero → degenerates to
  `wall = b·meta`, a irrelevant.
- Insufficient data: fewer than 3 points → returns None.
- Real-ish data: use approximate values from the validation table
  (a≈3.4, b≈2.8) with some noise, verify R²>0.99.

### Integration tests: descendant aggregation (`cost_extraction_tests.rs`)

Add a fixture crate that exercises all descendant types and verify that
the parent symbol's `event_times_ms` contains the expected event labels.
The fixture should contain:

- A generic function `fn foo<T: Debug>(x: T)` — events for `foo::T`
  must roll up to `foo`.
- An async function — events for the `{{opaque}}` future type must roll
  up to the function.
- A function with const blocks — anonymous const events must roll up.

Tests should assert on **specific event keys**, not just the sum:

- `foo.event_times_ms` contains `"typeck"` with value > 0 (body event).
- `foo.event_times_ms` contains `"predicates_of"` with value > 0
  (per-item event — fires for `foo`, `foo::T`, etc., all rolling up).
- The total `Σ(event_times_ms.values()) > 0`.
- No event key is `""` or contains `{{closure}}`, `{{opaque}}`, etc.
  (descendants should be aggregated away, not appear as keys).

This catches regressions where a rustc update adds/removes/renames
events — the test will fail if expected events disappear.

### Integration tests: schema changes (done)

Schema migration from `frontend_cost_ms` to `event_times_ms: HashMap<String,
f64>` is complete. All integration tests use the new field names.

### Cross-workspace validation

After all code changes, re-run extraction on the 13 validation
workspaces in `/home/debian/github/validation/` and verify:

1. Per-target `Σ(Symbol.event_times_ms)` +
   `Σ(TargetTimings.event_times_ms)` accounts for all profiled
   self-time (they're complementary, no double-counting).
2. The two-term regression `wall = a·attr + b·meta` achieves R² values
   comparable to the Python validation results in the table above.
3. No target has empty `event_times_ms` for all symbols (would indicate
   the aggregation fix didn't take effect).
4. Spot-check specific symbols in a few workspaces: verify that
   `event_times_ms` contains expected event labels (`typeck`,
   `mir_borrowck`, `predicates_of`, etc.) with non-zero values.

This validation is manual (run `cargo tarjanize` on each workspace and
inspect the output) — it doesn't need automated test infrastructure.

## Open Questions

1. **Metadata cost for split targets**: Using max of constituent metadata costs
   is a lower bound. Monitor in practice and add correction if needed.

2. **Non-linear trait system effects**: The two-term model is linear. If trait
   system costs scale super-linearly, we may need an interaction term. Watch
   for this in bevy and omicron validation.

3. **Pathological crates**: nexus-db-queries and helix-core are outliers due to
   measurement artifacts, not model failures. Need to understand the
   nexus-db-queries 91.8% gap (uninstrumented wall time during compilation).

4. **Is `mir_built` available during `after_analysis`?** Only relevant if we
   later want to extract structural properties like MIR statement count. The
   current event-based cost model doesn't need MIR access.

---

## Appendix A: The 24 Events

From cross-workspace stepwise regression (13 workspaces, 1719 targets), these
24 events are the union of: the universal pooled model features, features
appearing in 2+ per-workspace models, and top univariate predictors. Everything
else is noise.

Inspecting raw `.mm_profdata` files (helix-term 144M/5244 function bodies,
anyhow 6M/133 bodies) confirms which events carry `additional_data[0]` that
can be attributed to specific symbols via DefPaths. Each event below is marked:

- **DefPath**: Clean DefPath in `additional_data[0]`, directly attributable to
  a symbol after descendant aggregation. Descendants include closures, generic
  params, anonymous consts, opaque types, constructors, and coroutine bodies —
  all must be walked up to their nearest ancestor symbol. These are `Query` or
  `IncrementalResultHashing` event kinds.
- **Module**: `LocalModDefId(DefId(...))` arg — module-level, needs special
  parsing to extract the module DefId.
- **Type**: `PseudoCanonicalInput{...}` arg — contains type descriptions, not
  DefPaths. Cannot be attributed to a specific symbol.
- **None**: `GenericActivity` with no `additional_data` at all. Pure per-target
  overhead, not attributable to any symbol.
- **Trivial**: 0-2 invocations per crate. Not worth modeling.

### Per-item events (6) — scale with DefId count

| Event | What it measures | Attribution |
|-------|-----------------|-------------|
| `predicates_of` | Where-clauses and trait bounds per item | **DefPath** — e.g. `helix_term::application::Terminal` |
| `inferred_outlives_of` | Inferred lifetime bounds per ADT | **DefPath** — same clean path format |
| `generics_of` | Generic parameter lists per item | **DefPath** — includes upstream: `core::marker::Send` |
| `def_span` | Source span per DefId (trivial per-call) | **DefPath** — includes lifetime params: `foo::'_`, `foo::'_[1]` |
| `explicit_predicates_of` | Explicit where-clauses per item | **DefPath** — same as `predicates_of` |
| `type_of` | Type of each item | **DefPath** — same clean path format |

All 6 attributable via DefPath. These fire once per DefId, including
descendants (closures, opaque types, generic params, const blocks).

### Function body events (5) — scale with body size

| Event | What it measures | Attribution |
|-------|-----------------|-------------|
| `typeck` | Type-checks a function body | **DefPath** — fires per body including descendants: `foo::{{closure}}`, `foo::{{closure}}::{{closure}}` |
| `mir_borrowck` | Borrow checking per function | **DefPath** — same descendant pattern as `typeck` |
| `check_well_formed` | Item well-formedness validation | **DefPath** — fires for `{{use}}` items too, not just functions |
| `check_mod_privacy` | Privacy checking per module | **Module** — `LocalModDefId(DefId(0:4 ~ crate[hash]::module))`. Module-level, not per-item |
| `check_liveness` | Dead code / unused variable detection | **DefPath** — constants and associated items: `foo::SUFFIX`, `foo::{{constant}}` |

4 of 5 attributable via DefPath. `check_mod_privacy` is module-level only.
`typeck` and `mir_borrowck` are the dominant pair (~15-20% of wall time).

### Trait system events (3) — scale with trait impl landscape

| Event | What it measures | Attribution |
|-------|-----------------|-------------|
| `specialization_graph_of` | Specialization DAG for a trait | **DefPath** — trait name: `core::ops::drop::Drop`, `core::fmt::Display` |
| `impl_self_is_guaranteed_unsized` | Unsized self-type check per impl | **DefPath** — impl block: `foo::{{impl}}`, `foo::{{impl}}[1]` |
| `impl_super_outlives` | Supertrait lifetime bounds per impl | **DefPath** — impl block, includes upstream: `alloc::alloc::{{impl}}[1]` |

All 3 attributable via DefPath. Note: these reference both local and upstream
impls, meaning some cost is attributed to external symbols.

### Metadata decode events (4) — scale with dependency surface

| Event | What it measures | Attribution |
|-------|-----------------|-------------|
| `metadata_decode_entry_implementations_of_trait` | Reading trait impls from upstream | **None** — `GenericActivity`, no args |
| `metadata_decode_entry_impl_trait_header` | Reading impl headers from upstream | **None** — `GenericActivity`, no args |
| `metadata_decode_entry_mir_coroutine_witnesses` | Reading coroutine types from upstream | **None** — `GenericActivity`, no args |
| `metadata_decode_entry_generics_of` | Reading generic params from upstream | **None** — `GenericActivity`, no args |

All 4 unattributable. `GenericActivity` events with no `additional_data`.

### Type system events (2) — scale with type complexity

| Event | What it measures | Attribution |
|-------|-----------------|-------------|
| `needs_drop_raw` | Recursive check: does this type need Drop? | **Type** — `PseudoCanonicalInput{...}`, not a simple DefPath |
| `inhabited_predicate_type` | Is this type inhabited (non-empty)? | **DefPath** — clean type path: `helix_core::Position` |

Mixed: `inhabited_predicate_type` has DefPaths; `needs_drop_raw` has type
descriptions (filtered out by `normalize_frontend_path`). Only 2-4% of wall.

### Other events (4) — per-crate overhead / target type flags

| Event | What it measures | Attribution |
|-------|-----------------|-------------|
| `maybe_building_test_harness` | Whether building a test harness | **Trivial** — 1 invocation, no args |
| `is_mir_available` | MIR availability check | **Trivial** — often not present |
| `lib_features` | Library feature gate checking | **Trivial** — 2 invocations |
| `crate_inherent_impls` | Inherent impl enumeration | **Trivial** — 1-2 invocations |

All trivial. 0-2 invocations per crate with negligible cost.

### Attribution summary

| Category | Events | Attributable |
|----------|-------:|:-------------|
| Per-item | 6 | 6 of 6 (DefPath) |
| Function body | 5 | 4 of 5 (DefPath) |
| Trait system | 3 | 3 of 3 (DefPath) |
| Metadata decode | 4 | 0 of 4 (no args) |
| Type system | 2 | 1 of 2 (mixed) |
| Other | 4 | 0 of 4 (trivial) |
| **Total** | **24** | **14 of 24** |
