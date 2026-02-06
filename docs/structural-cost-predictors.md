# Structural Cost Predictors

What structural properties of Rust code drive compilation time, and how can we extract them?

## Context

Our per-symbol `frontend_cost_ms` attribution captures only ~3% of frontend wall time but predicts well for lib targets: R²=0.997 (tokio, n=6) and R²=0.822 (omicron, n=136). Individual rustc query self-times predict wall time even better (R²=0.995 for tokio, R²=0.913 for omicron). These queries are doing work proportional to structural properties of the code — properties we could measure directly during extraction.

This document maps the top compiler events to the code properties that drive them, then identifies what we could extract to build better cost predictors.

## Cost Breakdown by Category

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

*\*Omicron excludes nexus-db-queries/lib outlier — see cost-model-validation.md Section 7.6.*

**Workspace-dependent cost profiles**: Trait system costs dominate omicron (34%) but are modest in tokio (16%). Function body analysis is consistent across both (~27%). Metadata decode is similar in both (~15%). Per-item queries are proportionally larger in tokio (18% vs 8%) because tokio has more items relative to its trait complexity. Crate-level resolution is a non-trivial fixed-per-target cost (7-10%) that doesn't scale with code volume.

The priorities for structural extraction: trait impl count + dependency surface (for omicron-like workspaces) and function body size + DefId count (for tokio-like workspaces). Type system queries are too small to justify extraction complexity in either case.

## Cross-Workspace Stepwise Regression Results

Forward stepwise regression across 13 workspaces (2026-02-06), finding the minimum set of self-profile event labels that predict `frontend_wall_ms` per target. Each workspace model selects features greedily by adjusted R², stopping at adj R² > 0.999 or 10 features.

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

*\*Omicron excludes nexus-db-queries/lib outlier. Without exclusion: adj R²=0.952 with 10 features.*

**All 13 workspaces achieve adj R² > 0.999** with 1-10 features. The event self-time model is universal — a small number of rustc query self-times explain >99.9% of variance in compilation wall time across diverse Rust codebases.

### Feature frequency across models

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

The remaining 31 features appear in only one model (workspace-specific noise).

**Key findings**:

1. **Per-item events dominate** as top univariate predictors (10 of 13 workspaces). These events count items and naturally proxy for crate size. The `predicates_of` / `inferred_outlives_of` / `explicit_predicates_of` / `generics_of` family is the strongest universal signal.

2. **`metadata_decode_entry_impl_trait_header`** is the most universal secondary feature (6 models), capturing the cost of reading upstream trait impl metadata. This confirms that dependency surface area is a consistent secondary cost driver.

3. **Bevy is the hardest to fit** — needs all 10 features and has the weakest top univariate predictor (R²=0.854). Its trait system dominates at 44.3% of wall time, reflecting heavy generics/ECS usage. No single per-item event captures the diverse cost profile.

4. **Omicron's outlier is real** — excluding `nexus-db-queries/lib` (26% of total wall time, a massive generated database query layer) brings R² from 0.952 to 0.999. This crate has fundamentally different cost characteristics from normal Rust code.

5. **The long tail of singleton features is noise**, not signal. Each workspace picks up a few events that happen to correlate with its specific crate size distribution but don't generalize. A universal cost model should use only the recurring features.

### Universal pooled model (all workspaces combined)

Forward stepwise regression on all 1719 targets pooled together (419 candidate features present in >= 10% of targets). This tests whether a **single model** can predict wall time across all workspaces simultaneously.

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

**Key differences from per-workspace models:**

1. **Different features win universally.** `needs_drop_raw` (type_system, univariate R²=0.783) is the best pooled predictor despite never winning per-workspace. It scales uniformly with code volume across all codebases. Per-item events like `predicates_of` (the per-workspace champion, univariate R²=0.995 in tokio) drop to R²=0.66 when pooled because their absolute scale shifts between workspaces.

2. **The model draws from all categories**: type_system, profiler_overhead, fn_body, metadata, trait_system, other, per_item. No single category dominates — the universal model needs cross-category coverage to handle diverse codebases.

3. **`self_profile_alloc_query_strings`** (profiler overhead) is a strong universal predictor because it's essentially "total profiler events recorded" — a proxy for total compilation work. Not useful for production cost estimation since it's measurement overhead, but confirms that total event count is a strong structural signal.

4. **Weakest fits** are sqlx (0.969, n=9) and opte (0.989, n=21) — small sample sizes. Every workspace with n>=30 achieves R²>0.993.

5. **Per-workspace models still win** — every workspace achieves adj R²>0.999 with its own model (1-10 features), while the universal model ranges from 0.969 to 0.999. The gap is largest for small workspaces where the universal model's coefficients don't fit the local distribution as tightly.

## Event Categories

### Per-Item Queries

These run once per `DefId` (item definition) in the crate. Their total cost scales linearly with item count, modulated by per-item complexity. In tokio they're 17.9% of wall; in omicron 8.2%.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `predicates_of` | Where-clauses and trait bounds for an item | Item count × bounds per item |
| `generics_of` | Generic parameter list (type, lifetime, const params) | Item count × params per item |
| `type_of` | The type of an item (return type, field type, etc.) | Item count |
| `explicit_predicates_of` | Explicit where-clauses (vs inferred) | Same as `predicates_of` |
| `param_env` | All bounds in scope, normalized | Item count × inherited bound depth (supertraits) |
| `constness` | Whether an item is const | Pure item count (trivial per-call) |
| `codegen_fn_attrs` | `#[inline]`, `#[cold]`, `#[target_feature]`, etc. | Function count |
| `intrinsic_raw` | Whether a function is a compiler intrinsic | Function count |
| `lookup_stability` | `#[stable]` / `#[unstable]` attributes | Item count |
| `inferred_outlives_of` | Inferred lifetime outlives bounds for ADTs | Struct/enum count × field count |
| `associated_item_def_ids` | Lists associated items of a trait or impl | Trait/impl count × items per trait |
| `associated_item` | Looks up a single associated item | Trait/impl item references |
| `associated_items` | All associated items for an impl/trait | Trait/impl count |
| `inherent_impls` | Inherent impl blocks for a type | Type count |
| `defaultness` | Whether an impl item is `default` | Impl item count |
| `fn_sig` | Function signature | Function count |
| `adt_def` | ADT definition (struct/enum layout) | Struct/enum count |
| `def_span` | Source span for a DefId | Pure item count (trivial) |
| `def_kind` | What kind of item a DefId is (fn, struct, etc.) | Pure item count (trivial) |
| `def_ident_span` | Span of an item's identifier | Pure item count (trivial) |
| `is_doc_hidden` | Whether an item has `#[doc(hidden)]` | Item count |
| `lookup_deprecation_entry` | Deprecation attributes | Item count |

**Why these predict so well**: They all scale with "number of things in the crate." A crate with 500 items takes ~5x longer on these queries than one with 100 items. Since compilation time is dominated by per-item work, item count is an almost perfect proxy for wall time.

### Type System Queries (2-4% of wall)

Called per distinct type encountered during compilation. Recursive through type structure.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `is_copy_raw` | Does this type implement `Copy`? | Distinct types × field depth. Recursive check through all fields. |
| `needs_drop_raw` | Does this type need `Drop`? | Same recursive structure. |
| `dropck_outlives` | Drop check lifetime bounds | Types with Drop impls × field nesting depth |
| `layout_of` | Memory layout (size, alignment, field offsets) | Type complexity and nesting |
| `adt_sizedness_constraint` | Sizedness requirements for ADTs | ADT count × field depth |
| `is_sized_raw` | Does this type implement `Sized`? | Same recursive structure as `is_copy_raw` |
| `adt_dtorck_constraint` | Drop-check constraints for ADTs | ADTs with Drop × field nesting |
| `inhabited_predicate_type` | Is this type inhabited (non-empty)? | Type nesting depth |
| `inhabited_predicate_adt` | Is this ADT inhabited? | ADT field count × variant count |
| `try_normalize_generic_arg_after_erasing_regions` | Normalize type after erasing regions | Type complexity in generic contexts |

**Not worth modeling as a category**: These queries are excellent *predictors* of wall time (R²=0.90 for `needs_drop_raw` in omicron, R²=0.99 in tokio) because they scale with code volume, but they're only **2-4% of actual wall time**. Notably, `adt_sizedness_constraint` and `needs_drop_raw` are among omicron's top-5 predictors despite being tiny — they're essentially counting items. Not worth the extraction complexity for such a small cost share, but useful as proxy signals.

### Function Body Analysis (26-27% of wall)

Per function body, cost scales with body size/complexity. This is the largest category in both workspaces when all sub-events are included.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `typeck` | Type-checks a function body | Function count × body complexity (expressions, closures, match arms) |
| `mir_borrowck` | Borrow checking (query-level) | Function count × CFG complexity (borrows, lifetimes, control flow) |
| `MIR_borrow_checking` | Borrow checking (activity span) | Same — wraps `mir_borrowck` with additional bookkeeping |
| `mir_built` | Builds MIR from THIR | Function count × expression/statement count |
| `evaluate_obligation` | Evaluates a single trait obligation | Total trait constraints encountered during typeck. `where T: A + B + C` = 3+ evaluations. Closures and generics multiply this. |
| `type_op_prove_predicate` | Prove a predicate in a typing context | Predicate count × complexity during borrow checking |
| `check_well_formed` | Validates item well-formedness | Item count × bound complexity |
| `check_mod_privacy` | Privacy/visibility checking per module | Module count × item count |
| `check_liveness` | Dead code / unused variable detection | Function count × variable count |
| `check_match` | Match exhaustiveness and usefulness | Function count × match arms × pattern complexity |
| `check_unsafety` | Validates unsafe block usage | Function count (fast for safe fns, slow for unsafe) |
| `check_mod_unstable_api_usage` | Detects use of unstable APIs | Module item count × API surface |
| `thir_body` | Builds THIR from HIR for a function body | Function count × expression count |
| `mir_promoted` | Extracts promoted constants from MIR | Function count × const expression count |
| `mir_drops_elaborated_and_const_checked` | Drop elaboration + const checking | Function count × drop complexity |
| `mir_for_ctfe` | MIR for compile-time function evaluation | Const fn count × body size |
| `mir_const_qualif` | Const qualification checking | Const/static count |
| `optimized_mir` | Final optimized MIR | Function count × MIR size |
| `mir_pass_*` | MIR optimization passes (~20 individual passes) | Function count × MIR size (basic blocks × statements) |
| `normalize_canonicalized_projection` | Normalize associated type projections | Generic code volume × associated type usage |
| `type_op_ascribe_user_type` | Check user type annotations in MIR | Annotation count in function bodies |
| `type_op_normalize_fn_sig` | Normalize function signature types | Function count × generic complexity |
| `type_op_normalize_clause` | Normalize where-clause predicates | Predicate count in generic contexts |
| `type_op_normalize_ty` | Normalize a type during type checking | Type complexity in generic contexts |
| `implied_outlives_bounds` | Compute implied lifetime outlives bounds | Generic function count × lifetime params |
| `region_scope_tree` | Build scope tree for lifetime analysis | Function count × nesting depth |
| `check_transmutes` | Validate `transmute` calls | Transmute call count |
| `has_ffi_unwind_calls` | Check for FFI functions that can unwind | Function count (trivial per-call) |
| `check_tail_calls` | Validate tail call usage | Function count (trivial per-call) |
| `opaque_types_defined_by` | Which opaque types a function defines | Function count × `impl Trait` usage |
| `upvars_mentioned` | Captured variables for closures | Closure count × captured variable count |
| `coroutine_kind` | What kind of coroutine (async, gen, etc.) | Async/generator function count |
| `mir_coroutine_witnesses` | Types witnessed by a coroutine | Coroutine count × yield point count |
| `coroutine_hidden_types` | Hidden types inside async/generator bodies | Coroutine count |
| `check_coroutine_obligations` | Validate coroutine trait obligations | Coroutine count × bound complexity |
| `resolve_instance_raw` | Resolve a function call to a concrete impl | Call site count × monomorphization depth |
| `thir_abstract_const` | Abstract const evaluation on THIR | Const generic expression count |
| `used_trait_imports` | Which trait imports are actually used | Function count × trait method calls |
| `should_inherit_track_caller` | Whether to propagate `#[track_caller]` | Function count (trivial per-call) |
| `nested_bodies_within` | Find nested function bodies (closures, etc.) | Function count × closure nesting |

**What drives these**: Two factors multiply: (1) number of function bodies, and (2) size of each body. A crate with many small functions and a crate with few large functions can have similar total cost. MIR passes are particularly sensitive to total MIR size (basic blocks × statements across all functions). The `mir_pass_*` family alone contains ~20 individual passes, each iterating over the full MIR; the largest are `state_transform`, `elaborate_drops`, and `promote_temps`.

### Trait System (16-34% of wall)

These scale with the trait implementation landscape, including upstream crates. The most workspace-dependent category: 34% in omicron (heavy serde/diesel usage) vs 16% in tokio.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `implementations_of_trait` | All impls of a given trait across all crates | Traits used × total impls in the ecosystem. Looking up `Iterator` impls searches ALL crates. |
| `specialization_graph_of` | Specialization DAG for a trait | Traits with multiple overlapping impls |
| `impl_trait_header` | Trait and self-type for an impl block | Total visible trait impls (local + upstream) |
| `impl_parent` | Parent trait of a specializing impl | Trait specialization depth |
| `trait_impls_of` | All impls of a trait within a crate | Trait count × local impl density |
| `compare_impl_item` | Validate impl items match trait definition | Trait impl count × items per impl |
| `coherent_trait` | Coherence checking for a trait | Trait count × overlapping impl potential |
| `crate_incoherent_impls` | Incoherent impls (inherent impls on foreign types) | Foreign type usage count |
| `incoherent_impls` | Find incoherent impls for a type | Same as above, per-type query |
| `collect_return_position_impl_trait_in_trait_tys` | RPITIT type collection | Trait methods returning `impl Trait` |
| `check_type_wf` | Well-formedness of types in trait context | Type count × trait bound complexity |
| `impl_super_outlives` | Lifetime outlives requirements from super traits | Impl count × supertrait depth |
| `impl_self_is_guaranteed_unsized` | Check if impl self type is unsized | Impl count (trivial per-call) |
| `enforce_impl_non_lifetime_params_are_constrained` | Validate generic params are constrained | Impl count × generic param count |
| `orphan_check_impl` | Orphan rule checking | Trait impl count |
| `impl_item_implementor_ids` | Map impl items to their implementors | Impl count |

**What drives these**: Partly the crate's own code (how many trait impls it defines), but largely the *dependency ecosystem*. A crate that uses traits with thousands of impls across its dependencies (e.g., `serde::Serialize`) pays for searching all of them. This cost is inherent to the dependency graph, not just the local code. The top four events (`implementations_of_trait`, `specialization_graph_of`, `impl_trait_header`, `impl_parent`) account for 80% of the category in omicron.

### Metadata Decoding (14-15% of wall)

Reading upstream crate `.rmeta` files. This is the cost of having dependencies. Consistent across both workspaces at ~15%.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `metadata_decode_entry_impl_trait_header` | Trait impl headers from upstream | Upstream trait impl count |
| `metadata_decode_entry_implementations_of_trait` | Finding impls across upstream crates | Traits used × upstream impl count |
| `metadata_decode_entry_impl_parent` | Parent trait info from upstream | Upstream specialization depth |
| `metadata_decode_entry_module_children` | Items exported by upstream modules | Upstream crate count × exported items per crate |
| `metadata_decode_entry_type_of` | Type info from upstream items | Upstream items referenced |
| `metadata_decode_entry_explicit_predicates_of` | Predicate info from upstream items | Upstream items referenced × bounds per item |
| `metadata_decode_entry_associated_item` | Associated item info from upstream | Upstream trait/impl item references |
| `metadata_decode_entry_def_kind` | DefKind from upstream items | Upstream items referenced |
| `metadata_decode_entry_generics_of` | Generic params from upstream items | Upstream items referenced |
| `metadata_decode_entry_inferred_outlives_of` | Outlives bounds from upstream | Upstream ADT references |
| `metadata_decode_entry_adt_def` | ADT definitions from upstream | Upstream struct/enum references |
| `metadata_decode_entry_defaultness` | Defaultness from upstream impl items | Upstream impl item references |
| `metadata_decode_entry_attrs_for_def` | Attributes from upstream items | Upstream items referenced |
| `metadata_decode_entry_is_doc_hidden` | Doc-hidden status from upstream | Upstream items referenced |
| `metadata_decode_entry_crate_incoherent_impls` | Incoherent impls from upstream | Foreign type usage |
| `metadata_decode_entry_*` (others) | Various per-item queries on upstream data | Upstream item references |
| `metadata_register_crate` | Initial metadata load for a dependency | One-time per upstream crate |

**What drives these**: Two factors: (1) number of direct + transitive dependencies, and (2) how much of each dependency's API surface is touched. A crate with 50 deps spends more time decoding metadata than one with 5, even if its own code is identical. Note that the `metadata_decode_entry_*` events mirror the per-item and trait system queries — they're the same queries, but answered by reading `.rmeta` files instead of computing locally.

### Crate-Level & Resolution (7-10% of wall)

Per-crate overhead that runs once regardless of code volume: macro expansion, name resolution, HIR construction, linting, and metadata generation. These don't scale with item count — they're fixed costs for each compilation target.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `expand_crate` | Macro expansion for the entire crate | Macro invocation count + proc macro count |
| `expand_proc_macro` | Execute a single proc macro | Proc macro count × input size |
| `late_resolve_crate` | Name resolution (post-expansion) | Item count × import complexity |
| `hir_crate` | Build HIR for the entire crate | Crate size (one-time) |
| `hir_crate_items` | Enumerate all HIR items | Item count (one-time scan) |
| `generate_crate_metadata` | Write `.rmeta` for downstream crates | Exported item count |
| `lint_mod` | Run lint passes per module | Module count × item count |
| `check_mod_attrs` | Validate attributes per module | Module count × attribute count |
| `shallow_lint_levels_on` | Compute lint levels | Item count |
| `lints_that_dont_need_to_run` | Filter out irrelevant lints | Lint configuration complexity |
| `effective_visibilities` | Compute effective visibility for all items | Item count × module nesting |
| `visible_parent_map` | Map items to their visible parent module | Module tree depth |
| `module_children` | Children of each module | Module count × items per module |
| `get_lang_items` | Locate compiler-known lang items | One-time per crate |
| `crate_variances` | Compute variance for all type params | Generic item count |
| `type_check_crate` | Top-level type checking coordination | One-time per crate |
| `local_def_id_to_hir_id` | DefId → HIR ID mapping | Item count |
| `opt_hir_owner_nodes` | HIR node lookup | Item count |
| `hir_attr_map` | Attribute map for HIR | Item count |
| `attrs_for_def` | Attributes for a specific DefId | Item count |
| `sanitizer_settings_for` | Sanitizer configuration per item | Item count (trivial) |
| `resolve_bound_vars` | Resolve lifetime/type vars in bounds | Generic item count |
| `assumed_wf_types` | Types assumed well-formed in context | Item count |
| `late_bound_vars_map` | Late-bound lifetime variables | Generic function count |
| `named_variable_map` | Named variable resolution | Function count |
| `method_autoderef_steps` | Auto-deref steps for method calls | Method call count |
| `typing_env_normalized_for_post_analysis` | Normalized typing env | Item count |
| `trivial_const` | Trivial const evaluation | Const expression count |
| `inherited_align` | Inherited alignment for repr | ADT count |
| `adt_destructor` | Destructor for an ADT | ADT count |
| `live_symbols_and_ignored_derived_traits` | Dead code analysis | Item count |
| `early_lint_checks` | Early-phase linting | One-time per crate |
| `drop_ast` | Free the AST after lowering | One-time per crate |
| `variances_of` | Variance of a specific item | Generic item count |
| `visibility` | Visibility of an item | Item count |

**What drives these**: Mostly one-time-per-crate overhead plus some per-item scanning. Unlike per-item queries (which run per DefId) or function body analysis (which scales with body size), these are closer to fixed costs. When splitting a crate, both halves pay this overhead — it doesn't divide proportionally.

### Profiler Overhead (3-4% of wall)

Cost of the self-profiling infrastructure itself.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `self_profile_alloc_query_strings` | Allocate strings for profile event labels | Total event count × label length |

**Not real compilation cost**: This is purely measurement overhead from `-Zself-profile`. It scales with the number of profiled events (i.e., with total compilation work), which is why it's a strong wall-time predictor (R²=0.88-0.99). Should be subtracted when estimating actual compilation time, since production builds don't have this overhead.

### Per-Crate Fixed Cost (1% of wall)

Tiny per-dependency queries that fire once per upstream crate. Collectively ~1% of wall.

| Event | What it computes | Cost driver |
|-------|-----------------|-------------|
| `defined_lang_items` | Lang items defined by an upstream crate | One per upstream crate |
| `externally_implementable_items` | Externally implementable items | One per upstream crate |
| `native_libraries` | Native library requirements | One per upstream crate |
| `crate_name` | Name of an upstream crate | One per upstream crate |
| `all_diagnostic_items` | Diagnostic items across all crates | One-time |
| `is_private_dep` | Whether a dep is private | One per upstream crate |
| `is_profiler_runtime` | Whether a crate is the profiler runtime | One per upstream crate |
| `used_crate_source` | Source location of a crate | One per upstream crate |
| `dep_kind` | Dependency kind (normal, dev, etc.) | One per upstream crate |
| `is_no_builtins` | Whether `#[no_builtins]` is set | One per upstream crate |
| `is_compiler_builtins` | Whether this is the compiler builtins crate | One per upstream crate |
| `missing_extern_crate_item` | Check for missing extern crate | One per upstream crate |
| `missing_lang_items` | Missing lang items | One per upstream crate |
| `entry_fn` | The `main` function entry point | One per crate |
| `proc_macro_decls_static` | Proc macro declarations | One per crate |

**Negligible and non-predictive**: These fire a fixed number of times per upstream crate with trivial per-call cost. R² is near zero because they don't vary with crate complexity. Not worth modeling.

## Structural Properties That Matter

The events above are driven by a small number of countable structural properties:

### 1. Total Item Count

The single most important driver. `predicates_of`, `generics_of`, `constness`, `codegen_fn_attrs`, `lookup_stability`, and many others run once per DefId. This is why they're such perfect wall-time predictors — they're essentially counting items.

**What counts as an item**: Everything with a `DefId` — not just module-level definitions. A single function generates DefIds for: itself, each generic parameter, each closure/async block inside it, each anonymous const (`[T; N]`), and opaque types (`impl Trait`). A function with 3 closures and 2 generic params produces 6+ DefIds.

**Our symbol count undercounts**: We collapse associated items into their parent and don't count closures, generic params, or anonymous consts at all. The true driver is the **total DefId count**, which can be much larger than the symbol count. We should track the DefId count per symbol — the symbol's own DefId plus all descendant DefIds (closures, generic params, const blocks, etc.). This lets us predict how per-item query costs redistribute when splitting.

### 2. Predicate / Bound Count

Drives `predicates_of`, `explicit_predicates_of`, `param_env`, `evaluate_obligation`. Not just item count, but the sum of trait bounds across all items.

```rust
// 0 predicates
fn foo() {}

// 3 predicates
fn bar<T: Clone + Debug + Send>(x: T) {}

// 6 predicates (2 type params × 3 bounds each)
fn baz<T: Clone + Debug + Send, U: From<T> + Into<String> + Default>() {}
```

**How to measure**: `sum over all items of tcx.predicates_of(def_id).predicates.len()`

### 3. Generic Parameter Count

Drives `generics_of` directly, and indirectly `predicates_of` (more generics usually means more bounds).

**How to measure**: `sum over all items of tcx.generics_of(def_id).own_params.len()`

### 4. Function Body Size

Drives `typeck`, `mir_built`, `mir_borrowck`, all `mir_pass_*`. Two possible measurements:

- **THIR expression count**: We already walk THIR for dependency extraction. Could count expressions and statements during the walk.
- **MIR basic block count**: `tcx.optimized_mir(def_id).basic_blocks.len()` for each function. More directly measures what MIR passes iterate over.
- **MIR statement count**: Total statements across all basic blocks. Even more precise.

### 5. Trait Impl Count (Local)

Drives `impl_trait_header`, `trait_impls_of`, `specialization_graph_of`. Count of `impl Trait for Type` blocks in the crate.

**We already have this partially**: We extract impl blocks as symbols. Could separate trait impls from inherent impls and count them.

### 6. Upstream Dependency Surface

Drives all `metadata_decode_*` events. This is the one property that's NOT about the crate's own code — it's about the dependency graph.

Possible measurements:
- Number of direct dependencies (we have this in `target.dependencies`)
- Total transitive dependency count
- Total items across all upstream crates (harder to measure)

## What We Could Extract

We already call most of the relevant `tcx` queries in `extract.rs` for dependency analysis. We just don't count the results. Adding structural counters would be cheap.

### Per-Symbol (augment `Symbol` struct)

| Property | How to get it | Which events it predicts |
|----------|--------------|------------------------|
| **DefId count** | Count descendant DefIds (closures, generic params, const blocks, opaque types) | All per-item queries — this is the fundamental unit |
| **Predicate count** | `tcx.predicates_of(def_id).predicates.len()` | `predicates_of`, `param_env`, `evaluate_obligation` |
| **THIR expression count** | Count during existing THIR walk (free) | `typeck`, `mir_built` |
| **MIR statement count** | `tcx.mir_built(def_id)` → sum statements across basic blocks | All `mir_pass_*`, `mir_borrowck` |

The **DefId count** is the most important new field. It directly measures the "weight" of a symbol in terms of per-item query invocations. A function with 5 closures and 3 generic params costs ~9x more in per-item queries than a simple non-generic function with no closures. Generic param count and associated item count are subsets of DefId count and not worth extracting separately.

### Per-Target (augment `TargetTimings` or add new struct)

| Property | How to get it | Which events it predicts |
|----------|--------------|------------------------|
| Total DefId count | Sum of per-symbol DefId counts (or `tcx.hir_crate_items()` directly) | All per-item queries |
| Total predicate count | Sum of per-item predicate counts | `predicates_of` family |
| Total MIR size | Sum of basic blocks across all fns | `mir_pass_*` family |
| Trait impl count | Count DefKind::Impl { of_trait: true } | Trait system events |
| Dependency count | Already have `target.dependencies.len()` | `metadata_decode_*` |

### Validation Approach

Once extracted, we can compute R² of these structural features against `frontend_wall_ms` on both tokio and omicron:

```python
# Hypothesis: structural counts predict as well as event self-times
# Tokio: predicates_of R²=0.995, per-symbol R²=0.814
# Omicron: check_liveness R²=0.913, per-symbol R²=0.375 (all) / 0.822 (lib)

r2_predicate_count = corr(walls, predicate_counts) ** 2       # should be ~0.99?
r2_sym_preds       = corr(walls, per_symbol_predicate_sum) ** 2  # hopefully > 0.9?

# Multi-feature model for omicron (where no single feature dominates)
r2_composite = regression(walls, [defid_sums, mir_sums, trait_impl_counts, dep_counts])
```

Because tokio and omicron have such different cost profiles (function body vs trait system), validation must succeed on both. If structural counts predict as well as event self-times in both workspaces, we can use them for cost estimation without needing profile data at all.

## Improving Profile-Based Attribution

### The Problem

Per-symbol `frontend_cost_ms` currently captures only ~3% of wall time (R²=0.81). The gap isn't because the profile data is missing — it's because we discard most of it. Query events like `predicates_of` carry the DefPath of the item they were called for (in `additional_data[0]`), but we only match DefPaths that correspond to extracted symbols.

A `predicates_of` call for closure `my_crate::foo::bar::{closure#0}` has a perfectly good DefPath, but we can't match it because we don't extract closures as symbols. Same for generic params, anonymous consts, async blocks, etc.

### Descendant DefPath Attribution

The fix: when a DefPath doesn't match any symbol directly, walk it up to its nearest ancestor symbol. `my_crate::foo::bar::{closure#0}` → `my_crate::foo::bar`. This attributes the closure's compilation cost to the function that contains it, which is correct for crate-splitting purposes (you can't move a closure without its parent function).

This is a change to `profile.rs` only — no schema changes needed. The existing `frontend_cost_ms` field on `Symbol` would simply capture a much larger fraction of wall time.

**What currently gets dropped** (events with DefPaths we can't match):
- Closures: `foo::bar::{closure#0}`, `foo::bar::{closure#0}::{closure#0}` (nested)
- Generic params: `foo::bar::T`, `foo::bar::N`
- Anonymous consts: `foo::bar::{constant#0}` (array lengths, const generic args)
- Opaque types: `foo::bar::{opaque#0}` (impl Trait return types)
- Async blocks: `foo::bar::{async_block#0}`

**What we'd gain**: All per-item query costs (`predicates_of`, `generics_of`, `constness`, etc.) for these descendant DefIds would be attributed to their parent symbol. Since per-item queries are the dominant cost category, this could bring attribution from ~3% to potentially 30-50%+ of wall time.

### Combined Approach

We want both improved profile attribution AND structural extraction. They serve different purposes and validate each other.

**Step 1: Improve profile attribution (descendant DefPath mapping)**

Walk unmatched DefPaths up to their nearest ancestor symbol. `typeck(foo::{closure#0})` → attribute to symbol `foo`. This is a `profile.rs`-only change that makes `frontend_cost_ms` much more accurate — the "measured cost" from the compiler.

Current state: per-symbol attribution for lib targets varies widely. Tokio/lib captures 38.7% of wall time; omicron's top libs range from 53.8% (nexus-db-schema) down to 7.7% (nexus-db-model). The ~3% average across all targets is dragged down by test targets. Descendant mapping should push all of these higher.

**Step 2: Extract structural counts per symbol**

During extraction, collect per-symbol structural features that predict compilation cost:

| Feature | How to collect | What it predicts | Cost to collect |
|---------|---------------|-----------------|----------------|
| DefId count | Count descendant DefIds (closures, generic params, const blocks) | Per-item query costs (8-18% of wall) | Needs `tcx.hir_body_owners()` or similar traversal |
| THIR expression count | Count during existing THIR walk in `walk_thir_expr` | `typeck`, `mir_built` cost | Free — already walking THIR for dependency extraction |
| MIR statement count | `tcx.mir_built(def_id)` → sum statements across all basic blocks | `mir_pass_*`, `mir_borrowck` cost (27% of wall) | One query per function body |
| Predicate count | `tcx.predicates_of(def_id).predicates.len()` | `predicates_of`, `evaluate_obligation` cost | One query per item |

These are new fields on `Symbol`. They travel with the symbol through condense and cost analysis.

**Note**: Use `mir_built` (unoptimized MIR), not `optimized_mir`. During `cargo check`, optimized MIR is not computed since codegen doesn't run. Unoptimized MIR is already available after borrow checking and has similar basic block / statement counts — MIR optimization doesn't dramatically change the size.

**Descendant aggregation**: Closures, async blocks, and nested bodies are separate DefIds with their own THIR and MIR. A function with 3 closures triggers `typeck` 4 times, `mir_borrowck` 4 times, etc. Structural counts must be summed across the symbol and all its descendants to predict the true function body cost. Our THIR walk in `extract.rs` currently only walks the top-level body. To get accurate counts, extraction needs to also query `thir_body` / `mir_built` for each descendant DefId and accumulate the totals onto the parent symbol. As a simpler first approximation, DefId count × top-level body size may suffice — validate this against profiled costs before adding the full descendant traversal.

**Step 3: Validate structural counts against profiled costs**

For profiled crates, check that per-symbol structural counts predict per-symbol `frontend_cost_ms`:

```python
# Per-symbol validation: do structural features explain measured cost?
r2 = corr(per_symbol_frontend_cost, per_symbol_mir_blocks) ** 2

# Per-target validation: do structural sums predict wall time?
r2 = corr(target_wall_times, target_mir_block_sums) ** 2
```

The two approaches validate each other: if a symbol has 50 DefIds and 200 MIR blocks, we'd expect high `frontend_cost_ms`. If they diverge, the symbol triggers disproportionate costs (e.g., expensive trait resolution) that the structural model needs to account for.

**Step 4: Predict costs for synthetic crates**

After splitting, we have new crates that have never been compiled. Each symbol carries its structural counts from the original extraction. We estimate the new crate's cost as:

```
estimated_wall_ms = f(sum(defid_counts), sum(mir_blocks), sum(predicate_counts), dep_count)
```

Where `f` is a regression model fitted against the profiled data from Step 3. This is the key payoff: structural counts are the *transferable* cost model that lets us predict compilation time for crates that don't exist yet.

## Open Questions

1. **Do `mir_pass_*` events carry DefPaths?** These are 27.9% of wall time for tokio/lib. If they're `GenericActivity` events without DefPaths, we can't attribute them to symbols via the profile — we'd rely entirely on structural MIR counts. Need to inspect raw profile data to check.

2. **Which structural features are additive?** For crate splitting, we need `cost(A ∪ B) ≈ cost(A) + cost(B)`. Per-item queries should be perfectly additive — moving 100 DefIds to a new crate moves exactly 100 query invocations. Trait system queries might not be — moving an impl to a different crate changes what's visible. This is especially critical for omicron where trait system is 34% of wall time.

3. **How do we handle metadata decode costs?** These depend on the dependency graph, not the crate's own code. When splitting a crate, both halves still need to decode the same upstream metadata. This is a fixed cost that doesn't split — it's per-target overhead proportional to dependency count. At ~15% of wall in both workspaces, this is a significant non-splittable cost.

4. **What about `evaluate_obligation`?** Called many times per function body, not once per item. Depends on how many trait bounds are exercised during type checking. Might need a compound proxy: `predicate_count × body_size`.

5. **How much does descendant DefPath mapping improve attribution?** tokio/lib is already at 38.7%. Omicron's top libs vary widely: nexus-db-schema at 53.8%, omicron-nexus at 30.5%, nexus-db-model at 7.7%. How much does parent-walking improve the low ones?

6. **What's happening with nexus-db-queries/lib?** This target has 1.3M ms wall time but only 109K ms in event costs (8.2%). 92% of its wall time is invisible to the profiler. This dwarfs all other targets (6.5x the next largest) and destroys correlation analysis when included. Needs dedicated investigation — possible profiler limitation, extreme trait resolution, or parallel frontend threads not captured.

7. **Do we need workspace-specific models?** Function body analysis (~27%) and metadata decode (~15%) are consistent across workspaces. The main divergence is trait system (34% omicron vs 16% tokio) and per-item queries (18% tokio vs 8% omicron). A composite model `f(defid_count, mir_blocks, trait_impl_count, dep_count)` should handle both, but the trait system component may need workspace-specific weighting since it depends on the dependency ecosystem, not just local code.

8. **What regression model for `f()`?** Linear might suffice if costs are roughly additive. But metadata decode is a fixed overhead per target, and trait system costs scale non-linearly with ecosystem size. Might need `f = fixed_overhead + linear(structural_counts)`.
