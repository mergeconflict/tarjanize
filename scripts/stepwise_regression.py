#!/usr/bin/env python3
"""Forward stepwise regression to find the best event-cost predictors of wall time.

Reads a tarjanize symbol_graph JSON (from `cargo tarjanize --profile`) and finds
the combination of self-profile event labels that best predicts frontend_wall_ms
using multivariate linear regression with forward feature selection.

Usage:
    python3 stepwise_regression.py <symbol_graph.json> [--exclude TARGET] [--max-features N]

Examples:
    # Basic usage
    python3 stepwise_regression.py ~/validation/data/tokio/symbol_graph_check.json

    # Exclude an outlier target
    python3 stepwise_regression.py ~/validation/data/omicron/symbol_graph_check.json \
        --exclude nexus-db-queries/lib

    # Limit to top 5 features
    python3 stepwise_regression.py data.json --max-features 5

Algorithm:
    Forward stepwise selection with OLS (ordinary least squares). At each step,
    tries adding every remaining event label as a feature, keeps the one that
    maximizes adjusted R². Stops when adjusted R² > 0.999 or max features reached.

    OLS is solved via Gaussian elimination (no numpy dependency). The intercept
    is always included. Adjusted R² penalizes model complexity:
        adj_R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)

    Only event labels present in >= 10% of targets are considered as candidate
    features, to avoid overfitting on rare events.

Output:
    - Forward stepwise table: features added in order of importance
    - Category breakdown: what % of wall each event category covers
    - Per-feature univariate R² for comparison
"""

import json
import math
import sys


def load_targets(path, exclude=None):
    """Load targets with profiling data from a symbol_graph JSON."""
    exclude = exclude or set()
    with open(path) as f:
        data = json.load(f)

    rows = []
    for pkg_name, pkg in data["packages"].items():
        for tgt_name, tgt in pkg.get("targets", {}).items():
            key = f"{pkg_name}/{tgt_name}"
            if key in exclude:
                continue
            timings = tgt.get("timings", {})
            fe_wall = timings.get("frontend_wall_ms", 0)
            if fe_wall <= 0:
                continue
            ec = timings.get("event_costs", {})
            rows.append({"key": key, "wall": fe_wall, "ec": ec})
    return rows


def pearson(xs, ys):
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n * sx * sy)


def ols_multi(X, y):
    """OLS regression: y = intercept + X @ beta.

    Returns dict with r2, r2_adj, beta, intercept. Returns None if singular.
    X is list of lists (n x p), y is list (n,).
    """
    n = len(y)
    p = len(X[0])

    # Add intercept column.
    Xa = [[1.0] + row for row in X]
    pa = p + 1

    # Compute X^T X.
    XtX = [[0.0] * pa for _ in range(pa)]
    for i in range(pa):
        for j in range(pa):
            for k in range(n):
                XtX[i][j] += Xa[k][i] * Xa[k][j]

    # Compute X^T y.
    Xty = [0.0] * pa
    for i in range(pa):
        for k in range(n):
            Xty[i] += Xa[k][i] * y[k]

    # Solve via Gaussian elimination with partial pivoting.
    aug = [XtX[i][:] + [Xty[i]] for i in range(pa)]
    for col in range(pa):
        max_row = col
        for row in range(col + 1, pa):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-12:
            return None  # singular matrix

        for row in range(col + 1, pa):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, pa + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution.
    beta = [0.0] * pa
    for i in range(pa - 1, -1, -1):
        beta[i] = aug[i][pa]
        for j in range(i + 1, pa):
            beta[i] -= aug[i][j] * beta[j]
        beta[i] /= aug[i][i]

    # R² and adjusted R².
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    y_pred = [sum(Xa[k][j] * beta[j] for j in range(pa)) for k in range(n)]
    ss_res = sum((y[k] - y_pred[k]) ** 2 for k in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    return {"r2": r2, "r2_adj": r2_adj, "beta": beta[1:], "intercept": beta[0]}


def forward_select(X_all, y, labels, max_features=10):
    """Greedy forward stepwise feature selection.

    At each step, adds the feature that maximizes adjusted R². Stops when
    adj R² > 0.999 or max_features reached.
    """
    n = len(y)
    selected = []
    remaining = list(range(len(labels)))
    results = []

    for step in range(min(max_features, len(labels))):
        best_r2 = -1
        best_idx = -1

        for idx in remaining:
            trial = selected + [idx]
            X_trial = [[X_all[k][j] for j in trial] for k in range(n)]
            res = ols_multi(X_trial, y)
            if res and res["r2_adj"] > best_r2:
                best_r2 = res["r2_adj"]
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

        X_sel = [[X_all[k][j] for j in selected] for k in range(n)]
        res = ols_multi(X_sel, y)
        results.append(
            {
                "step": step + 1,
                "added": labels[best_idx],
                "r2": res["r2"],
                "r2_adj": res["r2_adj"],
            }
        )

        if best_r2 > 0.999:
            break

    return results


# Event categories for reporting.
TRAIT_EVENTS = {
    "implementations_of_trait",
    "specialization_graph_of",
    "impl_trait_header",
    "impl_parent",
    "trait_impls_of",
    "compare_impl_item",
    "coherent_trait",
    "crate_incoherent_impls",
    "incoherent_impls",
    "collect_return_position_impl_trait_in_trait_tys",
    "check_type_wf",
    "impl_super_outlives",
    "impl_self_is_guaranteed_unsized",
    "enforce_impl_non_lifetime_params_are_constrained",
    "orphan_check_impl",
    "impl_item_implementor_ids",
}

TYPE_SYSTEM_EVENTS = {
    "is_copy_raw",
    "needs_drop_raw",
    "dropck_outlives",
    "layout_of",
    "adt_sizedness_constraint",
    "is_sized_raw",
    "adt_dtorck_constraint",
    "inhabited_predicate_type",
    "inhabited_predicate_adt",
    "try_normalize_generic_arg_after_erasing_regions",
    "is_async_drop_raw",
}

CRATE_LEVEL_EVENTS = {
    "expand_crate",
    "expand_proc_macro",
    "expand_derive_proc_macro_outer",
    "late_resolve_crate",
    "hir_crate",
    "hir_crate_items",
    "module_children",
    "visible_parent_map",
    "generate_crate_metadata",
    "effective_visibilities",
    "lint_mod",
    "check_mod_attrs",
    "lints_that_dont_need_to_run",
    "shallow_lint_levels_on",
    "get_lang_items",
    "diagnostic_items",
    "crate_variances",
    "type_check_crate",
    "drop_ast",
    "local_def_id_to_hir_id",
    "opt_hir_owner_nodes",
    "hir_attr_map",
    "attrs_for_def",
    "sanitizer_settings_for",
    "late_bound_vars_map",
    "named_variable_map",
    "resolve_bound_vars",
    "assumed_wf_types",
    "method_autoderef_steps",
    "typing_env_normalized_for_post_analysis",
    "trivial_const",
    "inherited_align",
    "adt_destructor",
    "live_symbols_and_ignored_derived_traits",
    "early_lint_checks",
    "variances_of",
    "visibility",
    "finalize_imports",
    "finalize_macro_resolutions",
}

PER_ITEM_EVENTS = {
    "predicates_of",
    "generics_of",
    "type_of",
    "explicit_predicates_of",
    "param_env",
    "constness",
    "codegen_fn_attrs",
    "intrinsic_raw",
    "lookup_stability",
    "inferred_outlives_of",
    "associated_item_def_ids",
    "associated_item",
    "associated_items",
    "inherent_impls",
    "defaultness",
    "fn_sig",
    "adt_def",
    "def_span",
    "def_kind",
    "def_ident_span",
    "is_doc_hidden",
    "lookup_deprecation_entry",
}


def categorize_event(label):
    """Categorize an event label for reporting."""
    if label in TRAIT_EVENTS:
        return "trait_system"
    if label in TYPE_SYSTEM_EVENTS:
        return "type_system"
    if label in CRATE_LEVEL_EVENTS:
        return "crate_level"
    if label in PER_ITEM_EVENTS:
        return "per_item"
    if label.startswith("metadata_decode_entry_") or label == "metadata_register_crate":
        return "metadata"
    if label.startswith("mir_pass_"):
        return "fn_body (mir_pass)"
    if label in {
        "typeck",
        "mir_borrowck",
        "MIR_borrow_checking",
        "mir_built",
        "evaluate_obligation",
        "type_op_prove_predicate",
        "check_well_formed",
        "check_mod_privacy",
        "check_liveness",
        "check_match",
        "check_unsafety",
        "check_mod_unstable_api_usage",
        "thir_body",
        "mir_promoted",
        "mir_drops_elaborated_and_const_checked",
        "mir_for_ctfe",
        "mir_const_qualif",
        "optimized_mir",
        "normalize_canonicalized_projection",
        "type_op_ascribe_user_type",
        "type_op_normalize_fn_sig",
        "type_op_normalize_clause",
        "type_op_normalize_ty",
        "implied_outlives_bounds",
        "region_scope_tree",
        "check_transmutes",
        "has_ffi_unwind_calls",
        "check_tail_calls",
        "opaque_types_defined_by",
        "upvars_mentioned",
        "coroutine_kind",
        "mir_coroutine_witnesses",
        "coroutine_hidden_types",
        "check_coroutine_obligations",
        "resolve_instance_raw",
        "thir_abstract_const",
        "used_trait_imports",
        "should_inherit_track_caller",
        "nested_bodies_within",
    }:
        return "fn_body"
    if label == "self_profile_alloc_query_strings":
        return "profiler_overhead"
    return "other"


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to symbol_graph JSON file")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Target to exclude (e.g., 'nexus-db-queries/lib'). Repeatable.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10,
        help="Maximum features in stepwise selection (default: 10)",
    )
    args = parser.parse_args()

    rows = load_targets(args.input, exclude=set(args.exclude))
    if not rows:
        print("No targets with profiling data found.", file=sys.stderr)
        sys.exit(1)

    total_wall = sum(r["wall"] for r in rows)

    # Collect labels present in >= 10% of targets.
    label_counts = {}
    for r in rows:
        for label in r["ec"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    min_count = len(rows) * 0.1
    labels = sorted(l for l, c in label_counts.items() if c >= min_count)

    y = [r["wall"] for r in rows]
    X_all = [[r["ec"].get(l, 0) for l in labels] for r in rows]

    print(f"Targets: {len(rows)}")
    print(f"Total wall time: {total_wall:.0f} ms ({total_wall/1000:.1f}s)")
    print(f"Candidate features: {len(labels)} (present in >= 10% of targets)")
    print()

    # --- Category breakdown ---
    cat_totals = {}
    for r in rows:
        for label, cost in r["ec"].items():
            cat = categorize_event(label)
            cat_totals[cat] = cat_totals.get(cat, 0) + cost
    all_events = sum(cat_totals.values())

    print("=== Category breakdown ===")
    print(f"{'Category':<25} {'Total ms':>10} {'Wall%':>7}")
    print("-" * 45)
    for cat, total in sorted(cat_totals.items(), key=lambda x: -x[1]):
        print(f"{cat:<25} {total:>10.0f} {total/total_wall*100:>6.1f}%")
    print(f"{'TOTAL events':<25} {all_events:>10.0f} {all_events/total_wall*100:>6.1f}%")
    print()

    # --- Top univariate predictors ---
    print("=== Top 20 univariate predictors (R²) ===")
    uni_results = []
    for i, label in enumerate(labels):
        vals = [X_all[k][i] for k in range(len(rows))]
        total_ms = sum(vals)
        pct = total_ms / total_wall * 100
        r = pearson(vals, y)
        uni_results.append((label, total_ms, pct, r * r, categorize_event(label)))
    uni_results.sort(key=lambda x: -x[3])

    print(f"{'Event':<55} {'Wall%':>6} {'R²':>7} {'Category'}")
    print("-" * 85)
    for label, total_ms, pct, r2, cat in uni_results[:20]:
        print(f"{label:<55} {pct:>5.1f}% {r2:>7.4f} {cat}")
    print()

    # --- Forward stepwise ---
    print(f"=== Forward stepwise regression (max {args.max_features} features) ===")
    results = forward_select(X_all, y, labels, max_features=args.max_features)

    print(f"{'Step':>4}  {'Adj R²':>8}  {'R²':>8}  {'Category':<20} Added feature")
    print("-" * 85)
    for r in results:
        cat = categorize_event(r["added"])
        print(
            f"{r['step']:>4}  {r['r2_adj']:>8.5f}  {r['r2']:>8.5f}  {cat:<20} {r['added']}"
        )

    if results:
        final = results[-1]
        print()
        print(
            f"Final model: {final['step']} features, "
            f"R²={final['r2']:.5f}, adj R²={final['r2_adj']:.5f}"
        )
        print(f"Features: {', '.join(r['added'] for r in results)}")


if __name__ == "__main__":
    main()
