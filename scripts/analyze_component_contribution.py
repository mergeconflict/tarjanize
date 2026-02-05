#!/usr/bin/env python3
"""
Analyze how each cost component contributes to model accuracy.

Tests the impact of including/excluding metadata and linking costs on the
model's R² correlation with actual build times. This analysis informed the
decision to keep metadata (+10% R²) and remove linking (+0.4% R²) from the
cost model.

Usage:
    python3 analyze_component_contribution.py <cargo_timing.html> <symbol_graph.json>

Arguments:
    cargo_timing.html - Cargo timing HTML file (contains build time table)
    symbol_graph.json - Tarjanize symbol graph with costs

Example:
    python3 analyze_component_contribution.py cargo-timing.html omicron.json

Output:
    - R² values for LIB, TEST, and merged targets
    - Comparison of different cost component combinations
    - Impact analysis showing contribution of each component

Key Findings:
    - Metadata: +10% R² improvement (essential)
    - Linking: +0.4% R² improvement (negligible, removed from model)
"""

import json
import re
import sys
from pathlib import Path


def load_actual_times(html_path):
    """Parse actual build times from cargo timing HTML."""
    html = Path(html_path).read_text()
    pattern = r'<td>(\d+)\.</td>\s*<td>([^<]+)</td>\s*<td>([\d.]+)s</td>'
    matches = re.findall(pattern, html)

    actual_times = {}
    for rank, name_info, time_str in matches:
        time_s = float(time_str)
        parts = name_info.split(' v0.1.0')
        if len(parts) < 1:
            continue
        package = parts[0].strip()
        rest = parts[1] if len(parts) > 1 else ""

        if "(test)" in rest:
            target = "test"
        elif '"bin"' in rest:
            bin_match = re.search(r'(\w+) "bin"', rest)
            target = f"bin/{bin_match.group(1)}" if bin_match else "bin"
        elif "build-script" in rest:
            continue
        else:
            target = "lib"

        key = f"{package}/{target}"
        if key not in actual_times:
            actual_times[key] = time_s * 1000  # Convert to ms

    return actual_times


def load_symbol_graph(path):
    """Load tarjanize symbol graph."""
    with open(path) as f:
        return json.load(f)


def compute_costs(graph, include_metadata=True, include_linking=False):
    """Compute target costs with configurable components."""
    modeled = {}

    for pkg_name, pkg in graph["packages"].items():
        for target_key, crate_data in pkg["targets"].items():
            key = f"{pkg_name}/{target_key}"

            def collect_frontend(module):
                total = sum(s.get("frontend_cost_ms", 0)
                            for s in module.get("symbols", {}).values())
                for sub in module.get("submodules", {}).values():
                    total += collect_frontend(sub)
                return total

            def max_backend(module):
                this_cost = sum(s.get("backend_cost_ms", 0)
                                for s in module.get("symbols", {}).values())
                sub_max = max((max_backend(sub)
                               for sub in module.get("submodules", {}).values()),
                              default=0)
                return max(this_cost, sub_max)

            frontend = collect_frontend(crate_data.get("root", {}))
            backend = max_backend(crate_data.get("root", {}))

            cost = frontend + backend
            if include_metadata:
                cost += crate_data.get("metadata_ms", 0)
            if include_linking:
                cost += crate_data.get("linking_ms", 0)

            modeled[key] = cost

    return modeled


def run_regression(data):
    """Run linear regression and return R²."""
    if len(data) < 3:
        return None

    actual = [p[1] for p in data]
    modeled = [p[2] for p in data]
    n = len(actual)

    sum_x = sum(modeled)
    sum_y = sum(actual)
    sum_xy = sum(m * a for m, a in zip(modeled, actual))
    sum_x2 = sum(m * m for m in modeled)

    denom = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return None

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    mean_y = sum_y / n
    ss_res = sum((a - (slope * m + intercept)) ** 2 for a, m in zip(actual, modeled))
    ss_tot = sum((a - mean_y) ** 2 for a in actual)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return r_squared


def analyze_config(actual_times, graph, include_metadata, include_linking):
    """Analyze a specific cost configuration."""
    modeled_times = compute_costs(graph, include_metadata, include_linking)

    lib_data = []
    test_data = []

    for key in actual_times:
        if key in modeled_times:
            entry = (key, actual_times[key], modeled_times[key])
            if "/test" in key:
                test_data.append(entry)
            elif "/bin/" not in key:
                lib_data.append(entry)

    # Merged lib+test
    pkg_lib = {k.split('/')[0]: (k, a, m) for k, a, m in lib_data}
    pkg_test = {k.split('/')[0]: (k, a, m) for k, a, m in test_data}

    merged_data = []
    for pkg in set(pkg_lib.keys()) | set(pkg_test.keys()):
        actual_sum = 0
        modeled_sum = 0
        if pkg in pkg_lib:
            actual_sum += pkg_lib[pkg][1]
            modeled_sum += pkg_lib[pkg][2]
        if pkg in pkg_test:
            actual_sum += pkg_test[pkg][1]
            modeled_sum += pkg_test[pkg][2]
        if actual_sum > 0:
            merged_data.append((pkg, actual_sum, modeled_sum))

    r2_lib = run_regression(lib_data)
    r2_test = run_regression(test_data)
    r2_merged = run_regression(merged_data)

    return r2_lib, r2_test, r2_merged


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    html_path = sys.argv[1]
    graph_path = sys.argv[2]

    actual_times = load_actual_times(html_path)
    graph = load_symbol_graph(graph_path)

    print("=" * 70)
    print("CONTRIBUTION OF COST COMPONENTS TO MODEL ACCURACY")
    print("=" * 70)
    print()
    print(f"{'Configuration':<35} {'LIB R²':>10} {'TEST R²':>10} {'MERGED R²':>10}")
    print("-" * 70)

    configs = [
        (False, False, "Frontend + Backend only"),
        (True, False, "Frontend + Backend + Metadata"),
        (False, True, "Frontend + Backend + Linking"),
        (True, True, "Frontend + Backend + Meta + Link"),
    ]

    results = {}
    for include_meta, include_link, label in configs:
        r2_lib, r2_test, r2_merged = analyze_config(
            actual_times, graph, include_meta, include_link)
        results[label] = (r2_lib, r2_test, r2_merged)
        print(f"{label:<35} {r2_lib:>10.4f} {r2_test:>10.4f} {r2_merged:>10.4f}")

    print()
    print("=" * 70)
    print("IMPACT ANALYSIS")
    print("=" * 70)
    print()

    base = results["Frontend + Backend only"]
    with_meta = results["Frontend + Backend + Metadata"]
    with_link = results["Frontend + Backend + Linking"]

    print("Effect of adding METADATA (to frontend+backend):")
    print(f"  LIB:    {base[0]:.4f} → {with_meta[0]:.4f}  "
          f"(Δ = {with_meta[0]-base[0]:+.4f})")
    print(f"  TEST:   {base[1]:.4f} → {with_meta[1]:.4f}  "
          f"(Δ = {with_meta[1]-base[1]:+.4f})")
    print(f"  MERGED: {base[2]:.4f} → {with_meta[2]:.4f}  "
          f"(Δ = {with_meta[2]-base[2]:+.4f})")
    print()

    print("Effect of adding LINKING (to frontend+backend):")
    print(f"  LIB:    {base[0]:.4f} → {with_link[0]:.4f}  "
          f"(Δ = {with_link[0]-base[0]:+.4f})")
    print(f"  TEST:   {base[1]:.4f} → {with_link[1]:.4f}  "
          f"(Δ = {with_link[1]-base[1]:+.4f})")
    print(f"  MERGED: {base[2]:.4f} → {with_link[2]:.4f}  "
          f"(Δ = {with_link[2]-base[2]:+.4f})")

    print()
    print("-" * 70)
    print("CONCLUSION")
    print("-" * 70)
    print()
    print(f"  Metadata contribution to LIB R²: {with_meta[0]-base[0]:+.3f} "
          f"({100*(with_meta[0]-base[0])/base[0]:+.1f}%)")
    print(f"  Linking contribution to LIB R²:  {with_link[0]-base[0]:+.3f} "
          f"({100*(with_link[0]-base[0])/base[0]:+.1f}%)")
    print()
    print("  >>> Metadata is essential; linking is negligible <<<")


if __name__ == '__main__':
    main()
