#!/usr/bin/env python3
"""
Analyze frontend vs codegen time breakdown for lib targets.

Parses cargo timing data to show what fraction of compilation time is spent
in frontend (type checking, borrow checking) vs codegen (LLVM code generation).
This is critical for understanding rmeta pipelining effects.

Usage:
    python3 analyze_frontend_codegen.py <cargo_timing.html> <symbol_graph.json>

Arguments:
    cargo_timing.html - Cargo timing HTML file from `cargo build --timings`
    symbol_graph.json - Tarjanize symbol graph (to filter to model packages)

Example:
    python3 analyze_frontend_codegen.py unit_data.json omicron.json

Output:
    - Per-package frontend/codegen breakdown
    - Distribution statistics
    - Aggregate totals

Key Finding:
    Frontend is typically 70% of lib compilation time (weighted by time).
    Critical path bottlenecks are even more frontend-heavy (88-98%).
    This means rmeta pipelining has massive impact on actual parallelism.

Note:
    Requires cargo timing data with section information (frontend/codegen).
    Not all builds include this data - it depends on cargo version and flags.
"""

import json
import re
import statistics
import sys


def load_unit_data(path):
    """Load and parse cargo timing unit data from HTML or JSON."""
    with open(path) as f:
        content = f.read()
        match = re.search(r'const UNIT_DATA = (\[.*?\]);', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(content)


def load_model_packages(path):
    """Load package names from tarjanize symbol graph."""
    with open(path) as f:
        graph = json.load(f)
    return set(graph['packages'].keys())


def analyze_breakdown(actual_data, model_packages):
    """Analyze frontend vs codegen breakdown for lib targets."""
    results = []

    for u in actual_data:
        # Filter to workspace packages (by name, not version)
        if u['name'] not in model_packages:
            continue

        target = u.get('target', '').strip()
        if target:  # Only lib targets (no target suffix)
            continue

        if not u.get('sections'):
            continue

        frontend = 0
        codegen = 0
        for section in u['sections']:
            name = section[0]
            timing = section[1]
            dur = timing['end'] - timing['start']
            if name == 'frontend':
                frontend = dur
            elif name == 'codegen':
                codegen = dur

        total = frontend + codegen
        if total > 0:
            results.append({
                'name': u['name'],
                'total': total,
                'frontend': frontend,
                'codegen': codegen,
                'frontend_pct': 100 * frontend / total if total > 0 else 0
            })

    if not results:
        print("No lib targets with section data found!")
        print("Make sure the cargo timing data includes frontend/codegen sections.")
        return

    # Sort by total time
    results.sort(key=lambda x: -x['total'])

    print("=" * 75)
    print("FRONTEND VS CODEGEN BREAKDOWN (lib targets)")
    print("=" * 75)
    print(f"\n{'Package':<40} {'Total':>8} {'Frontend':>8} {'Codegen':>8} {'FE %':>6}")
    print("-" * 75)
    for r in results[:20]:
        print(f"{r['name']:<40} {r['total']:>8.1f} {r['frontend']:>8.1f} "
              f"{r['codegen']:>8.1f} {r['frontend_pct']:>5.0f}%")

    if len(results) > 20:
        print(f"  ... ({len(results) - 20} more packages)")

    # Summary stats
    total_frontend = sum(r['frontend'] for r in results)
    total_codegen = sum(r['codegen'] for r in results)
    total_all = total_frontend + total_codegen

    print(f"\n" + "-" * 75)
    print(f"{'TOTAL (' + str(len(results)) + ' libs)':<40} {total_all:>8.1f} "
          f"{total_frontend:>8.1f} {total_codegen:>8.1f} "
          f"{100*total_frontend/total_all:>5.0f}%")

    # Distribution
    fe_pcts = [r['frontend_pct'] for r in results if r['total'] > 1]

    if fe_pcts:
        print(f"\nFrontend % distribution (libs > 1s):")
        print(f"  Mean:   {statistics.mean(fe_pcts):.0f}%")
        print(f"  Median: {statistics.median(fe_pcts):.0f}%")
        print(f"  Min:    {min(fe_pcts):.0f}%")
        print(f"  Max:    {max(fe_pcts):.0f}%")
        print(f"  Weighted by time: {100*total_frontend/total_all:.0f}%")

    # Highlight bottlenecks
    print("\n" + "-" * 75)
    print("CRITICAL PATH BOTTLENECKS")
    print("-" * 75)

    bottlenecks = ['nexus-db-model', 'nexus-db-queries', 'nexus-db-schema',
                   'omicron-nexus', 'oxide-client']
    print(f"\n{'Package':<30} {'Total':>8} {'Frontend':>8} {'Codegen':>8} {'FE %':>6}")
    print("-" * 65)
    for name in bottlenecks:
        matches = [r for r in results if name in r['name']]
        for r in matches:
            print(f"{r['name']:<30} {r['total']:>8.1f} {r['frontend']:>8.1f} "
                  f"{r['codegen']:>8.1f} {r['frontend_pct']:>5.0f}%")

    print("\n" + "-" * 75)
    print("IMPLICATIONS FOR TARJANIZE")
    print("-" * 75)
    print(f"""
Key insight: Frontend is {100*total_frontend/total_all:.0f}% of lib compilation time.

With rmeta pipelining:
- Downstream crates start after frontend completes (rmeta available)
- Codegen runs in parallel with downstream compilation
- Critical path is determined by FRONTEND costs, not total costs

This means:
1. Splitting a crate helps if it reduces frontend time
2. CGU optimization (codegen parallelism) is less important than we thought
3. Symbol-level backend costs may be misleading for critical path
""")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    unit_data_path = sys.argv[1]
    symbol_graph_path = sys.argv[2]

    actual_data = load_unit_data(unit_data_path)
    model_packages = load_model_packages(symbol_graph_path)
    analyze_breakdown(actual_data, model_packages)


if __name__ == '__main__':
    main()
