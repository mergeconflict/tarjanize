#!/usr/bin/env python3
"""
Analyze build parallelism at 1-second resolution.

Provides more precise parallelism measurements than the 10-second bucket
analysis, useful for identifying exact peak parallelism and timing.

Usage:
    python3 analyze_parallelism_1s.py <cargo_timing.html> <symbol_graph.json>

Arguments:
    cargo_timing.html - Cargo timing HTML file from `cargo build --timings`
    symbol_graph.json - Tarjanize symbol graph (to filter to model packages)

Example:
    python3 analyze_parallelism_1s.py unit_data.json omicron.json

Output:
    - Peak and average parallelism (instantaneous)
    - Parallelism timeline at 5-second display intervals
    - Time periods with high parallelism (>20 concurrent targets)
"""

import json
import re
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


def analyze_parallelism(actual_data, model_packages):
    """Analyze instantaneous parallelism at 1-second resolution."""
    # Filter to workspace units (by package name, not version)
    workspace = [u for u in actual_data
                 if u['name'] in model_packages
                 and 'build-script' not in u.get('target', '')]

    if not workspace:
        print("No matching targets found!")
        return

    # Calculate instantaneous parallelism at 1-second intervals
    max_time = int(max(u['start'] + u['duration'] for u in workspace)) + 1
    parallelism = []
    for t in range(0, max_time):
        active = sum(1 for u in workspace
                     if u['start'] <= t < u['start'] + u['duration'])
        parallelism.append(active)

    peak = max(parallelism)
    peak_time = parallelism.index(peak)
    avg = sum(parallelism) / len(parallelism)

    print("=" * 70)
    print("INSTANTANEOUS PARALLELISM (1-second resolution)")
    print("=" * 70)
    print(f"\nTotal targets: {len(workspace)}")
    print(f"Build duration: {max_time}s")
    print(f"\nPeak parallelism: {peak} targets (at t={peak_time}s)")
    print(f"Average parallelism: {avg:.1f} targets")

    # Show timeline at 5s intervals
    print(f"\nParallelism over time (5s intervals):")
    print(f"{'Time':>6}  {'Active':>6}  {'Bar'}")
    print("-" * 60)
    for t in range(0, max_time, 5):
        active = parallelism[t]
        bar = "#" * (active // 2)
        print(f"{t:>4}s   {active:>6}  {bar}")

    # When is parallelism high (>20)?
    high_parallel = [(t, p) for t, p in enumerate(parallelism) if p > 20]
    if high_parallel:
        print(f"\nTime with parallelism > 20: {len(high_parallel)}s")
        print(f"  From t={high_parallel[0][0]}s to t={high_parallel[-1][0]}s")

    # When is parallelism low (<5)?
    low_parallel = [(t, p) for t, p in enumerate(parallelism) if p < 5 and t > 50]
    if low_parallel:
        print(f"\nTime with parallelism < 5 (after t=50s): {len(low_parallel)}s")

    # Distribution summary
    print("\n" + "-" * 70)
    print("PARALLELISM DISTRIBUTION")
    print("-" * 70)
    brackets = [
        (0, 1, "Idle (0)"),
        (1, 5, "Low (1-4)"),
        (5, 20, "Medium (5-19)"),
        (20, 50, "High (20-49)"),
        (50, 200, "Very high (50+)"),
    ]
    for low, high, label in brackets:
        count = sum(1 for p in parallelism if low <= p < high)
        pct = 100 * count / len(parallelism)
        print(f"  {label:20} {count:>5}s ({pct:>5.1f}%)")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    unit_data_path = sys.argv[1]
    symbol_graph_path = sys.argv[2]

    actual_data = load_unit_data(unit_data_path)
    model_packages = load_model_packages(symbol_graph_path)
    analyze_parallelism(actual_data, model_packages)


if __name__ == '__main__':
    main()
