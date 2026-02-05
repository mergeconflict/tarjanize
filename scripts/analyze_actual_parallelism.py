#!/usr/bin/env python3
"""
Analyze actual build parallelism from cargo timing data.

Parses cargo --timings HTML (or extracted unit data) and analyzes parallelism
at 10-second granularity, aggregating targets to match tarjanize model
granularity (lib/test/bin per package).

Usage:
    python3 analyze_actual_parallelism.py <cargo_timing.html> <symbol_graph.json>

Arguments:
    cargo_timing.html - Cargo timing HTML file from `cargo build --timings`
    symbol_graph.json - Tarjanize symbol graph (to filter to model packages)

Example:
    python3 analyze_actual_parallelism.py \\
        ~/omicron/target/cargo-timings/cargo-timing-*.html \\
        omicron.json

Output:
    - Build phases with bottleneck identification
    - Parallelism timeline (10s buckets)
    - Critical path analysis
"""

import json
import re
import sys
from collections import defaultdict


def load_unit_data(path):
    """Load and parse cargo timing unit data from HTML or JSON."""
    with open(path) as f:
        content = f.read()
        # Try to extract from cargo timing HTML
        match = re.search(r'const UNIT_DATA = (\[.*?\]);', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Fall back to raw JSON array
        return json.loads(content)


def load_model_packages(path):
    """Load package names from tarjanize symbol graph."""
    with open(path) as f:
        graph = json.load(f)
    return set(graph['packages'].keys())


def aggregate_targets(actual_data, model_packages):
    """
    Aggregate actual targets to match model granularity.

    Multiple test targets for the same package are merged into one.
    Returns list of aggregated target dicts with timing info.
    """
    aggregated = {}  # key: "pkg/target_type" -> timing info

    for u in actual_data:
        # Filter to workspace packages (by name, not version)
        if u['name'] not in model_packages:
            continue

        target = u.get('target', '').strip()

        # Skip build-scripts, examples, benchmarks
        if 'build-script' in target:
            continue
        if 'example' in target or 'bench' in target:
            continue

        # Categorize target type
        if not target:
            target_type = 'lib'
        elif '(test)' in target or 'test' in target.lower():
            target_type = 'test'
        elif 'bin' in target:
            target_type = 'bin'
        else:
            target_type = 'lib'

        key = f"{u['name']}/{target_type}"

        if key not in aggregated:
            aggregated[key] = {
                'name': u['name'],
                'target_type': target_type,
                'start': u['start'],
                'end': u['start'] + u['duration'],
                'duration': u['duration'],
                'count': 1
            }
        else:
            # Merge: earliest start, latest end, sum duration for CPU time
            agg = aggregated[key]
            agg['start'] = min(agg['start'], u['start'])
            agg['end'] = max(agg['end'], u['start'] + u['duration'])
            agg['duration'] += u['duration']
            agg['count'] += 1

    # Convert to list format
    workspace = []
    for key, agg in aggregated.items():
        workspace.append({
            'name': agg['name'],
            'target': agg['target_type'],
            'start': agg['start'],
            'duration': agg['end'] - agg['start'],  # Wall-clock duration
            'cpu_time': agg['duration'],  # Total CPU time
            'count': agg['count']
        })

    return workspace


def analyze_parallelism(workspace):
    """Analyze build parallelism and print report."""
    if not workspace:
        print("No targets found!")
        return

    # Basic stats
    first_start = min(u['start'] for u in workspace)
    last_end = max(u['start'] + u['duration'] for u in workspace)
    total_time = last_end - first_start
    total_cpu = sum(u['cpu_time'] for u in workspace)
    avg_parallelism = total_cpu / total_time

    # Count by type
    by_type = defaultdict(int)
    for w in workspace:
        by_type[w['target']] += 1

    print(f"Aggregated targets: {len(workspace)}")
    print(f"  lib:  {by_type['lib']}")
    print(f"  test: {by_type['test']}")
    print(f"  bin:  {by_type['bin']}")

    print("\n" + "=" * 70)
    print("BUILD PARALLELISM ANALYSIS (actual, aggregated to model granularity)")
    print("=" * 70)
    print(f"\nTargets: {len(workspace)}")
    print(f"Wall-clock time:   {total_time:.1f}s")
    print(f"Total CPU time:    {total_cpu:.1f}s")
    print(f"Avg parallelism:   {avg_parallelism:.2f}x")

    # Parallelism over time (10s buckets)
    buckets = defaultdict(list)
    for u in workspace:
        start = u['start']
        end = start + u['duration']
        bucket_start = int(start // 10) * 10
        bucket_end = int(end // 10) * 10
        for b in range(bucket_start, bucket_end + 10, 10):
            buckets[b].append((u['name'], u['target'], u['duration']))

    # Find peak parallelism
    peak_parallelism = max(len(units) for units in buckets.values())
    peak_time = [t for t, units in buckets.items() if len(units) == peak_parallelism][0]

    print("\n" + "-" * 70)
    print("BUILD PHASES")
    print("-" * 70)

    # Phase 1: High parallelism
    phase1_start = None
    for t in sorted(buckets.keys()):
        if len(buckets[t]) > 30 and phase1_start is None:
            phase1_start = t
            break
    phase1_end = None
    for t in sorted(buckets.keys()):
        if phase1_start and t > phase1_start and len(buckets[t]) < 30:
            phase1_end = t
            break

    phase1_units = len([u for u in workspace
                        if u['start'] >= (phase1_start or 50)
                        and u['start'] < (phase1_end or 100)])
    print(f"\nPhase 1 ({phase1_start or 50}-{phase1_end or 100}s): Dependency explosion")
    print(f"  Units started: {phase1_units}")
    print(f"  Peak concurrency: ~{peak_parallelism} targets (at {peak_time}s)")

    # Phase 2: Medium crates
    phase2_start = phase1_end or 100
    phase2_end = 140
    phase2_big = [(u['name'], u['target'], u['duration']) for u in workspace
                  if u['start'] >= phase2_start and u['start'] < phase2_end
                  and u['duration'] > 15]
    print(f"\nPhase 2 ({phase2_start}-{phase2_end}s): Medium crates")
    print(f"  Key bottlenecks:")
    for name, tgt, dur in sorted(phase2_big, key=lambda x: -x[2])[:5]:
        print(f"    {name}/{tgt}: {dur:.1f}s")

    # Phase 3: Serialization
    phase3_start = 140
    phase3_end = 340
    phase3_big = [(u['name'], u['target'], u['start'], u['duration']) for u in workspace
                  if u['start'] >= phase3_start and u['duration'] > 25]
    print(f"\nPhase 3 ({phase3_start}-{phase3_end}s): Serialization bottleneck")
    print(f"  Key bottlenecks:")
    for name, tgt, start, dur in sorted(phase3_big, key=lambda x: -x[3])[:5]:
        print(f"    {name}/{tgt}: {dur:.1f}s (started {start:.1f}s)")

    # Time with low parallelism
    low_parallel_time = sum(10 for t, units in buckets.items()
                            if len(units) < 5 and t >= 50)
    print(f"\n  Time with parallelism < 5: {low_parallel_time}s "
          f"({100*low_parallel_time/total_time:.0f}% of build)")

    # Bottleneck summary
    print("\n" + "-" * 70)
    print("BOTTLENECK SUMMARY")
    print("-" * 70)

    critical = []
    for u in workspace:
        if 'nexus-db-model' in u['name'] and u['target'] == 'lib':
            critical.append((u['name'], u['target'], u['duration']))
        elif 'nexus-db-queries' in u['name'] and u['target'] == 'lib':
            critical.append((u['name'], u['target'], u['duration']))
        elif 'omicron-nexus' in u['name'] and u['target'] == 'lib':
            critical.append((u['name'], u['target'], u['duration']))

    if critical:
        print("\nThree main serialization points:")
        chain_total = 0
        for i, (name, tgt, dur) in enumerate(critical, 1):
            print(f"  {i}. {name}/{tgt} ({dur:.1f}s)")
            chain_total += dur

        print(f"\nThese {len(critical)} targets account for ~{chain_total:.0f}s on critical path")
        print(f"out of {total_time:.0f}s total ({100*chain_total/total_time:.0f}% of wall-clock)")

    # Parallelism timeline
    print("\n" + "-" * 70)
    print("PARALLELISM TIMELINE (10s buckets)")
    print("-" * 70)
    print(f"{'Time':>6}  {'Active':>6}  Top targets")
    for t in sorted(buckets.keys()):
        units = buckets[t]
        names = [f"{n}/{tgt}" for n, tgt, _ in sorted(units, key=lambda x: -x[2])[:3]]
        more = f" (+{len(units)-3})" if len(units) > 3 else ""
        print(f"{t:>4}s   {len(units):>6}  {', '.join(names)}{more}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    unit_data_path = sys.argv[1]
    symbol_graph_path = sys.argv[2]

    actual_data = load_unit_data(unit_data_path)
    model_packages = load_model_packages(symbol_graph_path)
    workspace = aggregate_targets(actual_data, model_packages)
    analyze_parallelism(workspace)


if __name__ == '__main__':
    main()
