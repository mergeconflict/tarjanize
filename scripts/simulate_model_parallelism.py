#!/usr/bin/env python3
"""
Simulate parallel build execution using the tarjanize cost model.

Uses `tarjanize cost` output which includes rmeta pipelining - downstream
targets can start when upstream rmeta is ready (after frontend), not when
upstream fully completes.

Usage:
    python3 simulate_model_parallelism.py <symbol_graph.json>

Arguments:
    symbol_graph.json - Tarjanize symbol graph with costs and dependencies

Example:
    python3 simulate_model_parallelism.py omicron.json

Output:
    - Simulated critical path and total CPU time (with pipelining)
    - Parallelism timeline (10s buckets)
    - Comparison-ready format matching analyze_actual_parallelism.py
"""

import re
import subprocess
import sys
from collections import defaultdict


def run_tarjanize_cost(symbol_graph_path):
    """
    Run tarjanize cost and parse the text output.

    Returns list of dicts with keys: name, start, rmeta, finish, cost
    Times are in seconds.
    """
    result = subprocess.run(
        ['tarjanize', 'cost', '-i', symbol_graph_path],
        capture_output=True,
        text=True,
        check=True
    )

    targets = []
    in_all_targets = False

    for line in result.stdout.split('\n'):
        # Look for "All targets by finish time" section
        if 'All targets by finish time' in line:
            in_all_targets = True
            continue

        if not in_all_targets:
            continue

        # Skip header and separator lines
        if line.startswith('--') or 'Start' in line and 'Rmeta' in line:
            continue

        # Parse data lines: "  845133.9   1450811.1   1722394.0    877260.1  omicron-nexus/lib  ..."
        # Format: Start, Rmeta, Finish, Cost, Target, Dependencies
        match = re.match(
            r'\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)',
            line
        )
        if match:
            start_ms = float(match.group(1))
            rmeta_ms = float(match.group(2))
            finish_ms = float(match.group(3))
            cost_ms = float(match.group(4))
            target = match.group(5)

            targets.append({
                'name': target,
                'start': start_ms / 1000,  # Convert to seconds
                'rmeta': rmeta_ms / 1000,
                'finish': finish_ms / 1000,
                'cost': cost_ms / 1000,
            })

    return targets


def analyze_simulation(targets):
    """Analyze simulated build and print report."""
    if not targets:
        print("No targets found in tarjanize cost output")
        return

    # Build workspace list with normalized target types
    workspace = []
    for t in targets:
        parts = t['name'].split('/')
        pkg = parts[0]
        tgt = '/'.join(parts[1:]) if len(parts) > 1 else 'lib'
        # Normalize bin targets
        if tgt.startswith('bin/'):
            tgt = 'bin'
        workspace.append({
            'name': pkg,
            'target': tgt,
            'start': t['start'],
            'finish': t['finish'],
            'duration': t['cost'],
        })

    # Stats
    total_cpu = sum(w['duration'] for w in workspace)
    critical_time = max(w['finish'] for w in workspace)
    avg_parallelism = total_cpu / critical_time if critical_time > 0 else 0

    # Count by type
    by_type = defaultdict(int)
    for w in workspace:
        by_type[w['target']] += 1

    print(f"Model targets: {len(workspace)}")
    print(f"  lib:  {by_type['lib']}")
    print(f"  test: {by_type['test']}")
    print(f"  bin:  {by_type['bin']}")

    print("\n" + "=" * 70)
    print("SIMULATED BUILD (model costs, rmeta pipelining, infinite parallelism)")
    print("=" * 70)
    print(f"\nTargets: {len(workspace)}")
    print(f"Critical path:     {critical_time:.1f}s")
    print(f"Total CPU time:    {total_cpu:.1f}s")
    print(f"Avg parallelism:   {avg_parallelism:.2f}x")

    # Parallelism over time (10s buckets)
    buckets = defaultdict(list)
    for w in workspace:
        start = w['start']
        end = w['finish']
        bucket_start = int(start // 10) * 10
        bucket_end = int(end // 10) * 10
        for b in range(bucket_start, bucket_end + 10, 10):
            buckets[b].append((w['name'], w['target'], w['duration']))

    print("\n" + "-" * 70)
    print("BUILD PHASES")
    print("-" * 70)

    # Find peak
    peak_parallelism = max(len(units) for units in buckets.values()) if buckets else 0
    peak_time = ([t for t, units in buckets.items()
                  if len(units) == peak_parallelism][0] if buckets else 0)

    # Phase 1
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

    phase1_units = len([w for w in workspace
                        if w['start'] >= (phase1_start or 0)
                        and w['start'] < (phase1_end or 100)])
    print(f"\nPhase 1 ({phase1_start or 0}-{phase1_end or 100}s): Dependency explosion")
    print(f"  Units started: {phase1_units}")
    print(f"  Peak concurrency: ~{peak_parallelism} targets (at {peak_time}s)")

    # Phase 2
    phase2_start = phase1_end or 100
    phase2_end = phase2_start + 200
    phase2_big = [(w['name'], w['target'], w['duration']) for w in workspace
                  if w['start'] >= phase2_start and w['start'] < phase2_end
                  and w['duration'] > 50]
    print(f"\nPhase 2 ({phase2_start}-{int(phase2_end)}s): Medium crates")
    print(f"  Key bottlenecks:")
    for name, tgt, dur in sorted(phase2_big, key=lambda x: -x[2])[:5]:
        print(f"    {name}/{tgt}: {dur:.1f}s")

    # Phase 3: Large targets
    phase3_big = [(w['name'], w['target'], w['start'], w['duration'])
                  for w in workspace if w['duration'] > 100]
    print(f"\nPhase 3: Serialization bottleneck")
    print(f"  Key bottlenecks:")
    for name, tgt, start, dur in sorted(phase3_big, key=lambda x: -x[3])[:5]:
        print(f"    {name}/{tgt}: {dur:.1f}s (started {start:.1f}s)")

    # Low parallelism time
    low_parallel_time = sum(10 for t, units in buckets.items() if len(units) < 5)
    print(f"\n  Time with parallelism < 5: {low_parallel_time}s "
          f"({100*low_parallel_time/critical_time:.0f}% of build)")

    # Bottleneck summary
    print("\n" + "-" * 70)
    print("BOTTLENECK SUMMARY")
    print("-" * 70)

    critical = []
    for w in workspace:
        if 'nexus-db-model' in w['name'] and w['target'] == 'lib':
            critical.append((w['name'], w['target'], w['duration']))
        elif 'nexus-db-queries' in w['name'] and w['target'] == 'lib':
            critical.append((w['name'], w['target'], w['duration']))
        elif 'omicron-nexus' in w['name'] and w['target'] == 'lib':
            critical.append((w['name'], w['target'], w['duration']))

    if critical:
        print("\nThree main serialization points:")
        chain_total = 0
        for i, (name, tgt, dur) in enumerate(critical, 1):
            print(f"  {i}. {name}/{tgt} ({dur:.1f}s)")
            chain_total += dur

        print(f"\nThese {len(critical)} targets account for ~{chain_total:.0f}s on critical path")
        print(f"out of {critical_time:.0f}s total "
              f"({100*chain_total/critical_time:.0f}% of critical path)")

    # Parallelism timeline
    print("\n" + "-" * 70)
    print("PARALLELISM TIMELINE (10s buckets)")
    print("-" * 70)
    print(f"{'Time':>6}  {'Active':>6}  Top targets")
    for t in sorted(buckets.keys())[:50]:
        units = buckets[t]
        names = [f"{n}/{tgt}" for n, tgt, _ in sorted(units, key=lambda x: -x[2])[:3]]
        more = f" (+{len(units)-3})" if len(units) > 3 else ""
        print(f"{t:>4}s   {len(units):>6}  {', '.join(names)}{more}")
    if len(buckets) > 50:
        print(f"  ... ({len(buckets) - 50} more buckets)")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    symbol_graph_path = sys.argv[1]

    targets = run_tarjanize_cost(symbol_graph_path)
    analyze_simulation(targets)


if __name__ == '__main__':
    main()
