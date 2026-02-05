#!/usr/bin/env python3
"""
Analyze self-profile data to detect event overlap/parallelism.

This script examines rustc's -Zself-profile output to determine if frontend
events overlap in time (indicating parallel execution), which would mean
summing durations gives CPU time rather than wall-clock time.

Usage:
    python3 analyze_profile_parallelism.py <profile_data_stem>

The script uses summarize (from rustc-perf) to decode profile data.
"""

import json
import subprocess
import sys
from collections import defaultdict


def get_profile_events(profile_stem):
    """
    Get events from profile using summarize tool.

    Returns list of (label, start_ns, duration_ns, thread_id, additional_data).
    """
    # Use summarize to get raw event data
    # summarize --json gives us detailed timing info
    result = subprocess.run(
        ['summarize', 'summarize', profile_stem, '--json'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error running summarize: {result.stderr}", file=sys.stderr)
        return []

    return json.loads(result.stdout)


def analyze_event_overlap(events):
    """Analyze how much events overlap in time."""
    if not events:
        print("No events to analyze")
        return

    # Build timeline of active events per thread
    # Each event has start time and end time
    timeline = []  # (time, +1 for start or -1 for end, label, thread)

    for event in events:
        start = event.get('start_ns', 0)
        dur = event.get('duration_ns', 0)
        end = start + dur
        label = event.get('label', '')
        thread = event.get('thread_id', 0)

        timeline.append((start, 1, label, thread))
        timeline.append((end, -1, label, thread))

    # Sort by time
    timeline.sort(key=lambda x: (x[0], -x[1]))  # starts before ends at same time

    # Count max concurrent events
    max_concurrent = 0
    current_concurrent = 0
    concurrent_at_max = []

    active_events = defaultdict(int)  # label -> count

    for time, delta, label, thread in timeline:
        if delta == 1:
            active_events[label] += 1
        else:
            active_events[label] -= 1
            if active_events[label] == 0:
                del active_events[label]

        current_concurrent += delta
        if current_concurrent > max_concurrent:
            max_concurrent = current_concurrent
            concurrent_at_max = list(active_events.keys())

    print(f"Max concurrent events: {max_concurrent}")
    print(f"Labels at max: {concurrent_at_max[:10]}")

    # Calculate total CPU time vs wall time
    if timeline:
        wall_start = min(t[0] for t in timeline)
        wall_end = max(t[0] for t in timeline)
        wall_time = wall_end - wall_start

        total_cpu = sum(e.get('duration_ns', 0) for e in events)

        print(f"\nWall time: {wall_time / 1e9:.3f}s")
        print(f"Total CPU time (summed): {total_cpu / 1e9:.3f}s")
        print(f"CPU/Wall ratio: {total_cpu / wall_time:.2f}x")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNote: Requires 'summarize' tool from rustc-perf")
        sys.exit(1)

    profile_stem = sys.argv[1]
    events = get_profile_events(profile_stem)
    analyze_event_overlap(events)


if __name__ == '__main__':
    main()
