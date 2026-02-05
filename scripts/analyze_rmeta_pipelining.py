#!/usr/bin/env python3
"""
Analyze rmeta pipelining effects in cargo builds.

Demonstrates that Cargo uses pipelined compilation where downstream crates
start compiling as soon as upstream rmeta (metadata) is available, NOT when
the full rlib is ready. This has major implications for critical path modeling.

Usage:
    python3 analyze_rmeta_pipelining.py <cargo_timing.html> [package_name]

Arguments:
    cargo_timing.html - Cargo timing HTML file from `cargo build --timings`
    package_name   - Optional: specific package to analyze (default: auto-detect)

Example:
    python3 analyze_rmeta_pipelining.py unit_data.json
    python3 analyze_rmeta_pipelining.py unit_data.json oxide-client

Output:
    - Timing relationship between lib and test targets
    - Evidence of pipelined execution (test starts before lib finishes)
    - Frontend/codegen breakdown if available in timing data

Key Finding:
    Test targets often start just seconds after their lib target starts,
    even when the lib takes 30+ seconds to compile. This is because:
    1. Lib's frontend (type checking) produces rmeta
    2. Test only needs rmeta to start compiling
    3. Lib's codegen runs in parallel with test's compilation
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


def analyze_package(actual_data, package_name):
    """Analyze lib vs test timing for a specific package."""
    lib_unit = None
    test_unit = None

    for u in actual_data:
        if u['name'] == package_name:
            target = u.get('target', '').strip()
            if not target:
                lib_unit = u
            elif 'test' in target:
                test_unit = u

    if not lib_unit:
        print(f"No lib target found for {package_name}")
        return False

    print("=" * 70)
    print(f"{package_name.upper()}: LIB vs TEST TIMING")
    print("=" * 70)

    print(f"\n{package_name}/lib:")
    print(f"  Start: {lib_unit['start']:.1f}s")
    print(f"  Duration: {lib_unit['duration']:.1f}s")
    print(f"  End: {lib_unit['start'] + lib_unit['duration']:.1f}s")

    # Check for sections (frontend/codegen breakdown)
    if lib_unit.get('sections'):
        print(f"  Sections:")
        for section in lib_unit['sections']:
            name = section[0]
            timing = section[1]
            dur = timing['end'] - timing['start']
            print(f"    {name}: {timing['start']:.2f}s - {timing['end']:.2f}s "
                  f"(dur: {dur:.2f}s)")

    if test_unit:
        print(f"\n{package_name}/test:")
        print(f"  Start: {test_unit['start']:.1f}s")
        print(f"  Duration: {test_unit['duration']:.1f}s")
        print(f"  End: {test_unit['start'] + test_unit['duration']:.1f}s")

        lib_end = lib_unit['start'] + lib_unit['duration']
        test_start_after_lib = test_unit['start'] - lib_unit['start']
        overlap = lib_end - test_unit['start']

        print(f"\nTiming relationship:")
        print(f"  Lib starts at: {lib_unit['start']:.1f}s")
        print(f"  Test starts at: {test_unit['start']:.1f}s")
        print(f"  Test starts {test_start_after_lib:.1f}s after lib starts")
        print(f"  Lib ends at: {lib_end:.1f}s")

        if overlap > 0:
            print(f"  Test started {overlap:.1f}s BEFORE lib finished!")
            print(f"\n  >>> EVIDENCE OF RMETA PIPELINING <<<")
            print(f"  The test target didn't wait for the full rlib.")
            print(f"  It started after rmeta was available (frontend done).")
        else:
            print(f"  Test started {-overlap:.1f}s after lib finished")
    else:
        print(f"\nNo test target found for {package_name}")

    return True


def find_best_examples(actual_data):
    """Find packages with the most dramatic pipelining evidence."""
    examples = []

    # Group by package (all packages, not just workspace)
    packages = {}
    for u in actual_data:
        name = u['name']
        target = u.get('target', '').strip()

        if name not in packages:
            packages[name] = {'lib': None, 'test': None}

        if not target:
            packages[name]['lib'] = u
        elif 'test' in target and packages[name]['test'] is None:
            packages[name]['test'] = u

    # Find packages with both lib and test, where test starts before lib ends
    for name, units in packages.items():
        lib_unit = units['lib']
        test_unit = units['test']

        if not lib_unit or not test_unit:
            continue

        lib_end = lib_unit['start'] + lib_unit['duration']
        overlap = lib_end - test_unit['start']

        if overlap > 5 and lib_unit['duration'] > 10:  # Significant overlap
            examples.append({
                'name': name,
                'lib_duration': lib_unit['duration'],
                'test_starts_after': test_unit['start'] - lib_unit['start'],
                'overlap': overlap,
            })

    # Sort by overlap (most dramatic first)
    examples.sort(key=lambda x: -x['overlap'])
    return examples[:10]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    unit_data_path = sys.argv[1]
    package_name = sys.argv[2] if len(sys.argv) > 2 else None

    actual_data = load_unit_data(unit_data_path)

    if package_name:
        analyze_package(actual_data, package_name)
    else:
        # Auto-detect interesting examples
        print("=" * 70)
        print("RMETA PIPELINING ANALYSIS")
        print("=" * 70)
        print("\nFinding packages with evidence of rmeta pipelining...")
        print("(test starts before lib finishes)\n")

        examples = find_best_examples(actual_data)

        if not examples:
            print("No clear examples found.")
            return

        print(f"{'Package':<35} {'Lib dur':>10} {'Test after':>12} {'Overlap':>10}")
        print("-" * 70)
        for ex in examples:
            print(f"{ex['name']:<35} {ex['lib_duration']:>10.1f}s "
                  f"{ex['test_starts_after']:>10.1f}s {ex['overlap']:>10.1f}s")

        print("\n" + "=" * 70)
        print("DETAILED ANALYSIS OF TOP EXAMPLES")
        print("=" * 70)

        # Analyze top 3 in detail
        for ex in examples[:3]:
            print()
            analyze_package(actual_data, ex['name'])


if __name__ == '__main__':
    main()
