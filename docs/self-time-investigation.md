# Self-Time Investigation

This document captures everything learned during the investigation of profile cost attribution and the self-time computation algorithm.

## The Problem: Nesting Inflation

When rustc compiles code, it emits self-profile events for each compilation phase (typeck, borrowck, etc.). These events can **nest**: when `typeck(foo)` runs, it might trigger `typeck(bar)`, which runs inside `typeck(foo)`.

If we naively sum event durations, we double-count:

```
|-------------- typeck(foo) 100ms --------------|
    |------ typeck(bar) 60ms ------|
```

- Naive sum: 100 + 60 = **160ms** (inflated)
- Actual work: **100ms** (bar's time is already inside foo's)

### Observed Inflation

From analysis of tokio profile data:
- Sum of raw event durations: **116s**
- Sum of self-times (from analyzeme): **22s**
- Wall-clock (from profile metadata): **22s**
- Cargo timing (actual build): **6.6s**

The nesting inflation is ~5-6x (116s / 22s).

The remaining 3.3x gap (22s / 6.6s) is **unexplained**. The profile metadata shows wall-clock ≈ self-time (22.3s ≈ 21.9s), suggesting minimal thread parallelism within the profiled run. Possible causes for the gap:
- Profile data and cargo timing from different builds
- Self-profiling overhead inflating measured time
- Profile metadata measuring broader scope than cargo timing
- Incremental vs clean build differences

This needs further investigation.

## Profile Event Format

Profile data is stored in `.mm_profdata` files, parsed by the `analyzeme` crate.

### Event Structure

```rust
struct Event<'a> {
    event_kind: Cow<'a, str>,      // "Query", "GenericActivity", etc.
    label: Cow<'a, str>,           // "typeck", "borrowck", etc.
    additional_data: Vec<Cow<'a, str>>,  // [0] = DefPath for frontend events
    payload: EventPayload,         // Timestamp::Interval { start, end }
    thread_id: u32,
}
```

### Event Kinds

From `measureme::rustc`:
- `Query` - Query system events (typeck, borrowck, etc.)
- `GenericActivity` - General compiler activities
- `IncrementalResultHashing` - Happens INSIDE queries, shouldn't be counted separately
- `IncrementalLoadResult` - Loading cached results
- `QueryBlocked` - Waiting for another thread

**Key insight**: Only `Query` and `GenericActivity` events should be counted as frontend work. Other event kinds happen inside queries and are already accounted for in the parent's duration.

### Event Stream Order

Events are emitted at their **end time**. For nested events:

```
e3 (innermost):  [30, 40)  - emitted first (ends at 40)
e2 (middle):     [20, 50)  - emitted second (ends at 50)
e1 (outermost):  [10, 60)  - emitted last (ends at 60)
```

Stream order: `[e3, e2, e1]`

To process parents before children, we iterate in **reverse**: `[e1, e2, e3]`

### DefPath in additional_data

Frontend events have the `DefPath` in `additional_data[0]`:

```
event_kind: "Query"
label: "typeck"
additional_data: ["my_crate::foo::bar"]  // DefPath
```

This is how we attribute costs to specific symbols.

## Self-Time Algorithm

The algorithm computes self-time by subtracting child durations from parents.

### Pseudocode

```
for event in events.reverse():
    // Pop events that don't contain current event
    while stack.top() doesn't contain event:
        finalized = stack.pop()
        record(finalized.path, finalized.self_time)

    // Subtract current duration from parent's self-time
    if stack.not_empty():
        stack.top().self_time -= event.duration

    // Push current event with initial self-time = full duration
    stack.push((event, event.duration))

// Finalize remaining events
for event in stack:
    record(event.path, event.self_time)
```

### Example Trace

Events (innermost first in stream):
- e2: typeck, path="bar", [10, 20), duration=10ms
- e1: typeck, path="foo", [0, 30), duration=30ms

Walking in reverse:

1. See e1 [0, 30):
   - Stack empty
   - Push (e1, 30ms)
   - Stack: [(e1, 30ms)]

2. See e2 [10, 20):
   - e1 contains e2, keep e1
   - Subtract: e1.self_time = 30 - 10 = 20ms
   - Stack: [(e1, 20ms)]
   - Push (e2, 10ms)
   - Stack: [(e1, 20ms), (e2, 10ms)]

3. Finalize:
   - Record e2: path="bar", self_time=10ms
   - Record e1: path="foo", self_time=20ms

Result: `{foo: 20ms, bar: 10ms}`, total = 30ms (correct!)

### Per-Thread Stacks

Each thread has its own stack. Events from different threads don't interact (they have disjoint time intervals on their respective threads).

## Implementation in profile.rs

### Key Changes

1. **Event kind filtering**: Only count `Query` and `GenericActivity` events
   ```rust
   if event_kind != QUERY_EVENT_KIND && event_kind != GENERIC_ACTIVITY_EVENT_KIND {
       return EventCategory::Skip;
   }
   ```

2. **Self-time computation**: Implemented the stack-based algorithm in `aggregate_profile()`

3. **Per-thread state**: Maintain separate stacks per `thread_id`

### Demonstration That It Works

Added logging to show raw duration sum vs recorded self-time sum:

```
starting aggregate_profile crate_name="tokio" num_events=1600682
self-time sums raw_ms=145619 self_time_ms=17345
```

- Raw duration: 145.6s
- Self-time: 17.3s
- Ratio: 8.4x reduction

This matches expected nesting inflation (~5-8x depending on codebase).

## The New Bug: Duplicate Accumulation

### Symptoms

The logged self-time for tokio lib is ~17.3s, but the frontend cost in the JSON is ~63s (3.6x higher).

### Evidence from Logs

```
starting aggregate_profile crate_name="tokio" num_events=1600682
...
self-time sums raw_ms=145619 self_time_ms=17345
starting aggregate_profile crate_name="tokio" num_events=1600682   <- SAME profile!
...
self-time sums raw_ms=145619 self_time_ms=17345                    <- SAME numbers!
starting aggregate_profile crate_name="tokio" num_events=1686568   <- Different profile
...
self-time sums raw_ms=155940 self_time_ms=17463
```

The same profile (1,600,682 events) is processed multiple times.

### Hypothesis

The `load_from_dir` function is being called multiple times for tokio:
1. Once for `tokio/lib` target
2. Once for `tokio/test` target
3. Once for each test binary

Each call loads ALL profile files in the directory, including files from previous compilations. The `ProfileData` struct accumulates costs across all profile files it loads.

If the same profile file is loaded multiple times (or the same normalized DefPath appears in multiple profile files), the costs accumulate.

### Math

- First tokio call: 17,345ms
- Second tokio call (same profile): 17,345ms
- Third tokio call (different profile): 17,463ms
- Sum: 52,153ms ≈ 52s

This is close to the 62.8s we see in the JSON. The remaining difference might be from additional profile files.

### Root Cause (Suspected)

The profile directory structure needs investigation. Questions:
1. Are profile files named uniquely per target?
2. Are old profile files being cleaned up between compilations?
3. Is `load_from_dir` being called multiple times for the same target?

### Investigation Update (Feb 2025)

Added tracing spans to debug this issue. Current observations:

1. Each target gets its own profile subdirectory (e.g., `tokio_lib/`, `tokio_test/`)
2. Each directory contains exactly 1 `.mm_profdata` file
3. The self-time reduction ratios look correct (8-10x for lib targets)

However, **we did not reproduce the duplicate accumulation bug** in the current run.
The logs in the original investigation showed the same profile being processed
multiple times, but our tracing-instrumented run shows proper isolation.

Possible explanations:
- The bug was fixed in a previous session
- It only manifests under certain conditions (e.g., incremental builds, specific cargo flags)
- The original logs were from a different version of the code

**Status: Bug not reproduced. Need to investigate further if it recurs.**

## Current R² Results

With the self-time algorithm (but with the duplicate bug):

| Configuration | LIB R² | TEST R² | MERGED R² |
|--------------|--------|---------|-----------|
| Frontend + Backend only | 0.001 | 0.25 | 0.83 |
| + Metadata | 0.008 | 0.49 | 0.85 |

The LIB R² is very poor (0.008) because the modeled costs are ~10x inflated due to the duplicate accumulation bug.

## Next Steps

1. **Fix the duplicate accumulation bug**:
   - Investigate profile directory structure
   - Ensure each profile file is only loaded once
   - Or deduplicate profile files by their unique ID

2. **Validate self-time correctness**:
   - Once duplicates are fixed, verify that modeled costs match actual times better
   - Expected: ~3x inflation (due to parallel frontend), not ~10x

3. **Consider parallel frontend**:
   - The remaining ~3x gap is due to `-Zthreads`
   - For cost modeling, this might be acceptable as a constant factor
   - Or we could try to estimate the parallelism factor

## Files Changed

- `crates/cargo-tarjanize/src/profile.rs`:
  - Added `QUERY_EVENT_KIND` and `GENERIC_ACTIVITY_EVENT_KIND` constants
  - Modified `categorize_event()` to filter by event_kind
  - Rewrote `aggregate_profile()` with self-time algorithm
  - Modified `record_event()` to return recorded duration for debugging
  - Added debug logging for raw vs self-time sums
