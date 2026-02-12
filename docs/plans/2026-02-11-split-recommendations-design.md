# Split Recommendations Design

> **Status**: Design
>
> **Problem**: Given a Rust workspace with a fitted cost model and build
> schedule, recommend crate splits that improve build parallelism, ranked
> by impact and presented interactively.
>
> **Supersedes**: the antichain enumeration approach in
> `docs/critical-path-pruning.md` for interactive use. That document
> remains relevant for its problem framing, convexity proofs, and the
> `tarjanize split` CLI concept, but the horizon-threshold algorithm
> described here replaces the antichain search for the viz tool.

## Motivation

The previous interactive split explorer showed a force-directed SCC DAG
and asked users to manually select SCCs for extraction. This didn't
scale: large targets like `omicron-nexus/lib` have thousands of SCCs,
and the visualization was unreadable. Worse, it provided no guidance
about which splits actually help.

This redesign replaces manual SCC selection with **computed
recommendations**. The tool identifies valid splits, evaluates their
impact on the build schedule, and presents ranked candidates. The user
decides which splits are worth the refactoring effort.

## Core algorithm: horizon-threshold splits

### Definitions

- **Target**: a compilation unit (`{package}/{target}`). All SCCs in a
  target compile together as one `rustc` invocation.
- **SCC**: a strongly connected component in the intra-target symbol
  dependency graph. SCCs are the atomic unit — they can't be split.
- **External dependency**: a target-level dependency (another crate that
  must finish compiling before this target can start).
- **Effective horizon**: for each SCC, the latest finish time among all
  external targets that the SCC (or any of its predecessors in the SCC
  DAG) depends on.

### Observation

A target T starts compiling at time `max_horizon` — the latest finish
time across all its external dependencies. But not every SCC in T
actually needs every external dependency. Some SCCs only depend on
"early" external targets and could start compiling much sooner if they
were in a separate crate.

The **effective horizon** captures this: it's the earliest time an SCC
*could* start compiling, accounting for both its own external
dependencies and its intra-target predecessors' dependencies.

### Algorithm

For a target T:

1. Compute the intra-target SCC DAG (existing `condense_target`).
2. For each SCC, resolve its symbols' external dependencies to finish
   times from the global schedule. This requires walking the SCC's
   symbols in the `SymbolGraph`, finding dependency paths that reference
   other targets (those not starting with T's `[pkg/target]::` prefix),
   extracting the target identifier from each path, and looking up that
   target's finish time from the forward pass.
3. Propagate effective horizons through the SCC DAG in topological
   order. "Predecessors" here means SCCs that s *depends on* — i.e.,
   following edges in the dependency direction (the source end of
   `IntraTargetGraph` edges, which point from dependency to dependent):
   ```
   effective_horizon(s) = max(
       max(finish(e) for e in s.external_deps),
       max(effective_horizon(pred) for pred in s.predecessors)
   )
   ```
4. `max_horizon = max(effective_horizon(s) for all s)` = T's current
   start time.
5. For each distinct threshold `τ < max_horizon`:
   - Downset = `{s : effective_horizon(s) ≤ τ}` — automatically
     downward-closed because predecessors always have ≤ effective
     horizon.
   - Upset = remaining SCCs.
   - Evaluate the split against the global schedule.

### Why this works

The set `{s : effective_horizon(s) ≤ τ}` is automatically a valid
downward-closed set (a valid downset for a binary cut). This is because
the effective horizon propagates through predecessors: if s depends on
s', then `effective_horizon(s) ≥ effective_horizon(s')`. So any
threshold cut produces a set where all predecessors of included nodes
are also included.

This avoids antichain enumeration entirely. The number of candidate
splits per target equals the number of distinct effective horizon values
below `max_horizon` — bounded by the number of distinct finish times
among the target's transitive external dependencies, not the SCC DAG
width.

Horizon-threshold splits capture all cuts that differ in *when the
downset can start compiling*. Two SCCs with the same effective horizon
always end up on the same side of any threshold cut, which is correct:
separating them provides no scheduling benefit since they can't start
any earlier than each other.

### Degenerate cases

- **All SCCs share the same effective horizon** (including leaf
  targets where every horizon is 0): no thresholds exist below
  `max_horizon`, so no candidates are returned. Parallelism-only
  splits — where both halves start at the same time but compile
  concurrently — are not captured by this algorithm. These could be
  added as a future extension.

### Evaluating a split

For a candidate split at threshold τ:

- **Downset D**: starts at time τ, finishes at `τ + cost(D)`.
- **Upset U**: starts at `max(τ + cost(D), max_horizon)`, finishes at
  `start(U) + cost(U)`.
- **Local improvement**: `finish(T) - max(finish(D), finish(U))`,
  where `finish(T) = start(T) + cost(T)`. How much earlier the two
  halves finish compared to the original target. This is a local
  estimate that assumes the rest of the schedule is unchanged.
- **Global improvement**: recompute the full schedule with T replaced by
  D and U. Difference in critical path length. This is O(V + E) per
  candidate in the target-level DAG.

`cost(D)` and `cost(U)` are cost model predictions for the hypothetical
new targets, not raw sums of SCC costs. The cost model includes
per-target overhead (rustc invocation, linking, etc.), so `cost(D) +
cost(U) > cost(T)` in general. This means some splits are
**detrimental** — the overhead of two smaller targets outweighs the
parallelism gained. Such splits will have negative improvement and are
filtered out.

The local improvement is always meaningful. The global improvement may
be zero for off-critical-path targets.

### Anchor constraints

Impl blocks have "anchors" (the trait and self type). The orphan rule
requires at least one anchor to be in the same crate as the impl. A
split that places an impl block's SCC on one side but both anchors on
the other is not just inconvenient — it's impossible to implement.

The algorithm must enforce this. The existing `condense_target` function
(which feeds the SCC DAG to the frontend) does not add anchor
back-edges — it only computes SCCs from symbol-level dependencies. The
`tarjanize-condense` pipeline handles anchors by adding synthetic
back-edges from the most "niche" anchor to the impl before
condensation, which forces them into the same SCC. For split
recommendations, we need the same treatment: add anchor back-edges
before computing the intra-target SCC DAG, so that impl blocks and
their anchors land in the same SCC and can never be separated by a
threshold cut.

This is a change to `condense_target`: before computing SCCs, add
synthetic back-edges for intra-target anchors the same way
`tarjanize-condense/src/scc.rs` does (step 2b: choose the anchor with
minimum in-degree, add a back-edge from anchor to impl). Only anchors
within the same target matter — external anchors are in a different
crate and don't affect the intra-target DAG.

TODO (carried over from `scc.rs`): the min-in-degree heuristic for
choosing which anchor to use is a rough proxy for "niche." Ideally
we'd use transitive dependents (requires computing reachability).

## API

### `GET /api/splits/{target_id}`

Returns ranked split candidates for a target.

The algorithm implementation lives in `tarjanize-schedule` (alongside
the existing `split.rs` and `target_graph.rs`). The API endpoint is
added to `tarjanize-viz/src/server.rs`.

```json
{
  "target": "nexus-db-queries/lib",
  "current_cost_ms": 4200.0,
  "candidates": [
    {
      "threshold_ms": 850.0,
      "local_improvement_ms": 1400.0,
      "global_improvement_ms": 203.4,
      "downset_scc_count": 43,
      "downset_cost_ms": 1100.0,
      "upset_scc_count": 5273,
      "upset_cost_ms": 3100.0,
      "downset_modules": ["db::model", "db::schema"],
      "upset_modules": ["db::queries", "db::schema", "db::update"],
      "split_modules": ["db::schema"]
    }
  ]
}
```

Fields per candidate:
- `threshold_ms`: the effective horizon threshold defining this cut.
- `local_improvement_ms`: how much earlier the two halves finish
  compared to keeping the target intact.
- `global_improvement_ms`: reduction in critical path length.
- `downset_scc_count`, `downset_cost_ms`: size and predicted cost of
  the extracted crate (cost model prediction with per-target overhead).
- `upset_scc_count`, `upset_cost_ms`: size and predicted cost of the
  remaining crate.
- `downset_modules`, `upset_modules`: which modules have at least one
  SCC in each half (a module appears in a list if *any* of its SCCs
  land on that side).
- `split_modules`: modules that appear in both `downset_modules` and
  `upset_modules` — modules that get divided across both halves,
  requiring the most careful refactoring.

## UI

### Layout

Single-page: Gantt chart in the main panel, sidebar on the left.

### Sidebar states

**No selection** (default): overall schedule stats (total time, critical
path length, target count, etc.).

**Target selected**: user clicks a target in the Gantt chart. Sidebar
shows:
- Target name and current cost.
- Ranked list of split candidates, each showing:
  - Improvement numbers: "saves 203ms critical path, 1,400ms local"
  - Downset summary: "43 SCCs (1,100ms) can start at T=850ms"
  - Split modules warning (if any): "splits db::schema across both
    crates"
- Split candidates are shown for all targets, not just critical-path
  ones. Off-critical-path targets will have zero global improvement but
  may still have meaningful local improvement.

**Split candidate selected**: user clicks a candidate in the list. The
Gantt chart re-renders with the split applied — the original target
replaced by two targets, schedule recomputed. The user immediately sees
the timing impact in context. The sidebar expands the candidate to show
the full module breakdown.

Clicking a different candidate swaps the preview. Clicking away from the
target returns to the original schedule.

### What's removed

The force-directed SCC DAG, the manual split workflow (new crate button,
SCC clicking, confirm/undo), convex hull module visualization. All
replaced by computed recommendations.

## TypeScript migration

The frontend currently uses JavaScript (`.js` files in
`crates/tarjanize-viz/templates/`). As part of this redesign, migrate
to TypeScript:

- Rename `renderer.js` → `renderer.ts`, `logic.js` → `logic.ts`,
  `dag.js` → `dag.ts` (or replace with new split UI code).
- Add `tsconfig.json` to the project root.
- Update `build.rs` to reference `.ts` entry points. esbuild handles
  TypeScript natively; no additional tooling needed.

## Complexity

- Effective horizon propagation: O(V + E) per target (topological sort
  + single pass).
- Number of candidate thresholds: bounded by the number of distinct
  finish times among the target's transitive external dependencies
  (typically tens).
- Schedule evaluation per candidate: O(V + E) in the target graph, with
  upstream caching reducing repeated work.
- Total per API call: O(D * (V + E)) where D = distinct thresholds for
  the requested target, V + E = target graph size.
- Total across all targets: O(T * D * (V + E)) where T = number of
  targets analyzed. Comfortably fast for interactive use.
