# Critical Path Pruning: Minimal Splits for Optimal Build Time

> **Status**: Design (not yet implemented)
>
> **Problem**: Given a Rust workspace and a fitted cost model, find a
> small set of crate splits that significantly improves the critical path
> via parallelism, where each split produces semantically meaningful
> output crates that a developer can reasonably implement.

## Motivation

The current `tarjanize condense` pipeline computes SCCs and partitions the workspace
into fine-grained pieces. For a workspace like Omicron (~160 original
crates, ~180k symbols), this produces ~60-70k synthetic crates -- most
containing only a few symbols. The average SCC is ~3 symbols, and many
are singletons.

This output is not actionable. No developer will reorganize 160 crates
into 60,000 pieces. But most of those boundaries don't matter. The
critical path (the longest weighted path through the dependency DAG,
assuming infinite parallelism) determines the minimum possible build
time. Splits far from the critical path contribute nothing to build time
improvement.

The output should be a practical plan: a small number of
recommended splits that achieve (or nearly achieve) the best possible
critical path. Each recommended split is a refactoring a developer needs
to implement, so fewer splits means less work.

## CLI: `tarjanize split`

This is a new subcommand that replaces `tarjanize condense`. The
pipeline becomes:

```
cargo tarjanize        -> symbol_graph.json
tarjanize cost         -> cost_model.json
tarjanize split        -> split_plan.json
```

The `split` command takes the original symbol graph plus a **required**
cost model, and outputs a set of recommended splits. The cost model is
required because the algorithm is driven by critical path computation,
which depends on accurate per-target cost predictions. The existing
`condense` subcommand and `tarjanize-condense` crate will be removed
once `split` is complete.

## Problem definition

### Definitions

- **Target**: a compilation unit (`{package}/{target}`, e.g.
  `my-pkg/lib`). The cost model and schedule operate at target
  granularity.
- **SCC**: a strongly connected component in the symbol dependency
  graph. Symbols in the same SCC must stay in the same target (they
  have cyclic dependencies).
- **Split**: a target boundary in the output that didn't exist in the
  original workspace. If an original target becomes k output targets,
  that's k-1 splits.
- **Cut**: a binary partition of a target's SCCs into two convex groups
  (see "Validity constraints" below). Each cut adds one split.

### Goal

Given the original workspace DAG (with critical path T_orig) and a
fitted cost model, find a set of cuts that:

1. **Improves the critical path** -- each cut should contribute to
   reducing the predicted build time, primarily by enabling parallelism
2. **Minimizes the number of splits** -- each split is developer work,
   so fewer is better
3. **Produces semantically meaningful groups** -- the output targets
   should correspond to recognizable subsystems or features, not
   arbitrary collections of symbols

These goals are in tension. The ideal algorithm balances all three, but
(1) and (2) are measurable while (3) is harder to quantify.

### Non-monotonicity of splits

The cost model includes per-target fixed overhead (rustc invocation,
linking, etc.). This means not every split improves the critical path:

- Splitting a target increases total work (two invocations instead of
  one) but may decrease the critical path (via parallelism)
- Some splits are **detrimental** -- the overhead of two small targets
  outweighs the parallelism gained
- The fully-split graph (every SCC as its own target) does NOT
  necessarily have the optimal critical path

The algorithm only accepts cuts that improve the schedule, so
detrimental splits are naturally rejected.

## Validity constraints

Not every partition of a target's SCCs is valid. Invalid partitions
create dependency cycles in the output DAG.

### The cycle problem

Consider SCCs P1, P2, P3 from the same original target, where
P1 → P3 → P2 and P1 → P2:

```
P1 -> P2
P1 -> P3 -> P2
```

P1 and P2 share a direct edge, but merging them (skipping P3) creates
a cycle: {P1, P2} depends on P3 (via P1 → P3) and P3 depends on
{P1, P2} (via P3 → P2).

### Convexity

A set of SCCs can be merged if and only if the set is **convex** in
the intra-target DAG: for every pair of members, every SCC on every
path between them must also be included. Equivalently, contracting the
merged set to a single node must not create a cycle.

For a **binary partition** (splitting a target into exactly two groups),
convexity is equivalent to the upset/downset condition: one group must
be an upward-closed set (upset) and the other a downward-closed set
(downset). All inter-group edges flow in one direction, guaranteeing
acyclicity.

For **multi-way partitions**, each group must individually be convex.

### Valid binary cuts biject with antichains

There is a bijection between valid binary cuts and **antichains** of
the intra-target SCC DAG. An antichain is a set of nodes where no two
are comparable (no directed path between any pair). Each antichain
defines a cut boundary:

- The **upset** = the antichain plus everything above it (all nodes
  that transitively depend on any node in the antichain)
- The **downset** = everything below (all remaining nodes)

Enumerating valid binary cuts reduces to enumerating antichains.

**Example: chain** S1 → S2 → S3 → S4. The antichains are the
individual nodes {S1}, {S2}, {S3}, {S4}, giving N-1 = 3 non-trivial
cuts.

**Example: diamond** A → B, A → C, B → D, C → D. Five antichains,
four non-trivial cuts:

| Antichain | Upset   | Downset |
|-----------|---------|---------|
| {A}       | {A}     | {B,C,D} |
| {B}       | {A,B}   | {C,D}   |
| {C}       | {A,C}   | {B,D}   |
| {B,C}     | {A,B,C} | {D}     |
| {D}       | all     | {}      |

### Antichain enumeration algorithm

1. Compute the target's intra-target SCC DAG
2. Compute the transitive closure (makes comparability an O(1) lookup)
3. Enumerate antichains by backtracking: process nodes in a fixed
   order, include or exclude each, prune branches where including a
   node would conflict with (be comparable to) an already-included node
4. For each antichain, compute the upset via upward reachability from
   the antichain members
5. Evaluate the upset/downset split by recomputing the global schedule

### Complexity

The number of antichains can be exponential in the DAG's width (the
size of its largest antichain). For chain-like DAGs -- which we expect
for most targets, since symbol dependencies tend to be roughly linear
-- the count is linear. For wide DAGs, we may need to truncate the
search or use heuristics (e.g., only consider antichains up to some
maximum size).

### Binary cuts vs N-way partitions

The algorithm applies binary cuts greedily: pick the best single cut,
apply it, recompute the schedule, repeat. Any valid N-way convex
partition can be decomposed into a sequence of N-1 binary cuts (there
always exists an antichain separating at least one group from the
rest), so iterated binary cuts can **reach** any partition.

The issue is **greediness**: the binary cut with the best single-step
improvement may not be on the path to the best multi-step partition.
For example, the optimal 3-way split {A}, {B,C}, {D} might require
first cutting {A}|{B,C,D} (a modest improvement), while the greedy
choice picks {A,B}|{C,D} (a larger immediate gain) from which
{A}, {B,C}, {D} is unreachable.

The complexity of enumerating N-way partitions is strictly worse. For
a chain of length n, binary cuts give n-1 options, but N-way partitions
give 2^(n-1) (every subset of cut points). For general DAGs, the
N-way count grows combinatorially on top of the already-exponential
antichain count.

In practice, the greedy binary approach evaluates each cut against the
current global schedule, so later cuts naturally adapt to earlier ones.
If greedy suboptimality turns out to matter on real data, limited
lookahead (evaluating pairs of cuts jointly) is a natural extension.

### Why intra-target merges can't create cycles with external targets

Merging SCCs from the same original target never creates cycles with
external targets. If SCCs C1 and C3 from target T had an external
target E between them (C1 → E → C3), that would imply T → E → T in
the original workspace graph -- a cycle that Cargo forbids. Cycles can
only arise among SCCs of the *same* original target.

## Evaluating cuts: schedule DP

### Forward/backward DP

The critical path and per-node slack are computed in two O(V + E)
passes over the target-level DAG. This is already implemented in
`tarjanize-viz/src/schedule.rs` (`forward_pass()` and
`backward_pass()`), to be extracted into a `tarjanize-schedule` crate:

1. **Forward pass**: process nodes in topological order. For each
   target t, `start[t] = max(finish[dep] for dep in dependencies)` and
   `finish[t] = start[t] + cost[t]`. The maximum finish time across all
   targets is the critical path length.

2. **Backward pass**: process in reverse topological order. For each
   target t, compute `longest_from[t]` -- the longest path from t to
   any sink. Then
   `slack[t] = critical_path - (start[t] + cost[t] + longest_from[t])`.
   Targets with zero slack are on the critical path.

Slack tells us how much a target's compilation can shift without
affecting the overall build time.

### Global effects of splits

A split doesn't just reduce a target's weight -- it restructures the
DAG. The downset half may have fewer external dependencies than the
original target, allowing it to start compiling much earlier. This
creates parallelism: external targets that only need symbols in the
downset no longer wait for the full target to finish.

The true effect of a split on the critical path can only be evaluated
by recomputing the full schedule. Splits interact through the global
schedule: the benefit of splitting target A depends on whether target B
was also split.

## Algorithmic approach

### Greedy binary cuts on critical path targets

Start from the original target-level DAG and apply greedy binary cuts:

1. Compute the critical path on the current target-level graph
2. Identify zero-slack targets, sorted by decreasing predicted
   compilation time (from the cost model)
3. For each zero-slack target (slowest first):
   a. Compute its intra-target SCC DAG (condense just this target's
      symbols). If the target is a single SCC, skip -- it can't be
      split.
   b. Enumerate valid binary cuts via antichains
   c. For each candidate cut, tentatively replace the target with two
      nodes in the target graph, recompute the schedule, record the
      critical path improvement
   d. **Short-circuit**: if the best improvement found so far exceeds
      the predicted compilation time of the next target in the queue,
      stop -- no remaining target can beat it (since improvement is
      bounded by target cost)
4. Apply the best cut across all targets
5. Repeat from step 1 until no cut improves the critical path

**Why cost bounds improvement**: the critical path runs through the
target being split. After splitting, it must still run through at least
one of the two halves (the external dependencies that put the target on
the critical path still exist). So the improvement is at most
`cost(T) - cost(slower half) ≤ cost(T)`.

**DP caching**: when evaluating N candidate cuts for the same target T,
every node upstream of T in topological order has identical finish
times across all candidates. Compute the forward pass up to T once,
then for each antichain only recompute from the split point onward. The
downset half inherits a subset of T's original external dependencies,
so its start time is computed from a smaller set of cached finish
times -- potentially much earlier than T's original start time. This is
exactly the parallelism benefit we're evaluating. The upstream cache
provides the correct inputs; the partial recomputation captures the
reduced dependency set. This turns N full O(V+E) passes into one full
pass plus N partial passes over the downstream subgraph.

### Output

Produce a `split_plan.json` containing the recommended splits. Most
original targets remain unchanged. Targets on the critical path that
benefit from splitting appear as a small number of output targets.

## Limitations and open questions

### Coordinated multi-target splits

The greedy per-target algorithm evaluates one split at a time. This
misses improvements that require coordinated splits across multiple
targets.

Consider a critical path chain: `db-model(10s) → db-queries(10s) →
app(10s)`, total 30s. Splitting `db-model` alone into an upset and
downset doesn't help -- `db-queries` depends on both halves, so it
still waits for the slower half. The greedy search sees zero
improvement and skips the split. But if all three targets are split
along aligned feature boundaries (feature-A symbols vs feature-B
symbols), each feature's chain compiles independently:
`model-A(5s) → queries-A(5s) → app-A(5s)` = 15s, a 50% reduction.
The benefit only appears when multiple targets are split in concert.

### Semantic coherence and clustering

A second concern beyond discoverability: **semantic coherence**. The
developer implementing a split needs to understand it. A cut that
groups "all disk-related types" in one half and "all network-related
types" in the other is actionable. An arbitrary antichain that mixes
concerns is harder to implement and maintain, even if it produces a
shorter critical path.

**Clustering** could address both coordinated splits and semantic
coherence. Community detection on the symbol dependency graph (e.g.,
modularity-based partitioning) identifies groups of tightly-connected
symbols -- which often correspond to features or subsystems. Aligning
clusters across neighboring targets in a chain enables coordinated
multi-target evaluation: split all targets along the same feature
boundary and recompute the schedule once to see the combined effect.

A possible hybrid: use clustering to *propose* candidate cuts (aligned
across the critical path chain), then use the schedule DP to *evaluate*
them. This keeps the rigorous evaluation framework but uses clustering
to navigate the search space instead of blind antichain enumeration.

Not yet designed in detail. Worth revisiting after validating the basic
algorithm on real data.

### Cost model for merged targets

The cost model predicts wall time from structural predictors
(attributed cost, metadata, other). When SCCs merge, the predictors
sum. The cost model's per-target overhead is implicit in its
coefficients. This means merging a chain of tiny targets can reduce
total predicted cost -- which is correct and desirable.

### Optimization framing

The problem can be framed as pure optimization: for each target, choose
one of its valid partitions (precomputed from upset/downset cuts). Any
combination of per-target partition choices produces a valid DAG --
there are no cross-target validity constraints. The challenge is finding
the combination that minimizes total splits while keeping the critical
path short. Known approximation techniques (greedy, LP relaxation,
etc.) may apply, but global schedule interactions between splits
complicate direct application.

### Budget mode

A future `tarjanize split --max-splits N` would output the graph
with at most N splits, choosing the N most impactful ones. The greedy
algorithm naturally supports this: stop after N cuts.

## Architectural changes

### Extract `tarjanize-schedule` crate

The scheduling primitives needed for pruning (forward/backward DP,
critical path computation, `TargetGraph` construction) currently live in
`tarjanize-viz` as `pub(crate)` items. Both viz and the pruning logic
within `tarjanize split` need these. Extract them into a shared
`tarjanize-schedule` crate:

- `TargetGraph` (target-level DAG with costs)
- `build_target_graph()` (SymbolGraph + CostModel → TargetGraph)
- `forward_pass()` / `backward_pass()` (DP for critical path and slack)
- `compute_schedule()` (full scheduling computation)

Both `tarjanize-viz` and the new `tarjanize-split` crate depend on
`tarjanize-schedule`.

## References

- [PLAN.md](../PLAN.md) -- full project specification
- [docs/cost-model-validation.md](cost-model-validation.md) -- cost
  model accuracy and validation methodology
