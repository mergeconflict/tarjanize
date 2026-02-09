# Critical Path Pruning: Minimal Splits for Optimal Build Time

> **Status**: Design proposal (not yet implemented)
>
> **Problem**: `tarjanize condense` produces thousands of synthetic crates for
> large workspaces. Most splits don't improve the critical path. We need the
> **minimum set of splits** that achieves the optimal critical path.

## Motivation

The current pipeline runs condense on the entire workspace symbol graph,
producing an optimized symbol graph where every SCC boundary becomes a crate
boundary. For a workspace like Omicron (~160 crates), this can recommend
splitting into thousands of synthetic crates — far too many for a human to
reorganize.

But most of those splits don't matter. The critical path (the longest weighted
path through the dependency DAG, assuming infinite parallelism) determines the
minimum possible build time. Splits that only affect crates far from the
critical path contribute nothing to build time improvement. We should only
recommend splits that are **necessary** to achieve the optimal critical path.

## Problem statement

Given the original workspace dependency DAG and the set of per-crate splits
recommended by condense, find the **minimum subset of splits** to apply such
that the critical path equals `T_opt` (the critical path of the fully-split
graph).

The original workspace structure is the baseline. Every split is a deviation
from that baseline, and we want to minimize deviations. We never recommend
merging originally-separate crates — the action space is strictly splits.

### Definitions

- **Original DAG**: the workspace as-is, with critical path `T_orig`
- **Fully-split DAG**: the condense output, with critical path `T_opt ≤ T_orig`
- **Split decision**: for each original crate that condense split into multiple
  pieces, a binary choice — apply the split or leave the crate whole
- `T(S)`: the critical path length when we apply the set of splits `S`
- `D`: the set of all possible splits (one per original crate that condense
  recommends splitting)

**Goal**: find minimum `S ⊆ D` such that `T(S) = T_opt`.

### Monotonicity

Splitting more crates can only decrease or maintain the critical path, never
increase it. Splitting replaces a single heavy node with lighter pieces and
can only add parallelism:

```
If S ⊆ S', then T(S) ≥ T(S')
```

This means `T_opt = T(D)` is the global optimum, and un-doing any split can
only make things worse or keep things equal.

## Over-long paths and the hitting set formulation

The key insight: in the original DAG, some paths are **over-long** — their
total weight exceeds `T_opt`. Each over-long path must have at least one crate
on it split to bring the path weight back under `T_opt`. This is the
**minimum hitting set** problem.

### Why paths matter

When we leave a crate un-split, it retains its original weight (the sum of all
its symbols' compilation costs) and its original edges. The critical path of
the resulting DAG is determined by the longest path, where un-split crates
contribute their full weight and split crates contribute only the weight of
whichever piece is on that path.

An over-long path exists because the crates along it are collectively too
heavy. Splitting at least one crate on the path breaks the heavy node into
lighter pieces, potentially allowing parts of it to compile in parallel with
other work, shortening the path.

### Subtlety: path-dependent savings

Splitting crate `C` doesn't reduce every path through `C` by the same amount.
`C` gets replaced by multiple pieces `{C₁, ..., C_k}`, and different paths
may route through different pieces. A path that only depends on `C₁` (the
heaviest piece) gets little benefit, while a path that depended on all of `C`
but now only needs `C₃` (a light piece) gets a large benefit.

This means the hitting set isn't a simple "does this path contain a split?"
check — we need to verify that the split provides **enough** weight reduction
on each specific path.

### Formulation

For each over-long path `P` in the original DAG, let `excess(P) = weight(P) -
T_opt`. For each crate `C_i` on path `P`, let `savings(C_i, P)` be the weight
reduction on `P` from splitting `C_i`: the difference between `C_i`'s merged
weight and the weight of whichever piece of `C_i` path `P` would route
through after splitting.

```
minimize  Σ xᵢ

s.t.      Σ savings(Cᵢ, P) · xᵢ  ≥  excess(P)    for each over-long path P
          xᵢ ∈ {0, 1}
```

This is a weighted set cover / hitting set variant. NP-hard in general.

## Complexity

### NP-hard in general

The problem is NP-hard because splits interact: two un-split crates on the
same path contribute their full weights additively, so leaving both un-split
can make a path over-long even though leaving either one alone would be fine.

Concrete example. Three original crates X, Y, Z, each split into two pieces
(weight 30 each, merged weight 60). `T_opt = 100`, with the true critical
path elsewhere:

```
         ┌→ Y₁(30) → Q(5)     path weight = 70, slack = 30
X₁(30) ──┤
P(5) →   └→ Z₁(30) → R(5)     path weight = 70, slack = 30
```

| Action | Longest new path | OK? |
|--------|-----------------|-----|
| Un-split X only | `P(5) → X(60) → Y₁(30) → Q(5) = 100` | Yes |
| Un-split Y only | `P(5) → X₁(30) → Y(60) → Q(5) = 100` | Yes |
| Un-split X + Y | `P(5) → X(60) → Y(60) → Q(5) = 130` | **No** |
| Un-split Y + Z | No shared path | Yes |

Un-splitting X or Y individually is safe, but both together create a path of
130. The optimal solution keeps only X split (1 split), un-splitting both Y
and Z. A greedy algorithm that un-splits X first gets stuck at 2 splits.

### Tractable in practice

Despite NP-hardness, real compilation DAGs have favorable structure:

1. **Most crates have large slack.** Only crates near the critical path are
   candidates for over-long paths. The vast majority of crates can be
   trivially ruled out.

2. **Few over-long paths.** The original DAG's paths near `T_opt` length form
   a small set — the critical path is typically a narrow chain, not a wide
   front.

3. **Small candidate set.** After filtering out crates with sufficient slack,
   the remaining candidates (crates that appear on over-long paths and whose
   split provides meaningful savings) should number in the tens, not thousands.

For a small candidate set, the hitting set can be solved exactly via brute
force, ILP, or branch-and-bound.

## Algorithm

### Step 1: identify over-long paths

Compute the critical path of the fully-split graph to get `T_opt`. Then, in
the original DAG (where every crate is at its merged weight), enumerate paths
with total weight > `T_opt`.

We don't need all paths — only **maximal** over-long paths. Use DFS from
sources with pruning: abandon a branch when the accumulated weight plus the
longest possible remaining suffix can't exceed `T_opt`. The longest suffix per
node can be precomputed in O(V + E).

### Step 2: identify candidate splits

For each over-long path, identify which crates on it are splittable (condense
recommended a split) and compute the path-specific savings. A crate that
condense didn't split provides no savings and can be ignored.

A quick filter: for each split crate `C`, compute its best-case `merged_cp`
in the fully-split graph:

```
merged_cp(C) = max(earliest_finish[p] for p ∈ external predecessors of any Cᵢ)
             + Σ w(Cᵢ)
             + max(longest_suffix[s] for s ∈ external successors of any Cᵢ)
```

If `merged_cp(C) > T_opt`, crate `C` **must** stay split — even in the best
case (all other crates fully split), un-splitting `C` alone exceeds `T_opt`.
These are unconditionally necessary and can be excluded from the hitting set
(they're always in the solution).

### Step 3: solve minimum hitting set

With the must-split crates fixed and the remaining candidate crates
identified, solve the hitting set:

```
minimize  Σ xᵢ

s.t.      Σ savings(Cᵢ, P) · xᵢ  ≥  excess(P)    for each over-long path P
          xᵢ ∈ {0, 1}
```

Solver options, in order of preference:

1. **Brute force** (≤ ~20 candidates): enumerate subsets. 2²⁰ ≈ 1M, fast.
2. **Greedy**: repeatedly pick the split that covers the most remaining excess
   across uncovered paths. O(log n) approximation ratio.
3. **ILP**: for larger instances, use an off-the-shelf solver (e.g., the
   `good_lp` crate).

Start with greedy; add exact solving if needed.

### Step 4: output

The final split set is: must-split crates (from step 2) ∪ hitting set crates
(from step 3). Produce a pruned symbol graph that applies only these splits,
leaving all other original crates intact.

## Data flow

```
symbol_graph.json ──→ condense ──→ optimized_symbol_graph.json
       │                                     │
       │         ┌───────────────────────────┘
       ▼         ▼
      prune (needs both graphs)
       │
       ▼
pruned_symbol_graph.json
```

The pruning step needs both graphs:
- The **original** graph provides crate weights and the over-long paths
- The **optimized** graph provides `T_opt` and the per-piece weights after
  splitting

## Open questions

1. **Weight model for merged crates.** Is the merged weight exactly the sum of
   piece weights, or does compiling together have additional overhead? Assume
   sum for now — this is conservative (overestimates merged weight, so we keep
   more splits than strictly necessary).

2. **Over-long path enumeration.** How many over-long paths exist in practice?
   If the number is manageable (hundreds), the hitting set is easy. If
   exponential, we need constraint generation (enumerate paths lazily, adding
   constraints as the solver finds violations).

3. **Path-dependent savings computation.** When crate `C` is split and path
   `P` goes through `C`, which piece does `P` route through? This requires
   tracing the dependency chain through `C`'s pieces — a path that entered via
   a dependency on `C₁` only needs `C₁`, not all of `C`. The savings are
   `w(C) - w(C₁)`.

4. **User experience.** How to present the results:
   - A report: "split these N crates to reduce critical path from X ms to Y ms"
   - A pruned symbol graph (current optimized output with unnecessary splits
     merged back)
   - Ranked by impact (biggest critical path reduction first) for incremental
     application

## References

- [PLAN.md](../PLAN.md) — full project specification
- [docs/cost-model-validation.md](cost-model-validation.md) — cost model
  accuracy and validation methodology
- Minimum hitting set: Karp (1972), one of the original 21 NP-complete problems
