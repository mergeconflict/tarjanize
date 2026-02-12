# Interactive Split Explorer

> **Status**: Design (not yet implemented)
>
> **Problem**: Given a Rust workspace, a fitted cost model, and a build
> schedule visualization, let a developer interactively explore crate
> splits -- seeing the effect on the build schedule in real time -- and
> export the resulting symbol graph.

## Motivation

The [critical path pruning](../critical-path-pruning.md) design proposed
an automated algorithm: enumerate antichains in the intra-target SCC
DAG, greedily pick the cut that best reduces the critical path, repeat.
This optimizes well for build time improvement but struggles with two of
the three stated goals:

1. **Maximizing build time improvement** -- addressed well by greedy
   search
2. **Minimizing splits** -- addressed only implicitly (stop when nothing
   helps)
3. **Retaining semantic coherence** -- not addressed at all

Semantic coherence is hard to automate. An antichain-based search
considers every valid partition of the SCC DAG, most of which are
arbitrary slicings that no developer would recognize as meaningful.
Module boundaries encode developer intent about cohesion, but modules
have cyclic dependencies (see [Data analysis](#data-analysis) below), so
they can't serve as atomic splitting units without symbol-level
refinement.

The alternative: **make the developer the decision-maker**. The tool's
job becomes visualization, validation, and schedule feedback. The
developer knows the semantic domain; the tool shows where splits matter
and enforces structural constraints.

Benefits:

- The developer knows best which symbols belong together
- Interactive exploration is more engaging than reading a plan file
- Trying splits before implementing them reduces wasted effort
- The algorithm reduces to schedule computation (already implemented)
  plus convexity validation

## Data analysis: Omicron workspace

Analysis of the Omicron symbol graph (~160 crates) reveals the structure
that the tool must handle.

### Critical path

The critical path (using modeled costs, R²=1.00) is 175s across 20
targets. No single crate dominates:

| Target                    | Cost  | % of critical path |
|---------------------------|-------|--------------------|
| omicron-nexus/test        | 52.7s | 30%                |
| omicron-nexus/lib         | 46.1s | 26%                |
| nexus-db-queries/lib      | 23.8s | 14%                |
| nexus-db-model/lib        | 22.1s | 13%                |
| remaining 16 targets      | 30.3s | 17%                |

The cost model (`wall = 1.01*attr + 1.05*meta + 1.01*other`) excludes
`nexus-db-queries/lib` as an outlier: its wall-clock time is 462s but
its modeled cost is only 24s. The 438s gap is likely diesel macro
expansion and complex type inference -- overhead that won't scale with
a symbol-level split. Using modeled costs gives a more realistic
picture of what splits can improve.

### Structure of nexus-db-queries/lib

Despite no longer dominating the critical path under modeled costs,
nexus-db-queries/lib is a useful case study because its internal
structure is representative of crates that benefit from splitting. It
has a **hub-and-spoke** structure:

```
Infrastructure (889ms attributed, ~1500 symbols)
  DataStore struct, {{impl}}[2], pagination, collection_*,
  update_and_check, raw_query_builder, cte_utils, ...
          |
          v
~70 domain modules (16,147ms attributed, ~3800 symbols)
  disk, vpc, instance, silo, inventory, deployment, ...
  Each adds methods to DataStore via impl blocks.
  Cross-dependencies are sparse: 114 edges among ~70 modules.
```

The downstream crate `omicron-nexus/lib` mirrors this structure:
`app::disk` uses `db::datastore::disk`, `app::vpc` uses
`db::datastore::vpc`, and so on. Feature columns run through the
dependency chain.

### SCC condensation

At the **symbol level**, the SCC DAG has 5,316 nodes. 5,312 are
singletons; only 4 multi-symbol SCCs exist (largest: 8 symbols). The
dependency graph is essentially acyclic at symbol granularity.

At the **module level**, the story is different. Condensing the 133
modules into SCCs yields:

- **1 large SCC** (45 modules, 2,624 symbols, 12,065ms): `db::datastore`
  (parent) + 43 submodules + 2 query modules. The cycles arise because
  the parent module contains the `DataStore` struct and shared impl
  blocks that both depend on and are depended on by submodule symbols.
- **1 small SCC** (2 modules, 190ms): `multicast` + `multicast::groups`
- **86 singleton modules**: independently assignable

The 45-module SCC can't be split along module boundaries, but it *can*
be split at the symbol level -- the cycles pass through a few hub
symbols, not dense cross-module entanglement.

### Use and ExternCrate items

`use` and `extern_crate` items constitute 70% of symbols (3,730 of
5,327 in nexus-db-queries/lib). Analysis shows they are **never
depended on** by any other symbol -- they have 490 outgoing edges but
zero incoming edges. They encode "module X re-exports symbol Y" but
don't constrain splits.

These items can be dropped entirely from the visualization and split
logic. Their re-export relationships are useful metadata (e.g., for
labeling public API surfaces) but are not structural dependencies.

After dropping Use/ExternCrate items, the graph is ~1,600 nodes across
~133 module clusters.

### Implications for visualization

- The user needs to work at **symbol-level granularity** to make valid
  splits within the large module SCC
- But they want to **think in terms of modules** -- module clustering
  in the visualization bridges this gap
- After dropping Use/ExternCrate items, ~1,600 nodes across ~133
  module clusters -- manageable with force-directed layout and pan/zoom

## Architecture: web app with Rust backend

The `tarjanize viz` command becomes a local web application:

- **Backend** (Rust): serves the symbol graph data, runs SCC
  condensation, computes schedules, validates convexity, applies splits,
  and exports results. All heavy computation stays in Rust.
- **Frontend** (JS): renders the Gantt chart and the target drill-down
  compound DAG. Handles user interaction (crate creation, symbol
  assignment, pan/zoom). Communicates with the backend via a local HTTP
  API.

No `--split` flag is needed. The viz command always starts the web app;
the split explorer is an integrated feature accessible by clicking on
targets.

### Pipeline

```
cargo tarjanize              -> symbol_graph.json
tarjanize viz                -> starts local web app (opens browser)
  user views schedule, explores splits
  user saves result           -> split_symbol_graph.json
tarjanize viz                -> view the split result
```

The backend loads the symbol graph at startup and automatically fits a
cost model from targets that have wall-clock profiling data
(`wall_time_ms > 0`). No separate `tarjanize cost` step is needed.

When viewing a synthetic (split) symbol graph, the model fits against
the unsplit original targets (which still have wall time) and predicts
costs for the split fragments using their structural predictors. Since
only a handful of targets are split in practice, there's always plenty
of training data.

`tarjanize cost` remains available as a standalone diagnostic tool for
inspecting model fit, outlier analysis, and printed reports, but it's
no longer a required step in the pipeline.

When the user creates splits interactively, the backend applies them,
recomputes the schedule, and streams updates to the frontend. When the
user saves, the backend writes the modified `SymbolGraph` directly --
no intermediate plan format. The output is a complete symbol graph with
the split targets, ready for further visualization or analysis.

### Why a web app over self-contained HTML

The current viz generates a self-contained HTML file with all data
embedded. For the split explorer, a web app is better because:

- All graph algorithms (SCC, convexity, schedule DP) stay in Rust --
  no JavaScript ports or WASM compilation needed
- The backend holds the full symbol graph in memory and can apply
  mutations incrementally
- Data doesn't need to be serialized into the HTML (the Omicron symbol
  graph is ~100MB of JSON)
- The backend can validate splits authoritatively before allowing saves

The trade-off is losing the "send someone an HTML file" property. The
existing non-interactive `tarjanize viz` could remain as a static HTML
export for sharing, while the interactive explorer requires a running
backend.

## Interactive split explorer

### Layout: dual-panel

The interactive web app has two panels:

- **Left panel**: Target drill-down view -- the intra-target symbol DAG
  with module clustering
- **Right panel**: Full build schedule Gantt chart (same as current viz)

The Gantt chart updates in real time as the user makes splits. Clicking
a target bar in the Gantt chart opens its drill-down in the left panel.

### Target drill-down: force-directed compound DAG

The left panel shows the symbol-level SCC DAG for the selected target
(with Use/ExternCrate items excluded), rendered as a **force-directed
compound graph**:

- **Symbols** are small nodes sized by attributed cost
- **Modules** are shown as translucent background regions behind their
  symbols. The force simulation positions symbols near their module
  centroid, so module regions emerge naturally. Where modules are
  tightly coupled (sharing hub symbols like `DataStore`), their regions
  will drift close together; the few hub symbols that create
  module-level cycles will position themselves at the boundary between
  the modules they connect.
- **Multi-symbol SCCs** get a visual border indicating "these symbols
  can't be separated"
- **Inter-module edges** are bundled for readability
- Modules with very few symbols can collapse into single nodes,
  expandable on click

The force simulation:

- Symbols are attracted to their module centroid (keeps modules
  cohesive)
- Modules repel each other (prevents overlap in most cases)
- Dependency edges create weak cross-module attraction (related modules
  drift closer)
- Pan/zoom for navigation; node details on hover

Module regions are drawn as translucent backgrounds (not hard
boundaries) because module-level cycles mean perfect non-overlapping
regions aren't feasible. The visual proximity of modules communicates
coupling; the translucent style avoids implying that module boundaries
are hard constraints.

### Split interaction

The user creates named crates and assigns symbols/modules to them:

1. Click **"New Crate"**, type a name (e.g., `nexus-db-queries-storage`)
2. Click module regions or individual symbols to assign them to this
   crate -- they take on the crate's color
3. The tool **enforces convexity**: when the user assigns a set of
   symbols, the tool automatically includes their downward closure (all
   transitive dependencies within the target). Symbols that cannot be
   added without violating convexity are grayed out and unselectable.
   This guarantees the new crate is always a valid downset.
4. The Gantt chart re-renders: the original target bar splits into
   multiple bars, each with its predicted cost. New dependency edges
   appear. The critical path updates.
5. Repeat: create more crates, assign more symbols, see the schedule
   evolve.

Everything not assigned stays in the original crate.

### Why enforce rather than report convexity violations

The [critical path pruning](../critical-path-pruning.md) design
established that valid binary cuts biject with antichains, and that each
antichain defines an upset/downset pair. Allowing invalid assignments
and then reporting violations creates an ambiguous UX: for a cycle
between groups A and B, there's no objective "wrong direction" -- either
side could move its symbols to fix the violation.

By constraining the interaction to valid selections, the tool avoids
this ambiguity entirely:

- Each new crate is carved as a **downset** (downward-closed set) from
  the remaining unassigned symbols
- When the user clicks a symbol, its transitive dependencies within the
  target are auto-included in the new crate
- When the user clicks a module, all symbols in that module are added,
  plus their transitive dependencies (which may pull in symbols from
  other modules)
- After carving one downset, the remaining symbols still form a valid
  DAG, and further downsets can be carved from it
- This guarantees that every intermediate state is a valid split: new
  crates depend on each other (and on the residual) but never in cycles

The downset constraint means the user builds crates "from the bottom
up" -- starting with the foundations and working toward the top. This
matches the natural structure of dependency graphs: infrastructure
first, then the things that depend on it.

### Heat map: guided suggestions

To help the user identify where splits matter most, the tool highlights
the **internal critical path** and computes **improvement potential**:

**Internal critical path**: within the selected target's SCC DAG,
compute the longest weighted path (using attributed costs). This is the
chain of symbols whose sequential compilation determines the target's
minimum possible build time. Highlight these symbols -- they're where
splits have the most leverage.

**Per-SCC slack**: run forward/backward DP on the intra-target SCC DAG,
analogous to the target-level schedule. Color each symbol by its slack:
zero-slack symbols (on the internal critical path) are hot; high-slack
symbols are cool. This tells the user "these symbols are the bottleneck;
splitting on either side of them enables parallelism."

**Improvement potential**: for each SCC on the internal critical path,
the backend tentatively evaluates splitting at that point (computing the
downset as a new crate, recomputing the global schedule). This gives a
concrete "splitting here saves Xms" annotation on the hottest symbols.
The computation is O(K * (V + E)) where K is the number of critical
SCCs within the target and V + E is the global target graph size --
fast enough for the backend to compute on every state change.

### Live schedule recomputation

When the user assigns symbols to a new crate, the backend:

1. Computes the new crate's dependencies from the symbol-level edges
   (which external targets and which sibling split-crates does it depend
   on?)
2. Predicts the new crate's cost from the cost model (`attr` = sum of
   symbol attributed costs; `meta` and `other` distributed
   proportionally -- see [Open questions](#open-questions))
3. Updates the target graph: replaces the original target with the new
   crates and the residual, each with their dependencies
4. Runs the forward/backward schedule DP: O(V + E) on the target graph
5. Pushes updated schedule data to the frontend, which re-renders the
   Gantt chart with updated bars, lanes, and critical path

Steps 1-5 happen on every assignment change. The schedule DP on ~160
targets completes in microseconds; the bottleneck is the frontend
re-render, not computation.

## Output

The interactive tool saves the result as a modified `SymbolGraph` (the
same format as the input `symbol_graph.json`). Split targets appear as
new targets in the output; their symbols, dependencies, and cost
predictors are fully resolved.

This keeps the output format simple and composable: the result can be
fed directly into `tarjanize viz` for viewing, or back into the split
explorer for further refinement. No intermediate plan format is needed
for the initial implementation. A plan format (recording which
modules/symbols moved where, for review or version control) can be
added later if the need arises.

The backend writes the output when the user clicks "Save." The file
includes all original targets (unchanged) plus the split replacements.

## Architectural changes

### Extract `tarjanize-schedule` crate

The scheduling primitives (forward/backward DP, critical path, slack,
`TargetGraph`) currently live in `tarjanize-viz` as `pub(crate)` items.
Both the viz and the split logic need them. Extract into a shared
`tarjanize-schedule` crate:

- `TargetGraph` (target-level DAG with costs)
- `build_target_graph()` (SymbolGraph + CostModel -> TargetGraph)
- `forward_pass()` / `backward_pass()`
- `compute_schedule()` (full scheduling computation)
- `ScheduleData`, `TargetData`, `Summary`

### Extend `tarjanize-viz` into a web app

The viz command starts a local HTTP server (e.g., using axum or warp):

**Backend responsibilities:**
- Load the `SymbolGraph`, fit a `CostModel` automatically, and hold
  both in memory
- Compute intra-target SCC DAGs on demand (when a target is opened)
- Validate and apply split operations (convexity enforcement, downset
  closure)
- Recompute the global schedule after each split
- Compute heat map data (internal critical path, per-SCC slack,
  improvement potential)
- Serve the modified `SymbolGraph` for export

**Frontend responsibilities:**
- Render the Gantt chart (existing PixiJS code, adapted to receive
  data from the backend)
- Render the target drill-down compound DAG (new: d3-force or
  similar, with module clustering)
- Handle user interaction: crate creation, symbol/module assignment,
  pan/zoom, hover details
- Communicate with the backend via JSON API on state changes

**API sketch:**

- `GET /schedule` -- current schedule data (for Gantt chart)
- `GET /target/{name}/graph` -- intra-target SCC DAG with module
  membership and costs (for drill-down)
- `GET /target/{name}/heatmap` -- improvement potential per SCC
- `POST /split` -- apply a split operation (assign symbols to a new
  crate); returns updated schedule
- `POST /unsplit` -- undo a split; returns updated schedule
- `GET /export` -- download the current `SymbolGraph` as JSON

### Remove `tarjanize-condense`

Once the split explorer is complete, the `tarjanize-condense` crate and
the `condense` subcommand are removed. The interactive explorer replaces
automated condensation.

## Open questions

### Cost distribution for split fragments

The cost model predicts wall time from three predictors:
`wall = c_attr * attr + c_meta * meta + c_other * other`. When a target
is split, `attr` distributes naturally (sum of per-symbol attributed
costs). But `meta` (metadata decode time) and `other` (remaining
unattributed time) are per-target measurements, not per-symbol.

Options:

- **Proportional by attr**: distribute `meta` and `other` in proportion
  to each fragment's `attr` share. Simple but assumes overhead scales
  with code volume.
- **Full duplication**: each fragment gets the original target's full
  `meta` and `other`. Conservative (overpredicts) but accounts for
  per-invocation fixed costs.
- **Fixed + proportional**: a fixed per-target overhead (estimated from
  the cost model's intercept behavior) plus proportional scaling for the
  remainder.

This affects prediction accuracy for split fragments and thus the
reliability of the schedule feedback.

### Scale: very large targets

nexus-db-queries/lib has ~1,600 non-use symbols. Force-directed layout
handles this but may be slow on initial load. Options if performance is
an issue:

- Pre-compute layout positions in the backend and serve them
- Use WebGL rendering (PixiJS) instead of SVG for the drill-down
- Progressive loading: render module boxes first, load symbol details on
  expand
- Level-of-detail: at low zoom, show module boxes only; at high zoom,
  show individual symbols

### Coordinated multi-target splits

The current design lets users split one target at a time while seeing
the global schedule impact. For the "feature column" pattern (matching
modules in nexus-db-model, nexus-db-queries, omicron-nexus), the user
would split each target separately, guided by the live schedule
feedback.

A more advanced interaction might show cross-target module alignment:
"these modules in target A correspond to these modules in target B
because they share dependency patterns." This is future work.
