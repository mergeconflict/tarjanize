# tarjanize

A tool to analyze Rust workspace dependency structures and identify opportunities for splitting crates into smaller, parallelizable units to improve build times.

**Why "tarjanize"?** The tool is named after [Robert Tarjan](https://en.wikipedia.org/wiki/Robert_Tarjan), whose algorithms are central to our analysis:
- **SCC computation** uses Tarjan's algorithm (1972) to identify strongly connected components
- **Union-find merging** uses path compression, whose near-linear time bound was proven by Tarjan & van Leeuwen (1984)

## Motivation

This project is motivated by [oxidecomputer/omicron#8015: Thoughts on improving omicron-nexus build times](https://github.com/oxidecomputer/omicron/issues/8015).

Rust compiles at crate granularity. In large workspaces like Omicron, linear dependency chains between crates cause sequential compilation, leaving most CPU cores idle. For example:

```
omicron-nexus → nexus-reconfigurator-execution → nexus-db-queries → nexus-auth → nexus-db-model
```

Each crate must wait for its predecessor to finish compiling. Additionally, individual crates like `omicron-nexus` are large, making each link in the chain slow.

The issue proposes breaking up crates into smaller, parallelizable pieces. This tool automates that analysis by:

1. Extracting a precise, symbol-level dependency graph across the entire workspace
2. Computing strongly connected components (SCCs) to identify symbols that must stay together
3. Using **union-find merging** to find the optimal partitioning that minimizes critical path cost

The algorithm is simple and provably correct:
- **Acyclicity**: The output is a DAG by construction (chains follow dependency direction)
- **Optimality**: Among all valid partitionings (those that keep SCCs atomic), the resulting partition minimizes critical path cost (the longest sequential compilation chain)
- **Coverage**: Every symbol and dependency is preserved in the output

**Note on critical path vs build time**: Critical path cost is the sum of compilation costs along the longest dependency chain—this is the theoretical minimum build time with unlimited parallelism. Actual build time depends on available cores, crate overhead (rustc invocation, linking), and scheduler efficiency. The algorithm optimizes for the unlimited-parallelism case; with limited cores, fewer larger crates may sometimes outperform many smaller ones due to reduced overhead.

**Usage context**: This tool is designed for infrequent, one-time analysis of a workspace's structure—typically when a codebase has grown large enough that build times are becoming painful. You might run it once to plan a refactoring effort, or periodically (e.g., once a year) to reassess the workspace structure. It is not intended to be run on every commit or as part of continuous integration.

**Benefits of smaller crates**:

1. **Build parallelism**: Independent crates can compile simultaneously, utilizing more CPU cores
2. **Better cache granularity**: Tools like sccache cache at crate granularity. With smaller crates, a code change invalidates a smaller cached unit — developers working on unrelated code still get cache hits for crates they didn't touch
3. **Faster incremental rebuilds**: Changing one file only recompiles its (now smaller) crate and dependents, not a monolithic crate

## Project Setup

Before starting implementation, set up the development environment:

### Rust project

```bash
cargo new tarjanize --bin
```

- [ ] Define JSON Schema file: `symbol_graph.schema.json`
- [ ] Set up test fixtures directory with small sample workspaces

### Continuous Integration

Set up CI (GitHub Actions or similar) to validate the project on every commit:

- [ ] `cargo build` — code compiles
- [ ] `cargo test` — unit and integration tests pass
- [ ] `cargo clippy` — no lint warnings
- [ ] `cargo fmt --check` — code is formatted
- [ ] Run the full pipeline on test fixture workspaces
- [ ] Validate output JSON against schemas

## Pipeline Overview

```
                         ┌─────────────────┐
                         │  Rust Workspace │
                         │   (all crates)  │
                         └────────┬────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 1: Extract Symbol Graph                                    │
│  Input:  <workspace_path>                                         │
│  Output: symbol_graph.json                                        │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 2: Condense and Partition                                  │
│  Input:  symbol_graph.json                                        │
│  Output: optimized_symbol_graph.json (new crate/module structure) │
│  Properties: Acyclicity, Optimality, Coverage, Edge Preservation  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
   Optional: Rename Crates (LLM-assisted)
│  Input:  optimized_symbol_graph.json                              │
   Output: renamed_symbol_graph.json
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 3: Generate Report                                         │
│  Input:  symbol_graph.json, optimized_symbol_graph.json           │
│  Output: report.md (cost improvement and crate relationships)     │
└───────────────────────────────────────────────────────────────────┘
```

## File Formats

Each phase is an independent transformation that reads from input file(s) and writes to output file(s). This enables debuggability (inspect intermediate outputs), resumability (skip phases if inputs unchanged), testability (unit test with fixture files), and composability (swap implementations).

**Error handling philosophy**: Partial failures are preferable to total failures. If the tool encounters invalid data (e.g., an edge referencing a non-existent symbol, a symbol that can't be analyzed), it should:
- Log a warning with context (which symbol, which edge, why it failed)
- Skip the problematic element and continue processing
- Include a summary of skipped elements in the output
- Only fail completely if the error makes further processing meaningless (e.g., input file doesn't exist, JSON is unparseable)

This approach produces useful results even when rust-analyzer can't fully analyze every symbol (common with macros, proc-macros, or incomplete code).

Intermediate files use **JSON** with **JSON Schema** validation (schema files use the `.schema.json` suffix). JSON is human-readable, has ubiquitous tooling (jq, editors), and excellent serde support. JSON Schema catches malformed files early, serves as documentation, and enables editor autocomplete. For very large workspaces, we may later add `--format messagepack`.

**One schema** covers both input and output of the main algorithm:

| Schema | Description | Producers | Consumers |
|--------|-------------|-----------|-----------|
| SymbolGraph | Full symbol graph with crate/module hierarchy | Phases 1 and 2 | Phases 2 and 3 |

The final output (Phase 3) is a **Markdown report** (`report.md`). See Phase 3 for details.

**Internal representations**: Phase 2 uses internal data structures (petgraph's `DiGraph`, union-find, etc.) for SCC computation and partitioning. These are not serialized; use `RUST_LOG=debug` or `RUST_LOG=trace` to inspect intermediate state.

### SymbolGraph

Contains all symbols from all crates in the workspace, along with all dependency edges (both within and across crates).

#### Schema (`schemas/symbol_graph.schema.json`)

The structure mirrors Rust's hierarchy: workspace → crates (as root modules) → modules (nested) → symbols.

```
SymbolGraph
├── workspace_name: string
├── crates: []                         # Each crate is represented as its root Module
│   └── Module
│       ├── name: string               # Crate name (from Cargo.toml) for root modules
│       ├── symbols: []
│       │   └── Symbol (union type)
│       │       ├── name: string
│       │       ├── file: string
│       │       ├── cost: number
│       │       └── oneOf:
│       │           ├── module_def: { kind, visibility? }
│       │           └── impl: { anchors? }
│       └── submodules?: [] (recursive Module)
└── edges: []
    └── Edge
        ├── from: string (path)
        └── to: string (path)
```

**Symbol identity**: No explicit ID field. Identity is implicit from position in the
module tree. Edges use full paths like `crate::module::Name`. For impls, use
`impl Trait for Type` or `impl Type` (inherent).

**Visibility**: The `visibility` field is optional. Absent means private (no visibility
modifier in source). Present values include `pub` and `pub(restricted)` (covers
`pub(crate)`, `pub(super)`, `pub(in path)`).

**Impl fields**: For `impl<P..> Trait<T1..=Tn> for T0`:
- `anchors`: Set of fully qualified paths for workspace symbols that can satisfy the orphan rule. Includes the self type T0, the trait, and any trait type parameters T1..Tn that are workspace symbols. Omitted if empty.

Phase 2 uses these to track orphan rule constraints. The orphan rule requires the impl
to be in the same crate as at least one anchor. We don't need to distinguish between
the self type, trait, and trait type parameters—any of them can satisfy the rule.

See `schemas/symbol_graph.schema.json` for the complete JSON Schema definition.

## CLI Interface

The tool can run the full pipeline in one command, or execute individual phases for debugging and incremental workflows.

```bash
# Run full pipeline on a workspace
tarjanize analyze /path/to/omicron --output-dir ./analysis

# Run individual phases
cargo tarjanize /path/to/omicron -o symbol_graph.json                      # Phase 1
tarjanize condense symbol_graph.json -o optimized_symbol_graph.json        # Phase 2
tarjanize report symbol_graph.json optimized_symbol_graph.json -o report.md  # Phase 3

# Optional: use LLM to suggest better crate names
tarjanize rename optimized_symbol_graph.json -o renamed_symbol_graph.json

# Restrict analysis to specific crates (allowlist)
tarjanize analyze /path/to/omicron --include nexus-db-queries,omicron-nexus --output-dir ./analysis

# Exclude specific crates from analysis (denylist)
tarjanize analyze /path/to/omicron --exclude test-utils,dev-tools --output-dir ./analysis
```

The `--include` and `--exclude` options are also available on the `extract` command for single-phase usage.

**Conditional compilation**: By default, the tool analyzes all code regardless of `#[cfg(...)]` attributes. Use these flags to control which configurations are analyzed:

```bash
# Analyze with specific features enabled
tarjanize analyze /path/to/workspace --features feature1,feature2

# Analyze with all features enabled
tarjanize analyze /path/to/workspace --all-features

# Analyze only for a specific target
tarjanize analyze /path/to/workspace --target x86_64-unknown-linux-gnu
```

The resulting dependency graph reflects whichever configuration is analyzed. Different configurations may produce different optimal partitionings.

**Note on test code**: Test code is included by default (via `--all-targets`) and treated the same as production code. See "Test code handling" in Phase 2 for the rationale.

## Phase 1: Extract Symbol Graph

**Input**: Path to Rust workspace (directory containing workspace `Cargo.toml`)
**Output**: `symbol_graph.json`

### Workspace loading

1. Parse workspace `Cargo.toml` to discover all member crates
2. Load all crates into rust-analyzer's analysis database simultaneously
3. This enables rust-analyzer to resolve cross-crate references

### Symbols to track

We track all symbols exposed by rust-analyzer's HIR (High-level Intermediate Representation), which provides a fully resolved view of the code after macro expansion.

**Excluded from analysis**:
- **Build scripts** (`build.rs`): These are separate compilation units that run before the crate compiles. They have their own dependency graph and are orthogonal to the main crate structure.

**Note on procedural macro crates**: Proc-macro crates are analyzed like any other crate. While the `#[proc_macro]` entry points must remain in a proc-macro crate, the implementation can be split into helper library crates — a common pattern for improving compile times.

**Macro handling**: rust-analyzer expands macros before building the HIR, so dependencies from expanded code are captured automatically. However, we must also record an explicit dependency on the macro definition itself — if a macro changes, all call sites need recompilation regardless of what the expansion contains. See [rust-analyzer's hir_expand documentation](https://docs.rs/ra_ap_hir_expand/latest/ra_ap_hir_expand/) for details on how macro expansions are tracked.

### Dependency types to capture

Examples of dependencies to track (non-exhaustive):

| Dependency Type | Example | Edge |
|-----------------|---------|------|
| Function call | `foo()` calls `bar()` | `foo → bar` |
| Type usage | `fn foo(x: Bar)` | `foo → Bar` |
| Field type | `struct Foo { b: Bar }` | `Foo → Bar` |
| Trait bound | `fn foo<T: Display>()` | `foo → Display` |
| Trait impl | `impl Display for Foo` | `impl → Display`, `impl → Foo` |
| Method call | `x.method()` | `caller → impl` |
| Associated type | `T::Type` | `user → impl` |

**Collapsing to containers**: We record dependencies on containers, not contained items:
- Impl methods/consts/types → collapse to the impl block
- Trait methods/consts/types → collapse to the trait
- Enum variants → collapse to the enum

This ensures items that must stay together (impl contents, trait contents) are treated atomically.

### Implementation approach

Use rust-analyzer crates:
- `ra_ap_vfs` — virtual file system that provides file contents to the analysis engine
- `ra_ap_ide` — high-level analysis API for loading workspaces and querying semantic information
- `ra_ap_hir` — typed API over rust-analyzer's High-level Intermediate Representation (symbol definitions, types, trait impls)

**rust-analyzer data model**:
- `ModuleDef` — items at module scope: Module, Function, Adt (struct/enum/union), Variant, Const, Static, Trait, TraitAlias, TypeAlias, BuiltinType, Macro
- `Impl` — NOT a ModuleDef; anonymous (no path), accessed via `Module::impl_defs(db)`
- `AssocItem` — lives inside impls/traits; check with `item.as_assoc_item(db)`, get container with `container(db)` → `Trait` or `Impl`

**Module enumeration** (three sources per module):
1. `declarations(db)` → ModuleDefs (structs, fns, traits, etc.)
2. `impl_defs(db)` → Impl blocks
3. `legacy_macros(db)` → `macro_rules!` macros

Skip: `extern_crate_decls`, `use_decls`, `scope()` (re-exports/imports don't create dependencies).

**Resolution APIs**:
| Syntax | API | Returns |
|--------|-----|---------|
| `x.method()` | `resolve_method_call()` | Function |
| `Type::new()` | `resolve_path()` | Function |
| `Type::CONST` | `resolve_path()` | Const |
| `<T as Trait>::Item` | `resolve_path()` | TypeAlias |

All resolved items should be checked with `as_assoc_item(db)` to collapse to their container.

**Local vs foreign**: Use `CrateOrigin::is_local()` to distinguish workspace members from external crates. Only track dependencies on local items; foreign deps are filtered out.

Steps:
1. Load the workspace into rust-analyzer's analysis database (all crates at once)
2. For each crate, recursively traverse modules via `children(db)`
3. For each module, enumerate symbols from `declarations()`, `impl_defs()`, and `legacy_macros()`
4. For each symbol, walk its syntax tree resolving paths and method calls to find dependencies
5. Collapse associated items to their containers; filter to local deps only
6. Build edges using full paths as identifiers

### Implementation tasks (Rust only)

- [x] Integrate rust-analyzer crates (`ra_ap_ide`, `ra_ap_hir`, `ra_ap_vfs`)
- [x] Implement workspace discovery from root `Cargo.toml`
- [x] Load all workspace crates into rust-analyzer's database
- [x] Implement symbol enumeration across all crates
- [x] Implement dependency edge detection, including cross-crate edges
- [x] Serialize output to `symbol_graph.json`
- [x] Write tests against fixture workspaces

**Known limitation**: Derive macros (`#[derive(...)]`) are not captured due to `sema.resolve_derive_macro()` returning None. Workspace-local derive macros are rare, so this is acceptable.

## Phase 2: Condense and Partition

**Input**: `symbol_graph.json`
**Output**: `optimized_symbol_graph.json`

This phase is the core of the algorithm. It computes strongly connected components (SCCs), merges them optimally using union-find, and produces a new `SymbolGraph` with the optimized crate structure. All intermediate representations (petgraph structures, union-find state) are internal and not serialized.

### Overview

1. **Build graph**: Parse `SymbolGraph` into petgraph's `DiGraph`
2. **Condense**: Run petgraph's `condensation()` to find SCCs
3. **Transitive reduction**: Compute transitive reduction to identify effective dependents
4. **Partition**: Use union-find to merge SCCs into optimal crate groupings
5. **Fix anchors**: Solve global hitting set for orphan rule constraints
6. **Build output**: Construct new `SymbolGraph` with merged crates

### Why SCCs matter

Symbols within an SCC have cyclic dependencies and **must** stay in the same crate — splitting them would create a crate cycle, which Cargo forbids. By condensing SCCs into single nodes, we guarantee that any partitioning of the condensed graph produces a valid (acyclic) crate graph.

**Note**: SCCs are always contained within a single crate. Cross-crate cycles are impossible because Cargo enforces acyclic crate-level dependencies. If crate A depends on crate B, symbols in B cannot reference symbols in A, so no SCC can span both.

### Test code handling

**Design decision**: Test code (`#[cfg(test)]` modules, `#[test]` functions) is treated identically to production code. No special handling is needed.

**Why this works**: The concern was that dev-dependencies might create artificial cycles. For example, if crate A depends on crate B, and B's tests use A as a dev-dependency, this appears cyclic at the crate level. However, at the symbol level:

```
b_test -> a -> b
   |           ^
   +-----------+
```

There is no cycle because **production code never depends on test code**. `#[cfg(test)]` items are invisible to production code, so you can never complete a cycle through test code. The edge `b -> b_test` cannot exist.

**Implication**: The algorithm may recommend test-only crates (crates containing only test code). This is a legitimate recommendation—test utility crates exist in practice (`foo-testutils`, shared fixtures, etc.). If a user prefers to keep tests with their production code, that's a policy decision they can apply when interpreting the recommendations.

### Formal properties

The algorithm guarantees these properties:

| Property | Description | How guaranteed |
|----------|-------------|----------------|
| **Acyclicity** | Output crate graph is a DAG | Merges follow dependency direction; condensation eliminates cycles |
| **Optimality** | Minimum critical path cost among valid partitionings | SCCs merge when all dependents are in the same crate |
| **Coverage** | Every symbol appears in exactly one output crate | Union-find assigns each SCC to exactly one set |
| **Edge Preservation** | All symbol dependencies preserved | Edges unchanged; only crate groupings change |

### Internal data structures

Phase 2 works with petgraph's `DiGraph` and parallel vectors rather than serialized JSON:

```rust
// After condensation: DiGraph<Vec<usize>, ()>
// Each node is an SCC containing symbol indices
let condensed: DiGraph<Vec<usize>, ()> = condensation(graph, true);

// Parallel vectors indexed by SCC node index
let scc_costs: Vec<f64>;           // Sum of symbol costs per SCC
let scc_anchors: Vec<Vec<u32>>;    // Anchor SCC indices per impl

// Symbol index for path ↔ usize conversion
let index: IndexVec<SymbolPath>;

// Union-find for merging SCCs into crates
let uf: UnionFind<u32>;
```

### Algorithm details

#### Step 1: Build graph and condense

1. Parse `SymbolGraph` into an index mapping symbol paths to `usize`
2. Build `DiGraph<(), ()>` with symbols as nodes, dependencies as edges
3. Run petgraph's `condensation()` → `DiGraph<Vec<usize>, ()>`
4. Compute `scc_costs[i]` = sum of costs for symbols in SCC i
5. For each impl, collect anchor SCC indices into `scc_anchors[i]`

#### Step 2: Precompute reachable sets (HMR preparation)

We use the Habib-Morvan-Rampon (HMR) algorithm to determine which dependencies are redundant. HMR computes the transitive reduction by tracking **reachable sets** — for each SCC, the set of all SCCs reachable from it (directly or transitively).

**Why reachable sets?** An edge X → Y is redundant iff Y is already reachable through some other dependency of X. By maintaining reachable sets, we can test redundancy in O(1) time.

**Precomputation:** We compute `reachable[scc]` for each SCC by processing in reverse topological order (leaves first, roots last):

```rust
// Process SCCs from leaves to roots
for scc in reverse_topo_order {
    reachable[scc] = {scc};
    for dep in sorted_dependencies(scc) {  // sorted by topo order
        if dep ∉ reachable[scc] {
            reachable[scc] = reachable[scc] ∪ reachable[dep];
        }
        // else: dep is redundant (already reachable through another dependency)
    }
}
```

**Example:**
```
        B
        │
        ↓
 A ───→ C ───→ D
 │             ↑
 └─────────────┘
```

Processing order: D, C, B, A

| SCC | Dependencies | Reachable set computation | Final reachable |
|-----|--------------|---------------------------|-----------------|
| D   | (none)       | {D}                       | {D}             |
| C   | D            | {C} ∪ reachable[D] = {C,D}| {C,D}           |
| B   | C            | {B} ∪ reachable[C] = {B,C,D}| {B,C,D}       |
| A   | C, D (sorted)| {A}, C∉{A}→{A,C,D}, D∈{A,C,D}→skip | {A,C,D} |

For A, the edge A→D is redundant because after processing A→C, D is already in reachable[A].

#### Step 3: Union-find merging with incremental HMR

**Definitions:**
- **dependents(X)**: SCCs that directly depend on X (have edges pointing to X in the original condensed graph)
- **effective_dependents(X)**: dependents whose edges to X are not redundant — the minimal dependents under reachability

**Key insight — set-level transitive reduction**: The transitive reduction can change when SCCs merge. Consider:

```
            B
            │
            ↓
     A ───→ C ───→ D
     │             ↑
     └───→ E ──────┘
```

Initially, there are no redundant edges. But when E merges into A (its only dependent), the merged set {A,E} has edges to both C and D. At the **set level**, {A,E}→D is now redundant because {A,E}→C→D exists.

**Algorithm — incremental HMR:**

1. **Precompute `reachable[scc]`** for each SCC (see Step 2)
2. **Initialize union-find**: each SCC starts in its own singleton set
3. **For each set, track**:
   - `set_sccs`: the SCCs that have been merged into this set
   - `set_external_deps`: external dependencies (SCCs outside the set)
   - `set_reachable`: reachable set for this merged set
4. **Process SCCs in topological order** (dependents before dependencies):
   - **Terminology**: A "root" is an SCC with no dependents; a "leaf" has no dependencies. Process roots toward leaves.
   - If **zero dependents**: no-op (SCC is a root)
   - If **all dependents in same set** (call it S):
     - **Merge this SCC into S** using union-find
     - **Update set_external_deps[S]**: add this SCC's external dependencies
     - **Recompute set_reachable[S]** by applying HMR to set_external_deps[S]:
       ```
       set_reachable[S] = set_sccs[S]
       for dep in set_external_deps[S] sorted by topo order:
           if dep ∉ set_reachable[S]:
               keep edge S→dep
               set_reachable[S] = set_reachable[S] ∪ reachable[dep]
           else:
               redundant, remove from set_external_deps[S]
       ```
   - If **dependents in different sets**: no-op (SCC is a boundary)
5. **When checking if an SCC X is a boundary**:
   - For each dependent set S, check if S→X is in set_external_deps[S] (i.e., not redundant at the set level)
   - X's **effective dependent sets** are only those where S→X is not redundant

**Worked example** (same graph as above):

Processing order: A, B, E, C, D

| Step | SCC | Dependents | Sets | Action | set_external_deps after |
|------|-----|------------|------|--------|-------------------------|
| 1 | A | ∅ | — | root | {A}: {C, E} |
| 2 | B | ∅ | — | root | {B}: {C} |
| 3 | E | A | {A} | merge | {A,E}: HMR({C,E,D}) = {C,E}* |
| 4 | C | A,B | {A,E},{B} | boundary | unchanged |
| 5 | D | E,C | {A,E},{C} | check set deps | {A,E}→D redundant**, {C}→D not |

\* HMR on {C,E,D}: sort by topo → C,E,D. reachable={A,E}. C∉{A,E}→keep,reachable={A,E,C,D}. E∈{A,E}→skip (internal). D∈{A,E,C,D}→redundant!

\*\* At step 5, we check each dependent's set_external_deps. {A,E}'s deps are {C,E}, not D. So {A,E}→D doesn't exist at the set level.

Result: D's only effective dependent set is {C}, so D merges into {C}. Final: **{A,E}, {B}, {C,D}**

**Time complexity**: O(V × E) for precomputing reachable sets, plus O(E · α(V)) for the merge loop with set reachable updates. Total: O(V × E).

#### Step 4: Fix anchor constraints (global hitting set)

Each impl has an **anchor set**: SCCs containing workspace-local types/traits that can satisfy the orphan rule. The orphan rule requires each impl to end up in the same crate as at least one anchor.

**Why postprocessing works**: An impl always depends on its anchors, so anchor SCCs are processed after the impl's SCC. In most cases, the anchor naturally merges into the impl's set. Violations only occur when an anchor becomes a "boundary" SCC.

**Global hitting set algorithm:**

1. Collect **all** unsatisfied anchor constraints across the entire DAG
2. Solve a **single global weighted hitting set problem**:
   - Universe: union-find sets containing at least one anchor
   - Sets to hit: each unsatisfied constraint becomes a set of union-find sets
   - Weight: total profiled compilation cost of each union-find set
   - Goal: find minimum-weight collection satisfying all constraints
3. Perform merges using the union-find structure

**Why global optimization**: A per-set greedy approach can miss sharing opportunities. If sets S1 and S2 both accept anchor X, global optimization merges once instead of twice.

**Complexity**: NP-hard in general, but tractable here because most impls have 1-2 anchors and most constraints are satisfied naturally.

**Known issue — anchor constraints can create cycles:** The current algorithm (dependent-based merging followed by anchor constraint merging) can produce cyclic partition dependencies. Consider:

```
        B
        │
        ↓
  A ───→ C ───→ D
  │             ↑
  └── impl_D ───┘ (anchor: D)
```

Where A depends on C and impl_D, B depends on C, C depends on D, and impl_D depends on D with anchor constraint to D.

During dependent-based merging, impl_D merges into A's set (its only dependent). Then C and D become boundaries. During anchor constraint merging, D must join impl_D's set (now {A, impl_D}), producing:

- {A, impl_D, D}
- {B}
- {C}

This creates a cycle: {A, impl_D, D} → {C} (A depends on C) and {C} → {A, impl_D, D} (C depends on D).

The expected result is {A}, {B}, {C, D, impl_D} — three crates forming a valid DAG. The solution likely involves either:
1. Performing anchor merging before dependent-based merging
2. Constraining dependent-based merging to not merge impls into non-anchor sets
3. Some integrated approach

This remains an open problem. See `test_anchor_constraints_can_create_partition_cycles` in `scc.rs` for a minimal reproducer.

#### Step 5: Build output SymbolGraph

1. **Group symbols by union-find set**: Each set becomes a new crate
2. **Detect conflicts**: Multiple symbols with same (module_path, name) from different original crates
3. **Assign final paths**:
   - Non-conflicting: original module path
   - Conflicting: module path + `conflict_from_{original_crate}`
4. **Compute visibility**: Cross-crate edges → target must be `pub`
5. **Generate crate names**: Deterministic placeholder `{original-crate}-{id}`
6. **Build nested module trees** and serialize to `optimized_symbol_graph.json`

#### Conflict detection and resolution

When symbols from different original crates are merged into a single new crate, name conflicts can occur. Here's a worked example:

**Input state:**
```
symbol_graph.json:
  crate_a:
    foo/bar.rs: fn x(), fn y()
  crate_b:
    foo/bar.rs: fn x(), fn z()

After union-find merging:
  crate_001: [SCC containing all of the above]
```

**Conflict detection:**
```
(["foo", "bar"], "x") -> [crate_a::x, crate_b::x]  // CONFLICT
(["foo", "bar"], "y") -> [crate_a::y]              // ok
(["foo", "bar"], "z") -> [crate_b::z]              // ok
```

**Final placements:**
```
x from crate_a -> ["foo", "bar", "conflict_from_crate_a"]
x from crate_b -> ["foo", "bar", "conflict_from_crate_b"]
y from crate_a -> ["foo", "bar"]
z from crate_b -> ["foo", "bar"]
```

**Output module tree:**
```
lib
└── foo
    └── bar
        ├── symbols: [y, z]
        └── submodules:
            ├── conflict_from_crate_a
            │   └── symbols: [x]
            └── conflict_from_crate_b
                └── symbols: [x]
```

This approach preserves all symbols while avoiding name collisions. The `conflict_from_*` modules make it clear where each conflicting symbol originated.

### Visualizing union-find merging

Consider this dependency graph where each letter represents an SCC (arrows point from dependent to dependency):

```
              C
            ↗
    A ──→ B
            ↘
              D
```

Processing top-down (roots to leaves): A, B, C, D

| Step | SCC | Dependents | Their sets | Action | Sets after |
|------|-----|------------|------------|--------|------------|
| 1 | A | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 2 | B | A | {A} | union into {A} | {A,B}, {C}, {D} |
| 3 | C | B | {A,B} | union into {A,B} | {A,B,C}, {D} |
| 4 | D | B | {A,B,C} | union into {A,B,C} | {A,B,C,D} |

Result: **One crate containing {A, B, C, D}**

Now consider a graph where C has multiple dependents in different sets:

```
    A
      ↘
        C ──→ D
      ↗
    B
```

| Step | SCC | Dependents | Their sets | Action | Sets after |
|------|-----|------------|------------|--------|------------|
| 1 | A | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 2 | B | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 3 | C | A, B | {A}, {B} | no-op (boundary) | {A}, {B}, {C}, {D} |
| 4 | D | C | {C} | union into {C} | {A}, {B}, {C,D} |

Result: **Three crates: {A}, {B}, {C, D}** — allows parallel compilation.

Now consider a diamond pattern:

```
          B
        ↗   ↘
      A       D
        ↘   ↗
          C
```

Here A depends on both B and C, and both B and C depend on D.

Processing top-down: A, B, C, D

| Step | SCC | Dependents | Their sets | Action | Sets after |
|------|-----|------------|------------|--------|------------|
| 1 | A | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 2 | B | A | {A} | union into {A} | {A,B}, {C}, {D} |
| 3 | C | A | {A,B} | union into {A,B} | {A,B,C}, {D} |
| 4 | D | B, C | both in {A,B,C} | union into {A,B,C} | {A,B,C,D} |

Result: **One crate containing {A, B, C, D}**

The algorithm recognizes that D's dependents (B and C) have already been unified into the same crate, so merging D doesn't force any additional waiting.

Now consider what happens when there's an external dependent E:

```
          B
        ↗   ↘
      A       D
        ↘   ↗ ↑
          C   │
              │
          E ──┘
```

Processing top-down: A, B, C, E, D

| Step | SCC | Dependents | Their sets | Action | Sets after |
|------|-----|------------|------------|--------|------------|
| 1 | A | none | — | no-op (root) | {A}, {B}, {C}, {D}, {E} |
| 2 | B | A | {A} | union into {A} | {A,B}, {C}, {D}, {E} |
| 3 | C | A | {A,B} | union into {A,B} | {A,B,C}, {D}, {E} |
| 4 | E | none | — | no-op (root) | {A,B,C}, {D}, {E} |
| 5 | D | B, C, E | {A,B,C}, {E} | no-op (boundary) | {A,B,C}, {D}, {E} |

Result: **Three crates: {A, B, C}, {D}, {E}**

Here D remains separate because its dependents are in different sets. This preserves parallelism: `{A,B,C}` and `{E}` can compile in parallel after `{D}` finishes.

Now consider a graph with a redundant dependency edge:

```
        B
        │
        ↓
 A ───→ C ───→ D
 │             ↑
 └─────────────┘
```

Here A depends on both C and D directly, but C also depends on D. The A→D edge is redundant from a parallelism perspective: A cannot compile until C is built, and C cannot compile until D is built.

Without effective dependents (treating all dependents equally):

| Step | SCC | Dependents | Their sets | Action | Sets after |
|------|-----|------------|------------|--------|------------|
| 1 | A | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 2 | B | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 3 | C | A, B | {A}, {B} | no-op (boundary) | {A}, {B}, {C}, {D} |
| 4 | D | A, C | {A}, {C} | no-op (boundary) | {A}, {B}, {C}, {D} |

Result: **Four crates: {A}, {B}, {C}, {D}** — suboptimal, D is unnecessarily separate.

With effective dependents:

At step 4, D's direct dependents are {A, C}. Since A →* C (A transitively depends on C), A is dominated by C. So effective_dependents(D) = {C}.

| Step | SCC | Effective Dependents | Their sets | Action | Sets after |
|------|-----|----------------------|------------|--------|------------|
| 1 | A | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 2 | B | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 3 | C | A, B | {A}, {B} | no-op (boundary) | {A}, {B}, {C}, {D} |
| 4 | D | C | {C} | union into {C} | {A}, {B}, {C,D} |

Result: **Three crates: {A}, {B}, {C, D}** — optimal.

### Crate naming

Phase 2 generates **stable placeholder names** using a deterministic algorithm: `{original-crate}-{top-module}` or `{original-crate}-{id}`. For more meaningful names, use the separate `tarjanize rename` command which uses an LLM.

### Implementation tasks

- [x] Implement symbol graph parsing and index construction
- [x] Implement condensation using petgraph
- [ ] Implement transitive reduction using `petgraph::algo::tred`
- [ ] Build reverse adjacency list for effective dependents lookup
- [ ] Implement union-find merge loop using effective dependents
- [ ] Implement anchor constraint collection
- [ ] Implement global hitting set solver (greedy heuristic)
- [ ] Implement conflict detection for merged crates
- [ ] Implement visibility computation
- [ ] Implement module tree construction
- [ ] Implement crate name generation
- [ ] Serialize output to `optimized_symbol_graph.json`
- [ ] Add debug/trace logging at key points

### Rename command (optional LLM naming)

- [ ] Implement `tarjanize rename` subcommand
- [ ] Implement LLM-based crate naming with Anthropic API
- [ ] Load existing symbol graph, apply new names, write output

## Phase 3: Generate Report

**Input**: `symbol_graph.json`, `optimized_symbol_graph.json` (or `renamed_symbol_graph.json` if rename was used)
**Output**: `report.md`

The report is a human-readable Markdown summary of the optimization results. Phase 3 uses whatever crate names are present in its input file—placeholder names from Phase 2, or LLM-suggested names from the optional rename step. It is designed for understanding and communicating the benefits of the proposed split, not for driving automated implementation.

### Report contents

The report contains two main sections:

**1. Cost improvement summary**

A table comparing the original and optimized crate structures:

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Crate count | 5 | 8 | +3 |
| Critical path cost | 4490 | 2900 | 35% faster |

**2. Crate relationships**

A visualization showing which original crates contribute symbols to which new crates. This is a many-to-many relationship (a single original crate may be split across multiple new crates, and a new crate may contain symbols from multiple original crates in the case of merges):

```
Original                    Optimized
────────                    ─────────
nexus-db-queries ─────┬───→ nexus-db-queries-core
                      └───→ nexus-db-queries-silo
omicron-nexus ────────────→ omicron-nexus
```

### Implementation

- [ ] Compute critical path costs for original and optimized graphs
- [ ] Build the crate relationship mapping (original → optimized)
- [ ] Generate Markdown output with summary table and relationship diagram
- [ ] Write to `report.md`

This phase is intentionally simple. The complex work of computing actual code changes (visibility upgrades, import updates, etc.) is deferred to the implementation tool described in Future Work.

## Alternatives Considered

### Crate-at-a-time analysis (rejected)

An alternative approach would analyze a single crate in isolation, splitting it based solely on its internal SCC structure.

**The fundamental problem:** Crate-at-a-time can only **split** — it cannot **merge** symbols from different crates. Our union-find algorithm does both: it keeps SCCs separate when their dependents are in different crates (splitting), but merges SCCs when all their dependents are already in the same crate.

Without merging, crate-at-a-time would split every crate into its individual SCCs, potentially creating one crate per symbol. This produces a proliferation of micro-crates that increases build complexity without improving parallelism.

**Example:**

```
Crate-at-a-time on nexus-db-queries:
  SCCs: A, B, C, D (where D → C → A, C → B)
  Result: 4 separate crates (one per SCC)

Workspace analysis (union-find, processing top-down):
  D has 0 dependents → no-op (root)
  C has 1 dependent (D) → union into D's set
  A has 1 dependent (C) → union into D's set
  B has 1 dependent (C) → union into D's set
  Result: 1 crate containing {A, B, C, D}
```

Without seeing the full dependency graph, crate-at-a-time cannot make the merge decisions that keep crate count reasonable.

**Decision:** Workspace-level analysis is required. The union-find algorithm fundamentally needs to see all dependents of each SCC to make optimal split/merge decisions.

## Open Questions

1. **Compile time estimation**: The current design uses byte size as a proxy for compile time (see Future Work for potential improvements). Since the cost model only affects the reported improvement estimate, not the partitioning algorithm, byte size is acceptable as a starting point.

2. **Dev-dependency cycles**: Dev-dependencies can create cycles at the Cargo level (crate A dev-depends on crate B which depends on A). These are valid for test builds but may complicate analysis. Consider using tools like `cargo-cycles` or `cargo-ferris-wheel` to identify and clean up dev-dependency cycles before running tarjanize.

3. **Macro-generated code** (critical for correctness): Dependencies created by macro expansion must be captured for the analysis to be sound. rust-analyzer expands macros and provides dependency information through its HIR layer, so we analyze the expanded code, not the pre-expansion source. However, we should verify this works correctly for:
   - Procedural macros that generate cross-crate references
   - `#[derive]` macros that create trait impls
   - Declarative macros (`macro_rules!`) that reference external symbols

   If rust-analyzer's expansion is incomplete or incorrect, our dependency graph will be wrong, leading to invalid splits. This needs thorough testing with macro-heavy codebases.

4. **External consumers of workspace crates**: A crate in this workspace may also be used by other workspaces, git dependencies, or published to crates.io. Our tool optimizes for *this* workspace only — it doesn't see external consumers. If we reorganize symbols across crate boundaries, external projects break.

   Possible mitigations:
   - **Re-export facades**: Keep original crate names as thin crates that re-export from the reorganized crates, preserving the public API
   - **Configuration**: Let users mark crates as "has external consumers" to preserve their boundaries or require re-export facades
   - **Detection**: Check if crates are published to crates.io and warn accordingly

   This needs design work before the tool can be safely used on workspaces with externally-consumed crates.

## Future Work

These are out of scope for the initial implementation but would be valuable additions:

1. **Improved cost model**: The current design uses byte size as a proxy for compile time, which is a rough approximation. More accurate models could include:
   - Weighting generics by estimated monomorphization count
   - Counting trait impl complexity (more bounds = more inference work)
   - Distinguishing between debug and release builds (LLVM optimization dominates release)
   - Empirical calibration by measuring actual compile times on sample crates

   Note: The cost model only affects the *reported* improvement (Phase 3), not the partitioning algorithm itself—union-find merging is purely structural.

2. **Implementation tool**: A tool that takes `symbol_graph.json` and `optimized_symbol_graph.json` as inputs and generates incremental PRs to transform the workspace from the original to the optimized structure. This is a complex undertaking that requires careful design:
   - **Diff computation**: Compare the two symbol graphs to determine which symbols moved between crates
   - **Visibility analysis**: Determine which symbols need visibility upgrades (symbols that were crate-internal but are now accessed cross-crate)
   - **Import updates**: Find all dependents of moved symbols and update their import paths
   - **Module conflict resolution**: Handle cases where symbols from different original crates would collide in the same module
   - **Cargo.toml generation**: Create new crate manifests with correct dependencies
   - **Incremental PRs**: Generate one PR per logical change (e.g., per new crate) for easier review

   This tool is essential for making the analysis actionable, but requires significant design work around ordering of changes, handling of re-exports, and ensuring the workspace remains buildable at each intermediate step.

3. **Build time validation**: After applying changes, measure actual build times to validate the estimated speedup predictions.

4. **Performance benchmarking**: Benchmark the tool itself on large workspaces (e.g., Omicron) to understand runtime characteristics and identify optimization opportunities.

## References

- [GitHub Issue: Thoughts on improving omicron-nexus build times](https://github.com/oxidecomputer/omicron/issues/8015)
- [rust-analyzer architecture](https://rust-analyzer.github.io/blog/2020/10/24/introducing-ungrammar.html)
- [petgraph documentation](https://docs.rs/petgraph/latest/petgraph/)
- [petgraph tred module](https://docs.rs/petgraph/latest/petgraph/algo/tred/) — transitive reduction and closure algorithms
- [MATLAB: Graph condensation](https://www.mathworks.com/help/matlab/ref/digraph.condensation.html) — defines "condensation" as the DAG of SCCs
- [NetworkX: condensation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.condensation.html)
- [CP-Algorithms: Strongly Connected Components](https://cp-algorithms.com/graph/strongly-connected-components.html)
- [Wikipedia: Transitive reduction](https://en.wikipedia.org/wiki/Transitive_reduction) — the minimum equivalent graph preserving reachability
- Tarjan, R. E., & van Leeuwen, J. (1984). "Worst-case analysis of set union algorithms." *Journal of the ACM*, 31(2), 245-281. — proves O(m·α(n)) bound for union-find with path compression and union-by-rank
- Cormen, T. H., et al. *Introduction to Algorithms*, Chapter 21: "Data Structures for Disjoint Sets" — textbook treatment of union-find
