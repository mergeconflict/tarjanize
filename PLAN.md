# tarjanize

A tool to analyze Rust workspace dependency structures and identify opportunities for splitting crates into smaller, parallelizable units to improve build times.

**Why "tarjanize"?** The tool is named after [Robert Tarjan](https://en.wikipedia.org/wiki/Robert_Tarjan), whose algorithms are central to both phases of our analysis:
- **Phase 2** uses Tarjan's SCC algorithm (1972) to identify strongly connected components
- **Phase 3** uses union-find with path compression, whose near-linear time bound was proven by Tarjan & van Leeuwen (1984)

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

### Tools

- [ ] Install Lean 4 and Lake (build system)
- [ ] Install Creusot and Why3 — Creusot translates Rust code with contracts into Why3's intermediate language, and Why3 dispatches verification conditions to SMT solvers (Z3, CVC5, Alt-Ergo)
- [ ] Set up Rust project with `clap` for CLI, `serde` for JSON

### Lean project

Create the formal verification project:

```bash
lake new tarjanize-proofs
```

Add Mathlib dependency to `lakefile.lean`. We use Mathlib for:
- `Set` and `Finset` — representing nodes and edges
- `Set.ncard` — counting set cardinality (for dependent counts)
- Basic lemmas about sets, functions, and relations

### Core Lean definitions

These definitions form the mathematical foundation shared across phases. We define them upfront so that Phases 2 and 3 can build on a common vocabulary.

**Note**: The Lean code throughout this document is illustrative, showing the *intent* of each definition and theorem. The actual implementation will need adjustments for Mathlib compatibility, explicit type class constraints, and proof construction. The `sorry` placeholders indicate proofs to be completed during implementation.

```lean
import Mathlib.Data.Set.Basic
import Mathlib.Data.Finset.Basic

-- A directed graph: the fundamental structure we're partitioning.
-- We represent it as a set of edges (pairs of nodes).
structure Graph (α : Type*) where
  edges : Set (α × α)

-- The nodes of a graph: anything that appears in an edge.
def Graph.nodes (G : Graph α) : Set α :=
  {x | ∃ y, (x, y) ∈ G.edges ∨ (y, x) ∈ G.edges}

-- Reachability: can we get from x to y following edges?
-- This is needed to define SCCs (mutually reachable nodes) and acyclicity.
inductive Graph.Reachable (G : Graph α) : α → α → Prop
  | single : (x, y) ∈ G.edges → G.Reachable x y
  | trans : G.Reachable x y → G.Reachable y z → G.Reachable x z

-- Acyclicity: no node can reach itself.
-- This is the key property we must preserve — Cargo requires acyclic crate graphs.
def Graph.Acyclic (G : Graph α) : Prop :=
  ∀ x, ¬ G.Reachable x x

-- A partitioning assigns each node to a crate (represented as a natural number).
-- This is the output of our algorithm: which crate does each SCC belong to?
structure Partitioning (α : Type*) where
  assignment : α → ℕ

-- The induced crate graph: edges between crates based on edges between their nodes.
-- If node x depends on node y, and they're in different crates, there's a crate edge.
def inducedCrateGraph (G : Graph α) (P : Partitioning α) : Graph ℕ :=
  { edges := {(P.assignment x, P.assignment y) |
              (x, y) ∈ G.edges ∧ P.assignment x ≠ P.assignment y} }

-- A valid partitioning: the induced crate graph must be acyclic.
-- This is the central property we prove about our algorithm's output.
def ValidPartitioning (G : Graph α) (P : Partitioning α) : Prop :=
  (inducedCrateGraph G P).Acyclic
```

### Rust project

```bash
cargo new tarjanize --bin
```

- [ ] Define JSON Schema files (using `.schema.json` suffix):
  - `symbol_graph.schema.json`
  - `condensed_graph.schema.json`
- [ ] Set up test fixtures directory with small sample workspaces

### Continuous Integration

Set up CI (GitHub Actions or similar) to validate the entire project on every commit:

**Rust checks:**
- [ ] `cargo build` — code compiles
- [ ] `cargo test` — unit and integration tests pass
- [ ] `cargo clippy` — no lint warnings
- [ ] `cargo fmt --check` — code is formatted

**Lean checks:**
- [ ] `lake build` — all proofs type-check (in Lean, type-checking = proof validity)

**Creusot checks:**
- [ ] Run Creusot to generate Why3 files from annotated Rust code
- [ ] Run Why3 to verify all contracts (dispatches to SMT solvers)

**Integration checks:**
- [ ] Run the full pipeline on test fixture workspaces
- [ ] Validate output JSON against schemas

This ensures that algorithm changes don't break proofs, implementation changes don't violate contracts, and the pipeline remains end-to-end functional.

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
│  Phase 2: Compute SCCs and Condense Graph                         │
│  Input:  symbol_graph.json                                        │
│  Output: condensed_graph.json (DAG of SCCs)                       │
│  Properties: Acyclicity, Coverage, Connectedness, Maximality      │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 3: Compute Optimal Partitioning (Union-Find Merging)       │
│  Input:  condensed_graph.json                                     │
│  Output: optimized_condensed_graph.json (new crate groupings)     │
│  Properties: Acyclicity, Optimality, Coverage, Edge Preservation  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 4: Generate Optimized Symbol Graph                         │
│  Input:  symbol_graph.json, optimized_condensed_graph.json        │
│  Output: optimized_symbol_graph.json (new crate/module structure) │
│  Properties: Coverage, Edge Preservation                          │
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
│  Phase 5: Generate Report                                         │
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

Intermediate files (Phases 1-4) use **JSON** with **JSON Schema** validation (schema files use the `.schema.json` suffix). JSON is human-readable, has ubiquitous tooling (jq, editors), and excellent serde support. JSON Schema catches malformed files early, serves as documentation, and enables editor autocomplete. For very large workspaces, we may later add `--format messagepack`.

**Two schemas** cover the intermediate files:

| Schema | Description | Producers | Consumers |
|--------|-------------|-----------|-----------|
| SymbolGraph | Full symbol graph with crate/module hierarchy | Phases 1 and 4 | Phases 2, 4, and 5 |
| CondensedGraph | DAG of SCCs grouped by crate | Phases 2 and 3 | Phases 3 and 4 |

The final output (Phase 5) is a **Markdown report** (`report.md`). See Phase 5 for details.

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
│       │           └── impl: { self_type?, trait? }
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

**Impl fields**: `self_type` and `trait` are full paths if the target is local
(workspace member), or absent/null if foreign (external crate). Phase 2 uses these
to derive coherence edges.

See `schemas/symbol_graph.schema.json` for the complete JSON Schema definition.

### CondensedGraph

The symbol graph with SCCs condensed into atomic units, forming a DAG.

**Design principles**:
- References symbols via paths rather than duplicating symbol data
- Crates contain their SCCs (nested structure mirrors current workspace layout)
- Each SCC has a precomputed `cost` (sum of symbol costs) for build time calculations
- Edges between SCCs are included so Phase 3 doesn't need to load `symbol_graph.json`

**Key simplification**: Every symbol belongs to exactly one SCC. Single-symbol SCCs represent symbols with no cyclic dependencies. Multi-symbol SCCs represent symbols that must stay together due to cycles.

**SCC IDs**: Generated deterministically from the paths of contained symbols (e.g., hash of sorted paths). This ensures stability across runs.

#### Schema (`schemas/condensed_graph.schema.json`)

```
CondensedGraph
├── crates: []
│   └── Crate
│       ├── name: string
│       ├── cost: number (sum of SCC costs)
│       └── sccs: []
│           └── SCC
│               ├── id: string
│               ├── symbols: [] (paths referencing SymbolGraph)
│               └── cost: number
└── edges: []
    └── Edge
        ├── from: string (SCC id)
        └── to: string (SCC id)
```

See `schemas/condensed_graph.schema.json` for the complete JSON Schema definition.

## CLI Interface

The tool can run the full pipeline in one command, or execute individual phases for debugging and incremental workflows.

```bash
# Run full pipeline on a workspace
tarjanize analyze /path/to/omicron --output-dir ./analysis

# Run individual phases
tarjanize extract /path/to/omicron -o symbol_graph.json                    # Phase 1
tarjanize condense symbol_graph.json -o condensed_graph.json               # Phase 2
tarjanize optimize condensed_graph.json -o optimized_condensed_graph.json  # Phase 3
tarjanize reify symbol_graph.json optimized_condensed_graph.json \
    -o optimized_symbol_graph.json                                              # Phase 4
tarjanize report symbol_graph.json optimized_symbol_graph.json -o report.md  # Phase 5

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

# Exclude test code from analysis
tarjanize analyze /path/to/workspace --no-tests

# Analyze only for a specific target
tarjanize analyze /path/to/workspace --target x86_64-unknown-linux-gnu
```

The resulting dependency graph reflects whichever configuration is analyzed. Different configurations may produce different optimal partitionings.

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
- **Procedural macro crates**: These must remain as separate crates (Rust requirement) and cannot be merged with regular code. They are excluded entirely from the analysis.

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

- [ ] Integrate rust-analyzer crates (`ra_ap_ide`, `ra_ap_hir`, `ra_ap_vfs`)
- [ ] Implement workspace discovery from root `Cargo.toml`
- [ ] Load all workspace crates into rust-analyzer's database
- [ ] Implement symbol enumeration across all crates
- [ ] Implement dependency edge detection, including cross-crate edges
- [ ] Serialize output to `symbol_graph.json`
- [ ] Write tests against fixture workspaces

## Phase 2: Compute SCCs and Condense Graph

**Input**: `symbol_graph.json`
**Output**: `condensed_graph.json`

This phase transforms the symbol graph into a DAG suitable for partitioning by condensing strongly connected components.

### Why SCCs matter

Symbols within an SCC have cyclic dependencies and **must** stay in the same crate — splitting them would create a crate cycle, which Cargo forbids. By condensing SCCs into single nodes, we guarantee that any partitioning of the condensed graph produces a valid (acyclic) crate graph.

**Note**: SCCs are always contained within a single crate. Cross-crate cycles are impossible because Cargo enforces acyclic crate-level dependencies. If crate A depends on crate B, symbols in B cannot reference symbols in A, so no SCC can span both.

**Parallelization opportunity**: Since SCCs cannot span crates, Phase 2 can parallelize the expensive work:
1. Add coherence edges within each crate (parallel)
2. Find and condense SCCs within each crate (parallel)
3. **Sync point**: Combine results from all crates
4. Build edges between SCCs across the full graph (sequential, but O(|E|) edge scan)

### Algorithm

1. **Add coherence edges** to enforce orphan rules

   Rust's orphan rule requires that trait impls be defined in the same crate as either the trait or the type. We enforce this by adding synthetic edges that force impls into the same SCC as their "anchor" (the local item they must stay with).

   For each impl symbol, read its `self_type` and `trait` fields to determine locality:
   - **Only trait local** (self_type is null): Add `Trait → impl` (impl must stay with the trait)
   - **Only type local** (trait is null or foreign): Add `Type → impl` (impl must stay with the type)
   - **Both local**: Add `Type → impl` (either would work; we choose type by convention)
   - **Blanket impls** (`impl<T> Trait for T`): Add `Trait → impl` (blanket impls must stay with their trait)

   These synthetic edges create cycles that force the impl into the same SCC as its anchor, ensuring the orphan rule is satisfied after partitioning.

2. **Find SCCs within each crate** using Tarjan's algorithm
   - Can be parallelized since SCCs cannot span crates
   - Assign unique ID and compute cost (sum of symbol costs) for each SCC
   - Build mapping from symbol path → SCC ID

3. **Build edges between SCCs**
   - For each edge in the symbol graph, if source and target are in different SCCs, add an edge between those SCCs
   - Exclude the synthetic coherence edges (they served their purpose in step 2)

4. **Assemble output**: crates containing their SCCs, plus the SCC edge set

**Time complexity**: O(|V| + |E|) where V is symbols and E is edges (Tarjan's algorithm is linear).

### Formal properties

The algorithm guarantees four properties:

| Property | Description |
|----------|-------------|
| **Acyclicity** | The condensed graph is a DAG (cycles are eliminated by condensing SCCs) |
| **Coverage** | Every symbol from the input belongs to exactly one SCC (no symbols lost or duplicated) |
| **Connectedness** | All symbols within each SCC are mutually reachable |
| **Maximality** | Each SCC is maximal (no symbol outside the SCC is mutually reachable with symbols inside) |

### Implementation workflow

This phase requires formal verification. Follow these steps in order:

#### Step 2.1: Lean formalization

Define the mathematical objects for SCCs:

```lean
-- Strongly connected: mutually reachable
def StronglyConnected (G : Graph α) (x y : α) : Prop :=
  G.Reachable x y ∧ G.Reachable y x

-- An SCC is a maximal set of mutually reachable nodes
structure SCC (G : Graph α) where
  nodes : Set α
  nonempty : nodes.Nonempty
  connected : ∀ x y, x ∈ nodes → y ∈ nodes → StronglyConnected G x y
  maximal : ∀ z, (∃ x ∈ nodes, StronglyConnected G x z) → z ∈ nodes

-- The condensed graph (SCCs as nodes)
def condensedGraph (G : Graph α) (sccs : Set (SCC G)) : Graph (SCC G) :=
  { edges := {(s1, s2) | ∃ x y, x ∈ s1.nodes ∧ y ∈ s2.nodes ∧
                         (x, y) ∈ G.edges ∧ s1 ≠ s2} }
```

**Lean tasks:**
- [ ] Define `StronglyConnected`
- [ ] Define `SCC`
- [ ] Define `condensedGraph`
- [ ] Prove **Acyclicity**: condensed graph is always a DAG
- [ ] Prove **Coverage**: every node belongs to exactly one SCC
- [ ] Prove **Connectedness**: symbols within each SCC are mutually reachable
- [ ] Prove **Maximality**: each SCC is maximal

#### Step 2.2: Creusot annotations

Add contracts to Rust stub functions before implementation.

**Note**: The contracts below are illustrative pseudo-code showing the *intent* of each specification. Actual Creusot syntax requires defining model types (mathematical representations of data structures) and predicates. The implementation will need to:
- Define `#[logic]` functions for predicates like `is_dag()`, `contains_symbol()`, etc.
- Use Creusot's model types (`Seq`, `Set`, `Map`) rather than Rust iterators in specs
- Handle the gap between runtime types and their mathematical models

```rust
use creusot_contracts::*;

struct Edge {
    from: SymbolPath,
    to: SymbolPath,
}

#[requires(symbol_graph.is_valid())]
#[ensures(result.is_dag())]
#[ensures(forall<id: SymbolPath> symbol_graph.contains(id) ==>
          exists<scc: SccId> result.contains_symbol(scc, id))]
pub fn condense_graph(symbol_graph: &SymbolGraph) -> CondensedGraph {
    unimplemented!()  // Stub for annotation
}

#[requires(symbols.len() > 0)]
#[ensures(result.iter().all(|scc| scc.symbols.len() > 0))]
#[ensures(result.iter().flat_map(|scc| scc.symbols.iter()).count() == symbols.len())]
// Connectedness: all symbols within each SCC are mutually reachable
#[ensures(forall<scc: &Scc> result.contains(scc) ==>
          forall<x: SymbolPath, y: SymbolPath> scc.symbols.contains(x) && scc.symbols.contains(y) ==>
            mutually_reachable(edges, x, y))]
// Maximality: no symbol outside the SCC is mutually reachable with all symbols inside
#[ensures(forall<scc: &Scc, z: SymbolPath> result.contains(scc) && !scc.symbols.contains(z) ==>
          exists<x: SymbolPath> scc.symbols.contains(x) && !mutually_reachable(edges, x, z))]
fn tarjan_scc(symbols: &[SymbolPath], edges: &[Edge]) -> Vec<Scc> {
    unimplemented!()  // Stub for annotation
}
```

**Creusot annotation tasks:**
- [ ] Add `#[requires]` and `#[ensures]` to `condense_graph`
- [ ] Add contracts to `tarjan_scc`
- [ ] Verify contracts are well-formed (Creusot parses them)

#### Step 2.3: Rust implementation

Implement the annotated functions:

- [ ] Implement coherence edge injection (derive from impl `self_type`/`trait` fields)
- [ ] Implement Tarjan's SCC algorithm (or use petgraph)
- [ ] Condense SCCs into single nodes
- [ ] Build condensed DAG (excluding synthetic coherence edges)
- [ ] Compute cost for each SCC
- [ ] Serialize output to `condensed_graph.json`

#### Step 2.4: Creusot verification

- [ ] Run Creusot to verify implementation matches contracts
- [ ] Fix any verification failures
- [ ] Write tests verifying formal properties:
  - Acyclicity: output is always a DAG
  - Coverage: all input symbols appear in exactly one output SCC
  - Connectedness: symbols within each SCC are mutually reachable
  - Maximality: no symbol outside an SCC is mutually reachable with symbols inside

## Phase 3: Compute Optimal Partitioning (Union-Find Merging)

**Input**: `condensed_graph.json`
**Output**: `optimized_condensed_graph.json`

This phase computes the optimal crate partitioning using a **union-find** data structure to merge SCCs into crates. This is the core algorithm with the strongest formal requirements.

### Formal properties

The algorithm guarantees four properties:

| Property | Description | How guaranteed |
|----------|-------------|----------------|
| **Acyclicity** | Crate dependency graph is a DAG | Merges follow dependency direction; no cycles possible |
| **Optimality** | Minimum critical path cost among valid partitionings | SCCs merge when all dependents are in the same crate |
| **Coverage** | Every SCC appears in exactly one crate | Union-find assigns each SCC to exactly one set |
| **Edge Preservation** | All SCC dependencies preserved | Edges unchanged; only crate groupings change |

#### Definitions

- **dependents(n)**: the set of SCCs that depend on n (i.e., SCCs `m` where there exists an edge `m → n`)
- **all_dependents_same_set(n)**: true if all dependents of n belong to the same union-find set (trivially true if n has 0 or 1 dependents)

### Merge criterion

An SCC is merged into its dependents' crate **if all its dependents are already in the same crate**. This criterion:
- Merges single-dependent SCCs (the common case)
- Merges multi-dependent SCCs when their dependents have already been unified (e.g., diamonds)
- Keeps SCCs separate when their dependents are in different crates (preserving parallelism)

### Algorithm

1. **Topological sort** the SCC DAG

2. **Initialize union-find**: each SCC starts in its own singleton set

3. **Process SCCs in topological order** (dependents before their dependencies):

   **Terminology**: In this document, a "root" is an SCC with no dependents (nothing depends on it), and a "leaf" is an SCC with no dependencies (it depends on nothing). We process from roots toward leaves.
   - For each SCC, find all its dependents
   - If **zero dependents**: no-op (SCC is a root, remains in its own set)
   - If **all dependents are in the same set**: `union()` this SCC into that set
   - If **dependents are in different sets**: no-op (SCC is a boundary, remains in its own set)

4. **Assemble output**: each union-find set becomes a crate, edges unchanged

**Time complexity**: O(|V| + |E| · α(V)) where α is the inverse Ackermann function. Since α(V) ≤ 4 for any practical V, this is effectively **O(|V| + |E|)**.

**Important**: To achieve this near-linear time bound, the union-find implementation must use both **path compression** and **union-by-rank** (or union-by-size). See Tarjan & van Leeuwen (1984) for the analysis, or Cormen et al. *Introduction to Algorithms* Chapter 21.

Processing order matters: by going top-down (dependents before their dependencies), we know the final set assignment of all dependents before deciding whether to merge.

### Visualizing Union-Find Merging

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

Processing top-down: A, B, C, D

| Step | SCC | Dependents | Their sets | Action | Sets after |
|------|-----|------------|------------|--------|------------|
| 1 | A | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 2 | B | none | — | no-op (root) | {A}, {B}, {C}, {D} |
| 3 | C | A, B | {A}, {B} | no-op (boundary) | {A}, {B}, {C}, {D} |
| 4 | D | C | {C} | union into {C} | {A}, {B}, {C,D} |

Result: **Three crates: {A}, {B}, {C, D}**

This allows `{A}` and `{B}` to compile in parallel after `{C, D}` finishes.

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

### Implementation workflow

This phase has the strongest formal requirements. Follow these steps in order:

#### Step 3.1: Lean formalization

Define union-find merging and prove the four properties.

**Note on existing union-find formalizations**: The [Isabelle Archive of Formal Proofs](https://www.isa-afp.org/) contains a verified union-find implementation: [Relational Disjoint-Set Forests](https://www.isa-afp.org/entries/Relational_Disjoint_Set_Forests.html) by Walter Guttmann (2020), which includes proofs for path compression, path halving, path splitting, and union-by-rank. While this is in Isabelle rather than Lean, the proof structure and lemmas may be adaptable. As of 2025, there is no standard union-find formalization in Mathlib.

**Definitions:**

```lean
-- Dependents of a node (nodes that have edges TO this node)
def Graph.dependents (G : Graph α) (x : α) : Set α :=
  {y | (y, x) ∈ G.edges}

-- Union-find state: maps each node to its representative (root of its set)
-- In the algorithm, we build this incrementally; here we define the final result
structure UnionFind (α : Type*) where
  find : α → α
  -- find is idempotent: find(find(x)) = find(x)
  find_idempotent : ∀ x, find (find x) = find x

-- Two nodes are in the same set if they have the same representative
def UnionFind.sameSet (uf : UnionFind α) (x y : α) : Prop :=
  uf.find x = uf.find y

-- All dependents of x are in the same set
def allDependentsSameSet (G : Graph α) (uf : UnionFind α) (x : α) : Prop :=
  ∀ y z, y ∈ G.dependents x → z ∈ G.dependents x → uf.sameSet y z

-- The merge operation: if all dependents are in the same set, merge x into that set
-- This is the core of the algorithm, applied top-down (dependents before dependencies)
def mergeStep (G : Graph α) (uf : UnionFind α) (x : α) : UnionFind α :=
  if h : (G.dependents x).Nonempty ∧ allDependentsSameSet G uf x then
    -- Merge x into the set containing its dependents
    let representative := uf.find (G.dependents x).toFinset.min' h.1.toFinset
    { find := fun y => if y = x then representative else uf.find y
      find_idempotent := sorry }
  else
    uf  -- x stays in its own set (either no dependents, or dependents in different sets)

-- The full algorithm: fold mergeStep over nodes top-down (dependents before dependencies)
-- Starting from the identity union-find (each node is its own representative)
noncomputable def unionFindMerge (G : Graph α) [Fintype α] (hdag : G.Acyclic)
    (topoOrder : List α) : UnionFind α :=
  topoOrder.foldl (mergeStep G) { find := id, find_idempotent := fun _ => rfl }

-- The partitioning induced by union-find: nodes with the same representative are in the same crate
def ufToPartitioning (uf : UnionFind α) [Fintype α] : Partitioning α :=
  -- Map representatives to crate numbers (implementation detail)
  sorry

-- All crates in a partitioning
def allCrates (P : Partitioning α) : Set ℕ :=
  {c | ∃ x, P.assignment x = c}

-- Crate c1 depends on crate c2
def dependsOn (G : Graph α) (P : Partitioning α) (c1 c2 : ℕ) : Prop :=
  ∃ x y, P.assignment x = c1 ∧ P.assignment y = c2 ∧ (x, y) ∈ G.edges ∧ c1 ≠ c2

-- Critical path cost (cost of crate plus max cost of dependency chains)
noncomputable def criticalPathCost (G : Graph α) (P : Partitioning α)
    (cost : ℕ → ℕ) (crate : ℕ) : ℕ :=
  let deps := {c | dependsOn G P crate c}
  let maxDepCost := Finset.sup deps (criticalPathCost G P cost)
  maxDepCost + cost crate
```

**Theorems to prove:**

```lean
-- Key lemma: merging preserves acyclicity
-- If x merges into a set S, then x has no path to any node in S (since we process top-down)
lemma merge_preserves_acyclicity (G : Graph α) [Fintype α] (hdag : G.Acyclic)
    (uf : UnionFind α) (x : α) (hvalid : ValidPartitioning G (ufToPartitioning uf)) :
    ValidPartitioning G (ufToPartitioning (mergeStep G uf x)) := by
  sorry -- TO PROVE

-- Theorem 1: Acyclicity
theorem union_find_merge_acyclic (G : Graph α) [Fintype α]
    (hdag : G.Acyclic) (topoOrder : List α) (htopo : IsTopoOrder G topoOrder) :
    ValidPartitioning G (ufToPartitioning (unionFindMerge G hdag topoOrder)) := by
  sorry -- TO PROVE (by induction on topoOrder using merge_preserves_acyclicity)

-- Theorem 2: Optimality (max critical path cost is minimized)
-- Note: merging x when all dependents are in set S doesn't increase critical path,
-- because x must be compiled before anything in S anyway
theorem union_find_merge_optimal (G : Graph α) [Fintype α]
    (hdag : G.Acyclic) (topoOrder : List α) (htopo : IsTopoOrder G topoOrder)
    (cost : α → ℕ) :
    ∀ P : Partitioning α, ValidPartitioning G P →
      let ufPartition := ufToPartitioning (unionFindMerge G hdag topoOrder)
      let rootCrates P := {c ∈ allCrates P | ∀ c', c' ∈ allCrates P → ¬dependsOn G P c' c}
      let maxCost P := Finset.sup (rootCrates P) (criticalPathCost G P cost)
      maxCost ufPartition ≤ maxCost P := by
  sorry -- TO PROVE

-- Theorem 3: Coverage (every node ends up in exactly one set)
theorem union_find_merge_coverage (G : Graph α) [Fintype α]
    (hdag : G.Acyclic) (topoOrder : List α) (htopo : IsTopoOrder G topoOrder) :
    ∀ x ∈ G.nodes, ∃! c, (ufToPartitioning (unionFindMerge G hdag topoOrder)).assignment x = c := by
  sorry -- TO PROVE (follows from union-find properties)

-- Theorem 4: Edge Preservation
theorem union_find_merge_edge_preservation (G : Graph α) [Fintype α]
    (hdag : G.Acyclic) (topoOrder : List α) (htopo : IsTopoOrder G topoOrder) :
    let P := ufToPartitioning (unionFindMerge G hdag topoOrder)
    ∀ x y, (x, y) ∈ G.edges →
      (P.assignment x = P.assignment y ∨
       (P.assignment x, P.assignment y) ∈ (inducedCrateGraph G P).edges) := by
  sorry -- TO PROVE
```

**Lean tasks:**
- [ ] Define `UnionFind`, `sameSet`, `allDependentsSameSet`
- [ ] Define `mergeStep`, `unionFindMerge`
- [ ] Define `ufToPartitioning`
- [ ] Define `allCrates`, `dependsOn`, `criticalPathCost`
- [ ] Prove `merge_preserves_acyclicity` (key lemma)
- [ ] Prove `union_find_merge_acyclic`
- [ ] Prove `union_find_merge_optimal`
- [ ] Prove `union_find_merge_coverage`
- [ ] Prove `union_find_merge_edge_preservation`
- [ ] Eliminate all `sorry`s
- [ ] Document proof strategies

#### Step 3.2: Creusot annotations

Add contracts to Rust stub functions.

**Note**: The contracts below are illustrative pseudo-code showing the *intent* of each specification. See the note in Step 2.2 regarding actual Creusot syntax.

```rust
use creusot_contracts::*;

// Main algorithm entry point
// Acyclicity
#[requires(condensed_graph.is_dag())]
#[ensures(result.is_dag())]
// Optimality: max critical path cost of root crates is no worse than any valid partitioning
#[ensures(result.max_root_critical_path_cost() <= condensed_graph.max_root_critical_path_cost())]
// Coverage: every input SCC appears in exactly one output crate
#[ensures(forall<scc: SccId> condensed_graph.contains_scc(scc) ==>
          exists_unique<crate_id: CrateId> result.crate_contains_scc(crate_id, scc))]
// Edge Preservation: all input edges are preserved
#[ensures(result.edges == condensed_graph.edges)]
pub fn compute_union_find_merge(
    condensed_graph: &CondensedGraph
) -> CondensedGraph {
    unimplemented!()  // Stub for annotation
}

// Returns the set of SCCs that depend on this SCC (have edges pointing to it)
#[ensures(forall<other: SccId>
    result.contains(other) <==> condensed_graph.has_edge(other, scc_id))]
fn dependents(condensed_graph: &CondensedGraph, scc_id: SccId) -> Set<SccId> {
    unimplemented!()
}

// Returns true if all dependents of this SCC are in the same union-find set
#[ensures(result <==> forall<a: SccId, b: SccId>
    dependents(condensed_graph, scc_id).contains(a) &&
    dependents(condensed_graph, scc_id).contains(b) ==>
    uf.find(a) == uf.find(b))]
fn all_dependents_same_set(
    condensed_graph: &CondensedGraph,
    uf: &UnionFind,
    scc_id: SccId
) -> bool {
    unimplemented!()
}

// Perform one merge step: if all dependents are in the same set, merge this SCC into that set
#[requires(uf.is_valid())]
#[ensures(result.is_valid())]
// If merged, the SCC now has the same representative as its dependents
#[ensures(all_dependents_same_set(condensed_graph, uf, scc_id) &&
          !dependents(condensed_graph, scc_id).is_empty() ==>
          result.find(scc_id) == uf.find(dependents(condensed_graph, scc_id).any()))]
// If not merged, the SCC keeps its own representative
#[ensures(!all_dependents_same_set(condensed_graph, uf, scc_id) ||
          dependents(condensed_graph, scc_id).is_empty() ==>
          result.find(scc_id) == uf.find(scc_id))]
fn merge_step(
    condensed_graph: &CondensedGraph,
    uf: UnionFind,
    scc_id: SccId
) -> UnionFind {
    unimplemented!()
}

// Critical path cost: cost of crate plus max critical path cost of dependencies
#[ensures(result == condensed_graph.crate_cost(crate_id) +
          condensed_graph.dependencies(crate_id).iter()
            .map(|dep| critical_path_cost(condensed_graph, *dep))
            .max().unwrap_or(0))]
fn critical_path_cost(condensed_graph: &CondensedGraph, crate_id: CrateId) -> usize {
    unimplemented!()
}
```

**Creusot annotation tasks:**
- [ ] Add contracts to `compute_union_find_merge`
- [ ] Add contracts to `dependents`
- [ ] Add contracts to `all_dependents_same_set`
- [ ] Add contracts to `merge_step`
- [ ] Add contracts to `critical_path_cost`

#### Step 3.3: Rust implementation

Implement the annotated functions:

- [ ] Implement `UnionFind` with path compression and union-by-rank
- [ ] Implement `topological_sort` for the SCC DAG
- [ ] Implement `dependents` (precompute as adjacency list for O(1) lookup per edge)
- [ ] Implement `all_dependents_same_set` (iterate dependents, check `find()` equality)
- [ ] Implement `merge_step` (conditionally call `union()`)
- [ ] Implement `compute_union_find_merge` (fold top-down from roots to leaves)
- [ ] Implement `critical_path_cost` for verification
- [ ] Serialize output to `optimized_condensed_graph.json`

#### Step 3.4: Creusot verification

- [ ] Run Creusot to verify implementation
- [ ] Test: output is always acyclic
- [ ] Test: coverage — every input SCC appears in exactly one output crate
- [ ] Test: edge preservation — all input edges are preserved in output
- [ ] Test: optimality — no valid partitioning has lower max critical path cost (verified exhaustively for small graphs)
- [ ] Test: diamond pattern produces single crate (regression test for the union-find improvement)

**Note**: The optimality property (universal quantifier over all partitionings) may not verify in Creusot due to SMT solver limitations — that's acceptable since we proved it in Lean.

## Phase 4: Generate Reorganized Symbol Graph

**Input**: `symbol_graph.json`, `optimized_condensed_graph.json`
**Output**: `optimized_symbol_graph.json`

This phase transforms the optimized partitioning into a complete symbol graph with the new crate/module structure. The output uses the same schema as `symbol_graph.json`, enabling apples-to-apples comparison with the original.

### Formal Properties

| Property | Description |
|----------|-------------|
| **Soundness** | Every symbol in the output is referenced by some SCC in the condensed graph |
| **Completeness** | Every symbol referenced by an SCC in the condensed graph appears exactly once in the output |
| **Edge Preservation** | All edges from the input are preserved unchanged |
| **Path Preservation** | Non-conflicting symbols retain their original module path; conflicting symbols have a single synthetic ancestor inserted |
| **File Preservation** | Each symbol retains its original file path |
| **Visibility** | Each symbol is visible to all symbols that reference it |

### Algorithm

1. **Build symbol index** from original graph
   - Map path → (Symbol, original_crate, module_path)

2. **Build crate assignment** from condensed graph
   - Map path → new crate ID (needed for visibility computation)

3. **Compute visibility** by examining edges
   - Cross-crate reference → target must be `pub`
   - Cross-module reference (same crate) → target must be `pub(crate)`
   - **Note**: Visibility is only *widened*, never narrowed. If two crates merge and a `pub` symbol could become `pub(crate)`, we leave it as `pub` for backwards compatibility with any external consumers.

4. **For each new crate**, build its module tree:
   - Collect symbols by flattening SCCs → paths → lookup in symbol index
   - Detect conflicts: multiple symbols with same (module_path, name) from different original crates
   - Assign final paths:
     - Non-conflicting: original module path
     - Conflicting: module path + `conflict_from_{original_crate}`
   - Update visibility for symbols that need widening
   - Build nested module tree from (symbol, path) pairs

5. **Generate crate names** from original crate names and module paths (LLM or fallback)

6. **Assemble output**: new crates with generated names, edges unchanged

**Time complexity**: O(|V| + |E|) where V is symbols and E is edges — each step is linear.

### Crate Naming

Phase 4 generates **stable placeholder names** using a deterministic fallback algorithm: `{original-crate}-{top-module}` or `{original-crate}-{id}`. This ensures reproducible outputs across runs.

For more meaningful names, use the separate `tarjanize rename` command, which uses an LLM to suggest names based on crate contents:

```bash
# Generate optimized graph with placeholder names
tarjanize reify symbol_graph.json optimized_condensed_graph.json -o optimized_symbol_graph.json

# Optionally rename crates using LLM suggestions
tarjanize rename optimized_symbol_graph.json -o renamed_symbol_graph.json
```

**LLM prompt structure** (for the `rename` command):
```
You are naming new Rust crates created by splitting/merging existing crates.

For each crate below, suggest a name that:
- Uses lowercase kebab-case (e.g., "nexus-db-silo")
- Is concise (2-4 words)
- Reflects the common theme of the contained modules
- Uses the original crate name as a prefix when appropriate

Use the submit_crate_names tool to provide your suggestions.

## crate_001
Original crates: nexus-db-queries
Module paths: silo, silo::helpers, silo::config

## crate_002
Original crates: nexus-db-queries, nexus-auth
Module paths: authz, authz::policy
```

**Tool schema** for structured output:
```json
{
  "name": "submit_crate_names",
  "description": "Submit the suggested names for each crate",
  "input_schema": {
    "type": "object",
    "properties": {
      "names": {
        "type": "object",
        "additionalProperties": { "type": "string" },
        "description": "Map from crate ID to suggested name"
      }
    },
    "required": ["names"]
  }
}
```

### Visualization

**Input state:**
```
symbol_graph.json:
  crate_a:
    foo/bar.rs: fn x(), fn y()
  crate_b:
    foo/bar.rs: fn x(), fn z()

optimized_condensed_graph.json:
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

### Implementation Workflow

#### Step 4.1: Lean Formalization

```lean
-- Symbol placement: a symbol and its final module path
structure SymbolPlacement (α : Type*) where
  symbol : α
  path : List String

-- A module tree
inductive ModuleTree (α : Type*) where
  | node : String → List α → List (ModuleTree α) → ModuleTree α

-- All symbols in a module tree
def ModuleTree.allSymbols : ModuleTree α → List α
  | .node _ syms children => syms ++ children.flatMap ModuleTree.allSymbols

-- Build module tree from placements (specification)
def buildModuleTree (placements : List (SymbolPlacement α)) : ModuleTree α := sorry

-- Theorem 1: Soundness - every symbol in output came from placements
-- Theorem 2: Completeness - every placement appears exactly once in output
-- (Combined: output symbols = input symbols as sets, with no duplicates)
theorem build_module_tree_soundness_completeness (placements : List (SymbolPlacement α)) :
    (buildModuleTree placements).allSymbols.toFinset =
    (placements.map (·.symbol)).toFinset := by
  sorry

theorem build_module_tree_no_duplication (placements : List (SymbolPlacement α)) :
    (buildModuleTree placements).allSymbols.Nodup := by
  sorry
```

**Lean tasks:**
- [ ] Define `SymbolPlacement`, `ModuleTree`
- [ ] Define `buildModuleTree`
- [ ] Prove `build_module_tree_soundness_completeness`
- [ ] Prove `build_module_tree_no_duplication`

#### Step 4.2: Creusot Annotations

```rust
use creusot_contracts::*;

// Soundness: output symbols ⊆ input placements
// Completeness: input placements ⊆ output symbols (each exactly once)
#[requires(forall<p: &SymbolPlacement> placements.contains(p) ==>
           symbol_graph.contains(p.symbol.id))]
#[ensures(forall<id: SymbolPath>
    placements.iter().any(|p| p.symbol.id == id) <==>
    result.contains_symbol(id))]
#[ensures(result.symbol_count() == placements.len())]
pub fn build_module_tree(placements: &[SymbolPlacement]) -> Module {
    unimplemented!()
}

// Soundness: every output symbol is referenced by some SCC
// Completeness: every SCC-referenced SymbolPath appears exactly once in output
// Visibility: symbols with cross-crate edges targeting them are pub
#[requires(symbol_graph.is_valid())]
#[requires(condensed_graph.is_valid())]
#[ensures(forall<def_id: SymbolPath>
    condensed_graph.contains_def_id(def_id) <==>
    result.contains_symbol(def_id))]
#[ensures(result.edges == symbol_graph.edges)]
#[ensures(forall<edge: &Edge> symbol_graph.edges.contains(edge) ==>
    (result.crate_of(edge.from) != result.crate_of(edge.to)) ==>
    result.symbol(edge.to).visibility == "pub")]
pub fn generate_optimized_symbol_graph(
    symbol_graph: &SymbolGraph,
    condensed_graph: &CondensedGraph,
) -> SymbolGraph {
    unimplemented!()
}
```

**Creusot tasks:**
- [ ] Add contracts to `build_module_tree`
- [ ] Add contracts to `generate_optimized_symbol_graph`
- [ ] Add contracts to conflict detection helpers

#### Step 4.3: Rust Implementation

- [ ] Implement `traverse_modules` iterator
- [ ] Implement symbol index construction
- [ ] Implement symbol-to-new-crate mapping
- [ ] Implement visibility computation (widen to pub for cross-crate edge targets)
- [ ] Implement conflict detection
- [ ] Implement `build_module_tree` using trie approach
- [ ] Implement `sanitize_crate_name`
- [ ] Implement `generate_crate_names_fallback` (deterministic placeholder names)
- [ ] Implement `deduplicate_names`
- [ ] Serialize output to `optimized_symbol_graph.json`

#### Step 4.4: Creusot Verification

- [ ] Run Creusot to verify implementation
- [ ] Test soundness: every output symbol is referenced by some SCC
- [ ] Test completeness: every SCC-referenced symbol appears exactly once in output
- [ ] Test edge preservation: output edges equal input edges
- [ ] Test path preservation: non-conflicting symbols retain original paths
- [ ] Test file preservation: each symbol retains its original file path
- [ ] Test visibility: each symbol is visible to all symbols that reference it

#### Step 4.5: Rename command (optional LLM naming)

- [ ] Implement `tarjanize rename` subcommand
- [ ] Implement `generate_crate_names_llm` with Anthropic API
- [ ] Load existing symbol graph, apply new names, write output

## Phase 5: Generate Report

**Input**: `symbol_graph.json`, `optimized_symbol_graph.json` (or `renamed_symbol_graph.json` if rename was used)
**Output**: `report.md`

The report is a human-readable Markdown summary of the optimization results. Phase 5 uses whatever crate names are present in its input file—placeholder names from Phase 4, or LLM-suggested names from the optional rename step. It is designed for understanding and communicating the benefits of the proposed split, not for driving automated implementation.

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

## Verification Approach

This project uses **Lean 4** for proving the algorithm correct and **Creusot** for verifying the Rust implementation. Phases 2-4 have formal properties with both Lean proofs and Creusot verification. Phase 5 is a simple report generator without formal verification.

### Why both Lean and Creusot?

| Aspect | Lean 4 | Creusot |
|--------|--------|---------|
| **Proof method** | Constructive proofs checked by type theory | SMT solvers search for counterexamples |
| **Generality** | Truly universal ("for ALL splits...") | May timeout on complex quantifiers |
| **Failure mode** | Proof fails → you know something is wrong and *why* | Verification fails → bug? solver not smart enough? needs hints? |
| **What it verifies** | Abstract algorithm design | Specific Rust implementation |

**Why we need Lean (not just Creusot):**

1. **Optimality is hard for SMT**: The property "no alternative partitioning has lower build time" is a universal quantifier over all possible partitionings. SMT solvers struggle with unbounded universal quantification; Lean proves it constructively.

2. **Design vs. implementation**: Lean proves the *algorithm design* is correct. Creusot verifies the *implementation* matches some spec. If we skip Lean, we might correctly implement a flawed algorithm.

3. **Understanding**: Writing Lean proofs forces us to understand *why* the algorithm works. This knowledge helps debug issues later.

**Why we need Creusot (not just Lean):**

1. **Specification gap**: Lean proves properties of abstract definitions, not our actual Rust code. Creusot bridges this gap.

2. **Implementation bugs**: Even with a correct algorithm, we might implement it wrong. Creusot catches these.

3. **Refactoring confidence**: As we optimize the Rust code, Creusot re-verifies correctness.

**Confidence comparison:**

| Phase | Property | Lean | Creusot |
|-------|----------|------|---------|
| 2 | Acyclicity | ✅ | ✅ |
| 2 | Coverage | ✅ | ✅ |
| 2 | Connectedness | ✅ | ✅ |
| 2 | Maximality | ✅ | ✅ |
| 3 | Acyclicity | ✅ | ✅ |
| 3 | Optimality | ✅ | ❌ (universal quantifier) |
| 3 | Coverage | ✅ | ✅ |
| 3 | Edge Preservation | ✅ | ✅ |
| 4 | Coverage | ✅ | ✅ |
| 4 | Edge Preservation | ✅ | ✅ |

The only property that cannot be verified in Creusot is Phase 3's Optimality (universal quantifier over all partitionings).

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

   Note: The cost model only affects the *reported* improvement (Phase 5), not the partitioning algorithm itself—union-find merging is purely structural.

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
- [Lean 4 documentation](https://lean-lang.org/documentation/)
- [Creusot verification tool](https://github.com/creusot-rs/creusot) — see the [guide](https://creusot-rs.github.io/creusot/) for contract syntax and [pearlite](https://creusot-rs.github.io/creusot/pearlite/) for the specification language
- [MATLAB: Graph condensation](https://www.mathworks.com/help/matlab/ref/digraph.condensation.html) — defines "condensation" as the DAG of SCCs
- [NetworkX: condensation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.condensation.html)
- [CP-Algorithms: Strongly Connected Components](https://cp-algorithms.com/graph/strongly-connected-components.html)
- Tarjan, R. E., & van Leeuwen, J. (1984). "Worst-case analysis of set union algorithms." *Journal of the ACM*, 31(2), 245-281. — proves O(m·α(n)) bound for union-find with path compression and union-by-rank
- Cormen, T. H., et al. *Introduction to Algorithms*, Chapter 21: "Data Structures for Disjoint Sets" — textbook treatment of union-find
- [Isabelle AFP: Relational Disjoint-Set Forests](https://www.isa-afp.org/entries/Relational_Disjoint_Set_Forests.html) — verified union-find with path compression and union-by-rank in Isabelle/HOL
