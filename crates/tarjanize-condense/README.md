# tarjanize-condense

Phase 2 of the tarjanize pipeline: Compute strongly connected components (SCCs) from a symbol graph and produce a condensed DAG.

## Overview

This crate transforms a `SymbolGraph` (from Phase 1) into a `CondensedGraph` where:
- Cycles become single SCC nodes
- SCCs are topologically sorted (dependents before dependencies)
- Each SCC stores its dependencies as a set of SCC IDs
- Each SCC tracks impl anchors for orphan rule enforcement

The condensed graph is a complete flattening of the SymbolGraph structure—all SCCs from all crates appear in a single vector, ready for Phase 3's optimal partitioning algorithm.

## Algorithm

1. **Build per-crate graph views** - For each crate, create a `SymbolGraphView` implementing petgraph traits
2. **Run Tarjan per crate in parallel** - Since Cargo enforces a DAG between crates, SCCs can't span crate boundaries, enabling parallel computation via rayon
3. **Assign global SCC IDs** - Merge per-crate results with sequential IDs from an atomic counter
4. **Resolve cross-crate dependencies** - Map symbol dependencies to SCC IDs across crate boundaries
5. **Collect impl anchors** - For each impl block, record which SCCs contain its orphan rule anchors
6. **Topological sort** - Reverse petgraph's output so dependents appear before dependencies

## Orphan Rule Constraints

Rust's orphan rule requires trait impls to be in the same crate as either the trait or at least one of the covered types. For `impl<P1..=Pn> Trait<T1..=Tn> for T0`, the impl is valid if:
- The trait is local, OR
- At least one of T0..=Tn is local (including trait type parameters)

Rather than forcing impls into the same SCC as their anchors (which would reduce Phase 3's flexibility), we track anchor constraints separately. Each `Scc` has an `anchor_sets` field containing one `AnchorSet` per impl block:

| Type | Field | Description |
|------|-------|-------------|
| `Scc` | `anchor_sets` | Set of `AnchorSet` values, one per impl in this SCC |
| `AnchorSet` | `anchors` | SCC IDs containing types/traits that can satisfy the orphan rule |

Phase 3 must solve a **hitting set problem**: find a minimal set of SCCs such that every `AnchorSet` contains at least one SCC from that set. For example, if one impl has anchors {A, B} and another has {A, C}, the minimal hitting set is {A}. This gives Phase 3 flexibility to choose the optimal grouping while ensuring orphan rule compliance.

## Usage

```rust
use tarjanize_condense::{run, condense};
use tarjanize_schemas::SymbolGraph;

// From reader/writer (CLI pattern)
let input = std::fs::File::open("symbol_graph.json")?;
let mut output = Vec::new();
run(input, &mut output)?;

// Direct API
let symbol_graph: SymbolGraph = /* ... */;
let condensed = condense(&symbol_graph);
```

CLI:
```bash
tarjanize condense symbol_graph.json -o condensed_graph.json
tarjanize condense < symbol_graph.json > condensed_graph.json
```

## Review Checklist

### Module: `lib.rs` (Public API)

#### Types
- [ ] Re-export `CondensedGraph` from tarjanize-schemas
- [ ] Re-export `AnchorSet` from tarjanize-schemas
- [ ] Re-export `Scc` from tarjanize-schemas
- [ ] Re-export `CondenseError`

#### Functions
- [ ] `condense(symbol_graph: &SymbolGraph) -> CondensedGraph` - Main condensation logic
- [ ] `run(input: impl Read, output: &mut dyn Write) -> Result<(), CondenseError>` - CLI entry point

#### Tests
- [ ] `test_condense_empty` - Empty graph produces empty output
- [ ] `test_condense_simple_chain` - Chain of 3 symbols produces 3 SCCs with 2 dependencies
- [ ] `test_condense_cycle` - Cycle of 2 symbols produces 1 SCC with 0 dependencies
- [ ] `test_run_roundtrip` - JSON roundtrip through run()
- [ ] `test_run_invalid_json` - Invalid JSON returns deserialization error

---

### Module: `error.rs` (Error Handling)

#### Types
- [ ] `CondenseError` - Public error struct with backtrace
- [ ] `CondenseErrorKind` (crate-internal) - Enum with Deserialization, Serialization, Io variants

#### Functions
- [ ] `CondenseError::new(kind: CondenseErrorKind) -> Self` - Constructor with backtrace capture
- [ ] `CondenseError::is_deserialization(&self) -> bool` - Check if deserialization error
- [ ] `CondenseError::is_serialization(&self) -> bool` - Check if serialization error
- [ ] `CondenseError::is_io(&self) -> bool` - Check if I/O error
- [ ] `CondenseError::backtrace(&self) -> &Backtrace` - Access captured backtrace

#### Trait Implementations
- [ ] `Display for CondenseErrorKind` - Human-readable error messages
- [ ] `Display for CondenseError` - Error message with backtrace
- [ ] `std::error::Error for CondenseError` - Error trait with source()
- [ ] `From<std::io::Error> for CondenseError` - Auto-conversion from I/O errors

#### Tests
- [ ] `test_deserialization` - Deserialization error has correct type and message
- [ ] `test_serialization` - Serialization error has correct type and message
- [ ] `test_io_from` - I/O error converts correctly via From
- [ ] `test_backtrace_captured` - Backtrace is accessible
- [ ] `test_debug_impl` - Debug formatting works

---

### Module: `symbol_graph_view.rs` (Per-Crate Graph View)

#### Types
- [x] `SymbolGraphView<'a>` - View of a single crate's symbols implementing petgraph traits
  - [x] `symbols: Vec<&'a Symbol>` - Symbols indexed by position
  - [x] `paths: IndexSet<String>` - Symbol paths with bidirectional lookup

#### Functions
- [x] `SymbolGraphView::new(crate_name: &str, root_module: &'a Module) -> Self` - Build view from module tree (crate_name used as path prefix)
- [x] `path(&self, index: usize) -> Option<&str>` - Returns path for a node index
- [x] `cost(&self, index: usize) -> Option<f64>` - Returns cost for a node index

#### Trait Implementations (for petgraph's `tarjan_scc`)
- [x] `GraphBase` - `NodeId = usize`, `EdgeId = ()`
- [x] `NodeIndexable` - Index conversion for `usize` NodeIds
- [x] `IntoNodeIdentifiers` - Iterator over node indices (0..len)
- [x] `IntoNeighbors` - Iterator over neighbor indices

#### Tests
- [ ] `test_empty_crate` - Empty module produces empty view
- [ ] `test_single_symbol_no_deps` - Single symbol, correct path/index/cost
- [ ] `test_two_symbols_with_dependency` - Dependency creates neighbor edge
- [ ] `test_cross_crate_dependency_filtered` - Cross-crate deps filtered out
- [ ] `test_submodules` - Nested modules handled correctly
- [ ] `test_tarjan_scc_integration` - petgraph's tarjan_scc works with our traits

---

### Module: `impls.rs` (Impl Anchor Collection)

#### Functions
- [ ] `collect_impl_anchors(symbol_graph: &SymbolGraph, path_to_scc: &HashMap<String, u32>) -> HashMap<String, AnchorSet>` - Collect impl anchors for orphan rule constraints
- [ ] `collect_impl_anchors_for_module(...)` (private) - Recursively process modules

#### Tests
- [ ] `test_inherent_impl_with_local_type` - Inherent impl records self type anchor
- [ ] `test_trait_impl_with_local_trait_only` - Trait impl on foreign type records trait anchor only
- [ ] `test_trait_impl_with_both_local` - Both type and trait anchors recorded
- [ ] `test_impl_with_neither_local` - No anchor recorded (external impl)
- [ ] `test_non_impl_symbols_ignored` - Non-impl symbols don't produce anchors
- [ ] `test_submodules` - Submodule impls handled correctly
- [ ] `test_trait_with_type_parameters` - Trait type params are also anchors

---

### Module: `scc.rs` (SCC Computation)

#### Types
- [ ] `SCC_COUNTER: AtomicU32` (static) - Global counter for sequential SCC IDs
- [ ] `CrateSccResult<'a>` (private) - Per-crate Tarjan result with view and SCCs
- [ ] `SccBuilder` (private) - Intermediate struct for building SCCs incrementally

#### Functions
- [ ] `compute_condensed_graph(symbol_graph: &SymbolGraph) -> CondensedGraph` - Main SCC computation (parallel per-crate)
- [ ] `reset_scc_counter()` (test-only) - Reset counter for deterministic test IDs
- [ ] `find_symbol(graph: &SymbolGraph, path: &str) -> Option<&Symbol>` (private) - Find symbol by path
- [ ] `find_symbol_in_module(module: &Module, parts: &[&str]) -> Option<&Symbol>` (private) - Recursive helper

#### Tests
- [ ] `test_single_symbol_single_scc` - One symbol = one SCC
- [ ] `test_two_independent_symbols` - Two unconnected symbols = two SCCs
- [ ] `test_cycle_forms_single_scc` - Mutual dependency = one SCC
- [ ] `test_chain_creates_dependencies` - Chain creates inter-SCC dependencies
- [ ] `test_cross_crate_dependency` - Cross-crate dependency preserved
- [ ] `test_impl_anchors_captured` - Impl anchors are recorded in SCCs
- [ ] `test_empty_graph` - Empty input = empty output
- [ ] `test_submodule_symbol` - Submodule symbols and dependencies handled correctly

---

### Schema Types (from `tarjanize-schemas`)

#### Types
- [ ] `CondensedGraph` - Root structure with flat list of SCCs
- [ ] `Scc` - SCC with id, crate_name, symbols, cost, dependencies, anchor_sets
- [ ] `AnchorSet` - Orphan rule anchor with anchors set

---

## Test Summary

| Module | Test Count |
|--------|------------|
| lib.rs | 5 |
| error.rs | 5 |
| symbol_graph_view.rs | 6 |
| impls.rs | 7 |
| scc.rs | 8 |
| **Total** | **31** |

## Dependencies

- `tarjanize-schemas` - Schema types (SymbolGraph, CondensedGraph, etc.)
- `petgraph` - Tarjan's SCC algorithm via `tarjan_scc()`
- `indexmap` - `IndexSet` for bidirectional path↔index lookup without data duplication
- `rayon` - Parallel iteration for per-crate SCC computation
- `serde_json` - JSON serialization/deserialization
- `tracing` - Structured logging

## Design Decisions

1. **Per-crate parallel computation** - Since Cargo enforces a DAG between crates, SCCs can never span crate boundaries. This allows us to run Tarjan's algorithm on each crate independently via rayon's `par_iter()`.
2. **IndexSet for bidirectional lookup** - Using `IndexSet<String>` for paths provides both index→path and path→index lookup without data duplication, satisfying petgraph's `NodeId = usize` requirement.
3. **Sequential SCC IDs** - Using an atomic counter for simple, predictable IDs rather than hashing symbol paths
4. **Impl anchors as constraints** - Rather than forcing impls into the same SCC as their anchors (via artificial edges), we track anchors separately. This gives Phase 3 flexibility to choose optimal groupings while ensuring orphan rule compliance.
5. **Deterministic output** - Sorting symbols within SCCs and dependencies by ID ensures reproducible JSON
6. **Sans-IO design** - `run()` accepts `impl Read` and `&mut dyn Write` for testability and composability
7. **Flat structure** - All SCCs in a single vector (vs nested by crate) simplifies Phase 3's union-find algorithm
