# tarjanize-condense

Phase 2 of the tarjanize pipeline: Compute SCCs, merge into optimal crate groupings, and produce an optimized `SymbolGraph`.

## Overview

This crate transforms a `SymbolGraph` (from Phase 1) into an optimized `SymbolGraph` where symbols have been repartitioned into synthetic crates that maximize build parallelism while respecting Rust's orphan rule.

The output is the same `SymbolGraph` type as the input, but with a different package structure: each package contains a single `"synthetic"` target with the grouped symbols.

## Algorithm

The core algorithm is in `scc.rs` (`condense_and_partition`):

1. **Build symbol index** — Index all symbols across all packages/targets with `[package/target]::module::symbol` paths
2. **Build directed graph** — Symbols as nodes, dependencies as edges (using petgraph `DiGraph`)
3. **Add synthetic back-edges for anchors** — For each impl block, add an edge from its best anchor (minimum in-degree) back to the impl. This creates cycles that condensation will collapse, ensuring impls stay with their anchors (orphan rule)
4. **Condensation** — petgraph's `condensation()` finds SCCs and produces a DAG
5. **Precompute reachable sets** — For each SCC, compute transitive reachable sets (used by HMR)
6. **Union-find merging** — Process SCCs in reverse topological order (dependents before dependencies). Merge an SCC into its dependent's set when it has exactly one *effective* dependent (after HMR transitive reduction removes redundant edges). SCCs with 0 effective dependents are roots; SCCs with 2+ are boundaries — both stay separate
7. **Build output** — Group symbols by union-find set, rewrite all paths and dependencies to new `[package/synthetic]::...` format

### Key Concepts

**Effective dependents**: A dependent set S of SCC X is "effective" if the edge S→X is non-redundant — i.e., X is not already reachable from S through other dependencies. This is computed using incremental HMR (Habib-Morvan-Rampon) transitive reduction, which is recomputed after each merge.

**Metadata estimation**: Synthetic crates don't have real metadata times, so they're estimated via linear regression: `metadata_ms = 0.26 * frontend_ms + 1662` (R² = 0.705, derived from Omicron build data). See `docs/cost-model.md`.

**Path conflict resolution**: When symbols from different original crates end up in the same synthetic crate with conflicting paths, they're placed in `conflict_from_{original_crate}` submodules.

## Orphan Rule Constraints

Rust's orphan rule requires trait impls to be in the same crate as either the trait or at least one of the covered types. Rather than solving this as a post-hoc hitting set problem, the algorithm uses **synthetic back-edges**: before condensation, an edge is added from each impl's best anchor back to the impl. This creates a cycle that condensation collapses into a single SCC, guaranteeing the constraint is satisfied structurally.

The anchor heuristic chooses the anchor with minimum in-degree (fewest incoming edges), as a rough proxy for "most niche" — merging with it affects fewer other SCCs.

## Usage

```rust
use tarjanize_condense::run;

// From reader/writer (CLI pattern)
let input = std::fs::File::open("symbol_graph.json").unwrap();
let mut output = Vec::new();
run(input, &mut output).unwrap();
```

CLI:
```bash
tarjanize condense -i symbol_graph.json -o optimized_symbol_graph.json
tarjanize condense < symbol_graph.json > optimized_symbol_graph.json
```

## Dependencies

- `tarjanize-schemas` — Schema types (`SymbolGraph`, `Package`, `Crate`, `Module`, `Symbol`)
- `petgraph` — Condensation algorithm and `UnionFind`
- `indexmap` — `IndexSet` for bidirectional path↔index lookup
- `serde_json` — JSON serialization/deserialization
- `tracing` — Structured logging

## Design Decisions

1. **Synthetic back-edges for anchors** — Instead of a separate post-hoc hitting set phase, anchor constraints are encoded as graph edges before condensation. This is simpler and produces the same result.
2. **Incremental HMR** — Transitive reduction is recomputed after each union-find merge, because merging changes which edges are redundant at the set level. Without this, boundary detection would be incorrect.
3. **IndexSet for bidirectional lookup** — `IndexSet<String>` for paths provides both index→path and path→index lookup without data duplication, satisfying petgraph's `NodeId = usize` requirement.
4. **Sans-IO design** — `run()` accepts `impl Read` and `impl Write` for testability and composability.
5. **Same output type** — Outputs `SymbolGraph` (same as input) rather than a separate schema type. This simplifies downstream consumers and allows chaining phases.
