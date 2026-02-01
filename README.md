# tarjanize

Analyze Rust workspace dependency structures to identify opportunities for splitting crates into smaller, parallelizable units for improved build times.

Named after [Robert Tarjan](https://en.wikipedia.org/wiki/Robert_Tarjan), inventor of the strongly connected components algorithm.

## The Problem

Large Rust crates compile slowly because rustc processes them as a single unit. Even with incremental compilation, a change to one function can trigger recompilation of the entire crate. Splitting a crate into smaller pieces enables parallel compilation, but manually identifying safe split points is tedious and error-prone.

## The Solution

tarjanize analyzes your workspace at the symbol level, finding:

- **Strongly connected components (SCCs)**: Groups of symbols that mutually depend on each other and must stay together
- **Safe split points**: Where the dependency graph can be cleanly partitioned
- **Optimal partitions**: Groupings that maximize parallelism while respecting Rust's orphan rules

## Installation

Requires Rust nightly (uses internal rustc APIs for analysis).

```bash
cargo install --git https://github.com/mergeconflict/tarjanize cargo-tarjanize
cargo install --git https://github.com/mergeconflict/tarjanize tarjanize
```

## Usage

### Step 1: Extract the symbol graph

```bash
cd your-workspace
cargo +nightly tarjanize -o symbol_graph.json
```

This analyzes your workspace and outputs a JSON file containing all symbols and their dependencies.

Options:
- `-p, --package <SPEC>`: Analyze specific packages only
- `--profile`: Include compilation cost estimates (requires nightly)
- `-o, --output <FILE>`: Output file (default: stdout)

### Step 2: Compute the condensed graph

```bash
tarjanize condense symbol_graph.json -o condensed_graph.json
```

This computes SCCs and produces a DAG showing how symbols can be grouped.

## How It Works

tarjanize implements a multi-phase pipeline:

```
Phase 1: Extract Symbol Graph        cargo tarjanize → symbol_graph.json
Phase 2: Compute SCCs, Condense      tarjanize condense → condensed_graph.json
Phase 3: Optimal Partitioning        (planned)
Phase 4: Reorganize Symbol Graph     (planned)
Phase 5: Generate Report             (planned)
```

**Phase 1** uses a custom rustc driver to walk the HIR and THIR, extracting every symbol (functions, structs, traits, impls, etc.) and their dependencies. It captures impl block "anchors" needed for orphan rule compliance.

**Phase 2** runs Tarjan's algorithm to find SCCs, then condenses the graph into a DAG. Each node represents symbols that must stay together; edges represent dependencies between groups.

## Current Status

Phases 1 and 2 are implemented. The tool can analyze real workspaces and produce condensed graphs, but doesn't yet generate actionable refactoring suggestions.

## License

MIT
