# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo check                     # Fast compile check (no object code)
cargo build                     # Full build
cargo nextest run               # Run all tests
cargo nextest run <name>        # Run tests matching name
cargo nextest run -E 'test(foo)'  # Run tests matching expression
cargo fmt                       # Format code (80-char lines, edition 2024)
cargo clippy                    # Lint
```

## Project Overview

**tarjanize** analyzes Rust workspace dependency structures to identify opportunities for splitting crates into smaller, parallelizable units for improved build times. Named after Robert Tarjan (SCC algorithms).

## Architecture

The tool implements a 5-phase pipeline:

```
Phase 1: Extract Symbol Graph        → symbol_graph.json
Phase 2: Compute SCCs, Condense      → condensed_graph.json
Phase 3: Optimal Partitioning        → optimized_condensed_graph.json
Phase 4: Reorganize Symbol Graph     → optimized_symbol_graph.json
Phase 5: Generate Report             → report.md
```

**Phase 1 is in progress**. See PLAN.md for full specification.

## Code Style

**Comments are mandatory.** Every function, struct, enum, and non-trivial code block must have comments that explain:
1. **What** the code does
2. **Why** it's necessary - the reasoning, constraints, or design decisions that led to this approach

Don't just describe the mechanics; justify the existence and design of the code.

## Module Structure

- **main.rs** - Entry point; loads workspace, iterates crates
- **workspaces.rs** - `load_workspace()` configures rust-analyzer with proc macro expansion and build.rs analysis
- **crates.rs** - `visit_crate()` enumerates modules in a crate
- **modules.rs** - `visit_module()` collects symbols and their dependencies via `SymbolCollector`
- **dependencies.rs** - Core dependency extraction: `find_dependencies()` walks syntax trees, `is_local()` filters to workspace members

## Key Patterns

**Semantic Resolution**: Uses rust-analyzer's `Semantics<RootDatabase>` for name resolution and type inference. Key types: `Crate`, `Module`, `ModuleDef`.

**Source Tree Navigation**: Gets syntax via `HasSource::source()`, registers with Semantics via `parse_or_expand()`, then finds equivalent node via `find_node_in_file()` for walking.

**Container Collapsing**: Associated items (impl methods, trait methods, enum variants) collapse to their containers since they can't be split independently.

**Local Filtering**: Only workspace members are analyzed; external crates are filtered out via `CrateOrigin::is_local()`.

## Testing

Test fixture in `tests/fixtures/method_resolution/` - a minimal Cargo project testing method resolution (trait defaults vs impl overrides).
