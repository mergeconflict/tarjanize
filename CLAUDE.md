# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo check                     # Fast compile check (no object code)
cargo build                     # Full build
cargo nextest run               # Run all tests
cargo nextest run <name>        # Run tests matching name
cargo nextest run -E 'test(foo)'  # Run tests matching expression
cargo llvm-cov nextest          # Run tests with coverage report
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

**Coverage requirement:** All modules except `main.rs` must maintain ≥90% line coverage. Run `cargo llvm-cov nextest` to check. The `main.rs` file is excluded because it's the CLI entry point and not exercised by unit tests.

Tests use `ra_ap_test_fixture` for fast, in-memory fixture-based testing. Each test creates an in-memory database with the fixture syntax:
```rust
let db = RootDatabase::with_files(r#"
//- /lib.rs crate:test_crate deps:other_crate
pub fn my_function() {}
"#);
```

Test files target one property at a time (e.g., `test_fixture_fn_param_type`, `test_fixture_trait_supertrait`).

## Static Verification

Beyond `cargo clippy` and `cargo fmt`, consider running these tools periodically:

```bash
cargo audit                     # Check for security vulnerabilities in deps
cargo udeps                     # Find unused dependencies
cargo hack check --feature-powerset  # Verify all feature combinations compile
```
