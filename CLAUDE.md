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

## Workspace Structure

This is a pure virtual workspace - all crates live under `crates/`.

```
tarjanize/
├── Cargo.toml               # Virtual workspace manifest
└── crates/
    ├── tarjanize/           # CLI binary
    │   └── src/main.rs
    │
    ├── tarjanize-schemas/   # Schema definitions for all phases
    │   ├── src/
    │   │   ├── lib.rs
    │   │   └── symbol_graph.rs
    │   └── schemas/         # Golden JSON Schema files
    │       └── symbol_graph.schema.json
    │
    └── tarjanize-extract/   # Phase 1: Symbol graph extraction
        ├── src/
        │   ├── lib.rs       # Public API: run(), extract_symbol_graph(), load_workspace()
        │   ├── error.rs     # ExtractError (per M-ERRORS-CANONICAL-STRUCTS)
        │   ├── workspaces.rs
        │   ├── crates.rs
        │   ├── modules.rs
        │   └── dependencies.rs
        └── tests/fixtures/  # Integration test fixtures
```

## Crate Details

**tarjanize** (binary)
- **main.rs** - CLI entry point; parses args, calls `tarjanize_extract::run()`

**tarjanize-schemas** (library)
- **symbol_graph.rs** - `SymbolGraph`, `Module`, `Symbol`, `SymbolKind`, `Edge` types with serde and JSON Schema support

**tarjanize-extract** (library)
- **lib.rs** - Public API: `run()`, `extract_symbol_graph()`, re-exports schema types
- **error.rs** - `ExtractError` with backtrace, `ErrorKind` enum, and `is_xxx()` helpers
- **workspaces.rs** - `load_workspace()` configures rust-analyzer with proc macro expansion and build.rs analysis
- **crates.rs** - `extract_crate()` extracts a crate as its root module
- **modules.rs** - `extract_module()` collects symbols recursively via `SymbolCollector`
- **dependencies.rs** - `find_dependencies()` walks syntax trees, `is_local()` filters to workspace members

## Key Patterns

**Semantic Resolution**: Uses rust-analyzer's `Semantics<RootDatabase>` for name resolution and type inference. Key types: `Crate`, `Module`, `ModuleDef`.

**Source Tree Navigation**: Gets syntax via `HasSource::source()`, registers with Semantics via `parse_or_expand()`, then finds equivalent node via `find_node_in_file()` for walking.

**Container Collapsing**: Associated items (impl methods, trait methods, enum variants) collapse to their containers since they can't be split independently.

**Local Filtering**: Only workspace members are analyzed; external crates are filtered out via `CrateOrigin::is_local()`.

## Testing

**Coverage requirement:** All modules except `main.rs` must maintain ≥90% line coverage. Run `cargo llvm-cov nextest` to check. The `main.rs` file is excluded because it's the CLI entry point and not exercised by unit tests.

### Fixture-Based Unit Tests

Tests use `ra_ap_test_fixture` for fast, in-memory fixture-based testing. Each test creates an in-memory database with the fixture syntax:
```rust
let db = RootDatabase::with_files(r#"
//- /lib.rs crate:test_crate deps:other_crate
pub fn my_function() {}
"#);
```

Test files target one property at a time (e.g., `test_fixture_fn_param_type`, `test_fixture_trait_supertrait`).

### Golden File Testing

Schema files are verified against golden files to ensure stability. To update after intentional schema changes:
```bash
GENERATE_GOLDEN=1 cargo nextest run schema_matches_golden_file
```

### Integration Test Fixtures

Real Cargo projects in `tests/fixtures/` are used for integration tests. Each fixture must have an empty `[workspace]` table in its Cargo.toml to prevent being detected as part of the parent workspace.

## Static Verification

Beyond `cargo clippy` and `cargo fmt`, consider running these tools periodically:

```bash
cargo audit                     # Check for security vulnerabilities in deps
cargo udeps                     # Find unused dependencies
cargo hack check --feature-powerset  # Verify all feature combinations compile
```
