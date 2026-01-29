# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo check                     # Fast compile check (no object code)
cargo build                     # Full build
cargo nextest run               # Run all tests
cargo nextest run <name>        # Run tests matching name
cargo nextest run -E 'test(foo)'  # Run tests matching expression
cargo test --doc                # Run doc tests (nextest doesn't run these)
cargo llvm-cov nextest          # Run tests with coverage report
cargo fmt                       # Format code (80-char lines, edition 2024)
cargo clippy --all-targets      # Lint (--all-targets required in virtual workspaces)
```

**All code must pass `cargo clippy --all-targets` without warnings.** Fix any warnings before committing.

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

**Phase 2 is in progress**. See PLAN.md for full specification.

## Code Style

**Always use `/rust-development`** when writing or reviewing Rust code. This skill enforces Microsoft Pragmatic Rust Guidelines. Ask the project maintainer for the skill files.

**Comments are mandatory.** Every function, struct, enum, and non-trivial code block must have comments that explain:
1. **What** the code does
2. **Why** it's necessary - the reasoning, constraints, or design decisions that led to this approach

Don't just describe the mechanics; justify the existence and design of the code.

## Workspace Structure

This is a pure virtual workspace - all crates live under `crates/`.

```
tarjanize/
├── Cargo.toml               # Virtual workspace manifest (includes lint config)
└── crates/
    ├── tarjanize/           # CLI binary
    │   └── src/main.rs
    │
    ├── tarjanize-schemas/   # Schema definitions for all phases
    │   └── src/
    │       ├── lib.rs
    │       ├── symbol_graph.rs
    │       ├── condensed_graph.rs
    │       └── testutil.rs
    │
    ├── tarjanize-extract/   # Phase 1: Symbol graph extraction
    │   ├── src/
    │   │   ├── lib.rs       # Public API: run()
    │   │   ├── error.rs     # ExtractError
    │   │   ├── workspaces.rs
    │   │   ├── crates.rs
    │   │   ├── modules.rs
    │   │   ├── module_defs.rs
    │   │   ├── impls.rs
    │   │   ├── dependencies.rs
    │   │   └── paths.rs
    │   ├── doc/rust-analyzer/  # rust-analyzer API documentation
    │   └── tests/fixtures/  # Integration test fixtures
    │
    └── tarjanize-condense/  # Phase 2: SCC computation
        └── src/
            ├── lib.rs       # Public API: run()
            ├── error.rs     # CondenseError
            └── scc.rs       # SCC computation using petgraph
```

## Crate Details

**tarjanize** (binary)
- **main.rs** - CLI entry point; orchestrates extract and condense phases

**tarjanize-schemas** (library)
- **symbol_graph.rs** - `SymbolGraph`, `Module`, `Symbol`, `SymbolKind`, `Visibility` types
- **condensed_graph.rs** - `CondensedGraph`, `Scc`, `AnchorSet` types for Phase 2 output
- **testutil.rs** - Shared proptest strategies for generating arbitrary schema instances

**tarjanize-extract** (library)
- **lib.rs** - Public API: `run()`, re-exports `ExtractError` and schema types
- **error.rs** - `ExtractError` with backtrace, `ErrorKind` enum, and `is_xxx()` helpers
- **workspaces.rs** - `load_workspace()` configures rust-analyzer with proc macro expansion and build.rs analysis
- **crates.rs** - `extract_crate()` extracts a crate as its root module
- **modules.rs** - `extract_module()` recursively extracts symbols from module hierarchy
- **module_defs.rs** - `extract_module_def()` extracts ModuleDef items (functions, structs, etc.)
- **impls.rs** - `extract_impl()` extracts impl blocks with their dependencies
- **dependencies.rs** - `find_dependencies()` walks syntax trees using `NameRefClass::classify()`
- **paths.rs** - Path utilities: `qualified_path()`, `file_path()`, `relative_file_path()`

**tarjanize-condense** (library)
- **lib.rs** - Public API: `run()` reads SymbolGraph JSON, outputs CondensedGraph JSON
- **error.rs** - `CondenseError` for deserialization and I/O failures
- **scc.rs** - `compute_condensed_graph()` uses petgraph's condensation algorithm

## Key Patterns

**Semantic Resolution**: Uses rust-analyzer's `Semantics<RootDatabase>` for name resolution and type inference. Key types: `Crate`, `Module`, `ModuleDef`.

**Source Tree Navigation**: Uses `HasSource::source()` to get syntax nodes, then `sema.parse_or_expand()` to get a cached tree for resolution. See `doc/rust-analyzer/README.md` for details.

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

### Integration Test Fixtures

Real Cargo projects in `tests/fixtures/` are used for integration tests. Each fixture must have an empty `[workspace]` table in its Cargo.toml to prevent being detected as part of the parent workspace.

## Static Verification

Workspace-level lint configuration follows [M-STATIC-VERIFICATION](https://microsoft.github.io/rust-guidelines/guidelines/universal/index.html) from the Microsoft Pragmatic Rust Guidelines. See `[workspace.lints]` in the root `Cargo.toml` for the full configuration.

Beyond `cargo clippy` and `cargo fmt`, consider running these tools periodically:

```bash
cargo audit                     # Check for security vulnerabilities in deps
cargo udeps                     # Find unused dependencies
cargo hack check --feature-powerset  # Verify all feature combinations compile
```
