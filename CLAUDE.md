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
Phase 1: Extract Symbol Graph        → symbol_graph.json  (cargo tarjanize)
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
    ├── tarjanize/           # CLI binary (post-processing commands)
    │   └── src/main.rs
    │
    ├── tarjanize-schemas/   # Schema definitions for all phases
    │   └── src/
    │       ├── lib.rs
    │       ├── symbol_graph.rs
    │       ├── condensed_graph.rs
    │       └── testutil.rs
    │
    ├── cargo-tarjanize/     # Phase 1: Symbol graph extraction via rustc
    │   ├── src/
    │   │   ├── main.rs      # CLI entry point (orchestrator/driver modes)
    │   │   ├── orchestrator.rs  # Coordinates cargo check with RUSTC_WRAPPER
    │   │   ├── driver.rs    # Custom rustc driver for symbol extraction
    │   │   ├── extract.rs   # HIR/THIR analysis for dependency extraction
    │   │   └── profile.rs   # Self-profile parsing for cost estimation
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
- **main.rs** - CLI entry point for post-processing commands (condense phase)

**tarjanize-schemas** (library)
- **symbol_graph.rs** - `SymbolGraph`, `Module`, `Symbol`, `SymbolKind`, `Visibility` types
- **condensed_graph.rs** - `CondensedGraph`, `Scc`, `AnchorSet` types for Phase 2 output
- **testutil.rs** - Shared proptest strategies for generating arbitrary schema instances

**cargo-tarjanize** (binary)
- **main.rs** - Detects mode (orchestrator vs driver) and dispatches
- **orchestrator.rs** - Runs `cargo check` with `RUSTC_WRAPPER` set to this binary
- **driver.rs** - Custom rustc driver that runs extraction callbacks
- **extract.rs** - Walks rustc's HIR and THIR to extract symbols and dependencies
- **profile.rs** - Parses `-Zself-profile` output for accurate compilation costs

**tarjanize-condense** (library)
- **lib.rs** - Public API: `run()` reads SymbolGraph JSON, outputs CondensedGraph JSON
- **error.rs** - `CondenseError` for deserialization and I/O failures
- **scc.rs** - `compute_condensed_graph()` uses petgraph's condensation algorithm

## Key Patterns

**Rustc Driver**: `cargo-tarjanize` acts as a `RUSTC_WRAPPER`, intercepting rustc invocations. For workspace crates, it runs our extraction callbacks in `after_analysis`. For external crates, it passes through to normal compilation.

**THIR Analysis**: Uses THIR (Typed High-level IR) for body analysis because it preserves source-level information (static refs, named consts, const patterns) that MIR loses.

**Container Collapsing**: Associated items (impl methods, trait methods, enum variants) collapse to their containers since they can't be split independently.

**Profile Matching**: For accurate cost estimation, `--profile` runs with `-Zself-profile`. The profile uses compiler internal paths (`{{impl}}[N]`) which are stored in `Symbol.profile_key` for matching.

## Testing

**Coverage requirement:** All modules except `main.rs` must maintain ≥90% line coverage. Run `cargo llvm-cov nextest` to check. The `main.rs` file is excluded because it's the CLI entry point and not exercised by unit tests.

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
