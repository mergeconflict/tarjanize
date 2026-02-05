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

This project uses [jj (Jujutsu)](https://github.com/martinvonz/jj) for version control. Use `jj` commands instead of `git` (e.g., `jj describe`, `jj new`, `jj st`).

## Project Overview

**tarjanize** analyzes Rust workspace dependency structures to identify opportunities for splitting crates into smaller, parallelizable units for improved build times. Named after Robert Tarjan (SCC algorithms).

## Architecture

The tool implements a multi-phase pipeline:

```
Phase 1: Extract Symbol Graph          → symbol_graph.json  (cargo tarjanize)
Phase 2: SCC + Partition + Reorganize  → optimized_symbol_graph.json  (tarjanize condense)
Cost:    Critical Path Analysis        → report  (tarjanize cost)
```

Phase 1 extracts the symbol graph. Phase 2 computes SCCs, merges them via union-find into optimal crate groupings (respecting orphan rule anchor constraints), and outputs an optimized `SymbolGraph`. The cost command computes build metrics including critical path with rmeta pipelining.

See PLAN.md for the full specification and future phases.

## Code Style

**Always use `/rust-development`** when writing or reviewing Rust code. This skill enforces Microsoft Pragmatic Rust Guidelines. Ask the project maintainer for the skill files.

**Comments are mandatory.** Every function, struct, enum, and non-trivial code block must have comments that explain:
1. **What** the code does
2. **Why** it's necessary - the reasoning, constraints, or design decisions that led to this approach

Don't just describe the mechanics; justify the existence and design of the code.

## Workspace Structure

This is a pure virtual workspace - all crates live under `crates/`. Dependencies are centralized in `[workspace.dependencies]` and inherited via `.workspace = true`.

```
tarjanize/
├── Cargo.toml               # Virtual workspace manifest (dependencies, lints)
└── crates/
    ├── tarjanize/           # CLI binary (condense, cost subcommands)
    │   └── src/main.rs
    │
    ├── tarjanize-schemas/   # Schema definitions (SymbolGraph types)
    │   └── src/
    │       ├── lib.rs
    │       ├── symbol_graph.rs
    │       └── testutil.rs  # proptest strategies
    │
    ├── cargo-tarjanize/     # Phase 1: Symbol graph extraction via rustc
    │   ├── src/
    │   │   ├── main.rs      # CLI entry point (orchestrator/driver modes)
    │   │   ├── orchestrator.rs  # Coordinates cargo check with RUSTC_WRAPPER
    │   │   ├── driver.rs    # Custom rustc driver for symbol extraction
    │   │   ├── extract.rs   # HIR/THIR analysis for dependency extraction
    │   │   ├── profile.rs   # Self-profile parsing for cost estimation
    │   │   └── mono_items.rs  # CGU→symbol mapping for backend cost distribution
    │   └── tests/fixtures/  # Integration test fixtures
    │
    ├── tarjanize-condense/  # Phase 2: SCC + partition + reorganize
    │   └── src/
    │       ├── lib.rs       # Public API: run()
    │       ├── error.rs     # CondenseError
    │       └── scc.rs       # SCC, union-find merging, anchor constraint fixing
    │
    └── tarjanize-cost/      # Cost analysis: critical path with rmeta pipelining
        └── src/
            └── lib.rs       # critical_path(), run(), CriticalPathResult
```

## Crate Details

**tarjanize** (binary)
- **main.rs** - CLI with `condense` and `cost` subcommands, common I/O args pattern

**tarjanize-schemas** (library)
- **symbol_graph.rs** - `SymbolGraph`, `Package`, `Crate`, `Module`, `Symbol`, `SymbolKind`, `Visibility` types
- **testutil.rs** - Shared proptest strategies for generating arbitrary schema instances

**cargo-tarjanize** (binary)
- **main.rs** - Detects mode (orchestrator vs driver) and dispatches
- **orchestrator.rs** - Runs `cargo check` with `RUSTC_WRAPPER` set to this binary
- **driver.rs** - Custom rustc driver that runs extraction callbacks
- **extract.rs** - Walks rustc's HIR and THIR to extract symbols and dependencies
- **profile.rs** - Parses `-Zself-profile` output for accurate compilation costs
- **mono_items.rs** - Parses `-Zprint-mono-items` output for CGU→symbol mapping (backend cost distribution)

**tarjanize-condense** (library)
- **lib.rs** - Public API: `run()` reads SymbolGraph JSON, outputs optimized SymbolGraph JSON
- **error.rs** - `CondenseError` for deserialization, serialization, and I/O failures
- **scc.rs** - SCC computation via petgraph condensation, union-find merging, and anchor constraint fixing (orphan rule)

**tarjanize-cost** (library)
- **lib.rs** - Critical path analysis with rmeta pipelining. Models frontend (serial), backend (parallel CGUs), and overhead costs. Target-level analysis avoids dev-dependency cycles.

## Key Patterns

**Rustc Driver**: `cargo-tarjanize` acts as a `RUSTC_WRAPPER`, intercepting rustc invocations. For workspace crates, it runs our extraction callbacks in `after_analysis`. For external crates, it passes through to normal compilation.

**THIR Analysis**: Uses THIR (Typed High-level IR) for body analysis because it preserves source-level information (static refs, named consts, const patterns) that MIR loses.

**Container Collapsing**: Associated items (impl methods, trait methods, enum variants) collapse to their containers since they can't be split independently.

**Profile Matching**: For accurate cost estimation, `--profile` runs with `-Zself-profile`. The profile uses compiler internal paths (`{{impl}}[N]`) which are stored in `Symbol.profile_key` for matching.

**Target-Level Analysis**: The cost model operates at the compilation target level (`{package}/{target}` format, e.g. `my-pkg/lib`, `my-pkg/test`), not the package level. This naturally resolves dev-dependency "cycles" since test targets depend on lib targets, not vice versa.

**Rmeta Pipelining**: The cost model accounts for Cargo's pipelined compilation — downstream targets can start frontend work when upstream rmeta is ready (after frontend), without waiting for backend codegen to complete.

**Anchor Constraints**: Impl blocks have "anchors" (the trait and self type). When partitioning, the orphan rule requires at least one anchor to be in the same crate as the impl. The condense phase uses a hitting set algorithm to find minimal merges satisfying all constraints.

## Testing

**Coverage requirement:** All modules except `main.rs` must maintain ≥90% line coverage. Run `cargo llvm-cov nextest` to check. The `main.rs` file is excluded because it's the CLI entry point and not exercised by unit tests.

### Integration Test Fixtures

Real Cargo projects in `tests/fixtures/` are used for integration tests. Each fixture must have an empty `[workspace]` table in its Cargo.toml to prevent being detected as part of the parent workspace.

## Static Verification

Workspace-level lint configuration follows [M-STATIC-VERIFICATION](https://microsoft.github.io/rust-guidelines/guidelines/universal/index.html) from the Microsoft Pragmatic Rust Guidelines. See `[workspace.lints]` in the root `Cargo.toml` for the full configuration.

**Prefer `#[expect]` over `#[allow]`** for suppressing lints. `#[expect]` warns if the lint is never triggered, catching stale suppressions.

**CI runs with `RUSTFLAGS=-Dwarnings`**, so all warnings are errors. Test fixtures must use `#[expect(...)]` for intentional lint violations.

Beyond `cargo clippy` and `cargo fmt`, consider running these tools periodically:

```bash
cargo audit                     # Check for security vulnerabilities in deps
cargo udeps                     # Find unused dependencies
cargo hack check --feature-powerset  # Verify all feature combinations compile
```

## Documentation

Read these docs for deeper context on specific topics:

- **[PLAN.md](PLAN.md)** — Full project specification: algorithm details, optimality proofs, phase design
- **[docs/cost-model-validation.md](docs/cost-model-validation.md)** — Cost model validation against Omicron (~160 crates). Documents R²=0.856 accuracy, rmeta pipelining discovery, per-symbol cost skew (top 1% = 75% of cost), and why only lib targets matter for critical path
- **[COMPILATION_COSTS.md](COMPILATION_COSTS.md)** — How rustc compilation works: frontend/backend/overhead cost breakdown, CGU parallelism, what profiling data is available
- **[docs/validation-plan.md](docs/validation-plan.md)** — Plan to validate cost model across 10 popular Rust workspaces (Zed, Bevy, uv, ruff, etc.)
- **[docs/external-profile-plan.md](docs/external-profile-plan.md)** — Two-pass profiling plan to fix 10-15x cost inflation from extraction callback overhead
- **[PLAN_COST_TRACKING.md](PLAN_COST_TRACKING.md)** — Implementation plan for frontend/backend cost separation in schemas
- **[PLAN_TARGETS.md](PLAN_TARGETS.md)** — Implementation plan for separating lib/test/bin targets to fix dev-dependency cycles
- **[TARJAN.md](TARJAN.md)** — Reference: Tarjan's SCC algorithm explained
- **[scripts/README.md](scripts/README.md)** — Python analysis scripts for comparing model predictions vs actual cargo build times
