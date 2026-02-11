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

## Pipeline

```
cargo tarjanize        → symbol_graph.json             Extract symbol graph from workspace
tarjanize cost         → report + cost_model.json       Fit cost model, critical path analysis
tarjanize viz          → interactive HTML               Visualize the original build schedule
tarjanize condense     → optimized_symbol_graph.json    SCC + partition + reorganize
tarjanize viz          → interactive HTML               Visualize the condensed build schedule
```

`cargo tarjanize` extracts symbols and dependencies from a Rust workspace via a custom rustc driver. `tarjanize cost` fits a MAGSAC++ regression model to profile data and reports the critical path. `tarjanize viz` generates an interactive PixiJS Gantt chart of the build schedule. `tarjanize condense` computes SCCs, merges them via union-find (respecting orphan rule anchor constraints), and outputs an optimized `SymbolGraph`.

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
├── Cargo.toml                 # Virtual workspace manifest (dependencies, lints)
└── crates/
    ├── cargo-tarjanize/       # Phase 1: Symbol graph extraction via rustc
    │   ├── src/
    │   │   ├── main.rs        # CLI entry point (orchestrator/driver modes)
    │   │   ├── orchestrator.rs  # Coordinates cargo check with RUSTC_WRAPPER
    │   │   ├── driver.rs      # Custom rustc driver for symbol extraction
    │   │   ├── extract.rs     # HIR/THIR analysis for dependency extraction
    │   │   └── profile.rs     # Self-profile parsing for cost estimation
    │   └── tests/fixtures/    # Integration test fixtures (~207 directories)
    │
    ├── tarjanize/             # CLI binary (condense, cost, viz subcommands)
    │   └── src/main.rs
    │
    ├── tarjanize-schemas/     # Schema definitions (SymbolGraph, CostModel types)
    │   └── src/
    │       ├── lib.rs
    │       ├── symbol_graph.rs
    │       ├── cost_model.rs
    │       └── testutil.rs    # proptest strategies
    │
    ├── tarjanize-condense/    # SCC + partition + reorganize
    │   └── src/
    │       ├── lib.rs         # Public API: run()
    │       ├── error.rs       # CondenseError
    │       └── scc.rs         # SCC, union-find merging, anchor constraint fixing
    │
    ├── tarjanize-cost/        # Cost model fitting and critical path analysis
    │   └── src/
    │       └── lib.rs         # fit(), build_cost_model(), extract_predictors()
    │
    ├── tarjanize-magsac/      # MAGSAC++ robust N-variable linear regression
    │   └── src/
    │       └── lib.rs         # fit_magsac(), fit_magsac_with_params()
    │
    └── tarjanize-viz/         # Interactive HTML build schedule visualization
        └── src/
            ├── lib.rs         # Public API: run()
            ├── data.rs        # ScheduleData, TargetData structures
            ├── error.rs       # VizError
            ├── html.rs        # Askama HTML template generation
            └── schedule.rs    # Forward/backward DP, swim lane packing
```

## Crate Details

**cargo-tarjanize** (binary)
- **main.rs** - Detects mode (orchestrator vs driver) and dispatches
- **orchestrator.rs** - Runs `cargo check` with `RUSTC_WRAPPER` set to this binary. Two-pass strategy: clean profile pass with `-Zself-profile`, then extraction pass
- **driver.rs** - Custom rustc driver that runs extraction callbacks in `after_analysis`
- **extract.rs** - Walks rustc's HIR and THIR to extract symbols and dependencies
- **profile.rs** - Parses `-Zself-profile` output for accurate compilation costs

**tarjanize** (binary)
- **main.rs** - CLI with `condense`, `cost`, and `viz` subcommands. Common `IoArgs` pattern for input/output

**tarjanize-schemas** (library)
- **symbol_graph.rs** - `SymbolGraph`, `Package`, `Target` (alias `Crate`), `Module`, `Symbol`, `SymbolKind`, `Visibility`
- **cost_model.rs** - `CostModel` (serializable 3-variable regression model), `load_cost_model()`
- **testutil.rs** - Shared proptest strategies for generating arbitrary schema instances

**tarjanize-condense** (library)
- **lib.rs** - `run()` reads SymbolGraph JSON, outputs optimized SymbolGraph JSON
- **error.rs** - `CondenseError` for deserialization, serialization, and I/O failures
- **scc.rs** - SCC computation via petgraph condensation, union-find merging, and hitting set algorithm for anchor constraints (orphan rule)

**tarjanize-cost** (library)
- **lib.rs** - `fit()` uses MAGSAC++ to fit a 3-variable no-intercept regression model (`wall_time = c_attr * attr + c_meta * meta + c_other * other`). Also: `build_cost_model()`, `extract_predictors()`, critical path analysis

**tarjanize-magsac** (library)
- **lib.rs** - MAGSAC++ robust regression: `fit_magsac()`, `fit_magsac_with_params()`. Sigma consensus for outlier-robust fitting without manual threshold tuning

**tarjanize-viz** (library)
- **lib.rs** - `run()` generates self-contained HTML with PixiJS canvas Gantt chart
- **schedule.rs** - Forward/backward DP for critical path, swim lane packing for parallelism visualization

## Key Patterns

**Rustc Driver**: `cargo-tarjanize` acts as a `RUSTC_WRAPPER`, intercepting rustc invocations. For workspace crates, it runs our extraction callbacks in `after_analysis`. For external crates, it passes through to normal compilation.

**Two-Pass Profiling**: The orchestrator runs two separate `cargo check` passes — first a clean nightly build with `-Zself-profile` (no extraction overhead), then an extraction pass with the RUSTC_WRAPPER. This avoids 10-15x cost inflation from extraction callbacks.

**THIR Analysis**: Uses THIR (Typed High-level IR) for body analysis because it preserves source-level information (static refs, named consts, const patterns) that MIR loses.

**Container Collapsing**: Associated items (impl methods, trait methods, enum variants) collapse to their containers since they can't be split independently.

**Profile Matching**: For accurate cost estimation, profiles use compiler internal paths (`{{impl}}[N]`) which are stored in `Symbol.profile_key` for matching.

**Target-Level Analysis**: The cost model operates at the compilation target level (`{package}/{target}` format, e.g. `my-pkg/lib`, `my-pkg/test`), not the package level. This naturally resolves dev-dependency "cycles" since test targets depend on lib targets, not vice versa.

**Anchor Constraints**: Impl blocks have "anchors" (the trait and self type). When partitioning, the orphan rule requires at least one anchor to be in the same crate as the impl. The condense phase uses a hitting set algorithm to find minimal merges satisfying all constraints.

## Testing

**Coverage requirement:** All modules except `main.rs` must maintain >=90% line coverage. Run `cargo llvm-cov nextest` to check. The `main.rs` files are excluded because they're CLI entry points not exercised by unit tests.

### Integration Test Fixtures

~207 real Cargo projects in `crates/cargo-tarjanize/tests/fixtures/` are used for integration tests. Each fixture must have an empty `[workspace]` table in its Cargo.toml to prevent being detected as part of the parent workspace. Fixtures cover symbol paths, visibility, cross-crate dependencies, anchors, profile keys, cost extraction, and edge cases.

## Static Verification

Workspace-level lint configuration follows [M-STATIC-VERIFICATION](https://microsoft.github.io/rust-guidelines/guidelines/universal/index.html) from the Microsoft Pragmatic Rust Guidelines. See `[workspace.lints]` in the root `Cargo.toml` for the full configuration.

**Prefer `#[expect]` over `#[allow]`** for suppressing lints. `#[expect]` warns if the lint is never triggered, catching stale suppressions.

**CI runs with `RUSTFLAGS=-Dwarnings`**, so all warnings are errors. Test fixtures must use `#[expect(...)]` for intentional lint violations.

## Documentation

Read these docs for deeper context on specific topics:

- **[PLAN.md](PLAN.md)** — Full project specification: algorithm details, optimality proofs, phase design
- **[docs/cost-model-validation.md](docs/cost-model-validation.md)** — Cost model validation against Omicron (~160 crates). Documents R^2=0.856 accuracy, per-symbol cost skew, and why only lib targets matter for critical path
- **[COMPILATION_COSTS.md](COMPILATION_COSTS.md)** — Reference: how rustc compilation works (frontend/backend phases, CGU parallelism, profiling data). tarjanize only tracks frontend costs
- **[docs/structural-cost-predictors.md](docs/structural-cost-predictors.md)** — Analysis of which code structural properties drive compilation time
- **[docs/external-profile-plan.md](docs/external-profile-plan.md)** — Two-pass profiling plan to fix extraction callback overhead
- **[docs/critical-path-pruning.md](docs/critical-path-pruning.md)** — Algorithm for identifying unnecessary edges in dependency DAG
- **[scripts/README.md](scripts/README.md)** — Python analysis scripts for comparing model predictions vs actual cargo build times
