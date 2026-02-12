# Suggested Commands

## Build & Test
- `cargo check` — Fast compile check
- `cargo build` — Full build
- `cargo nextest run` — Run all tests
- `cargo nextest run -p tarjanize-schedule` — Run tests for a specific crate
- `cargo test --doc` — Run doc tests
- `cargo llvm-cov nextest` — Tests with coverage

## Code Quality
- `cargo fmt` — Format code (80-char lines, edition 2024)
- `cargo clippy --all-targets` — Lint (must pass without warnings)

## Version Control (jj, NOT git)
- `jj st` — Status
- `jj diff --git` — Show changes
- `jj describe -m "message"` — Set commit message
- `jj new` — Create a new empty change
- `jj log` — Show commit history

## Task Completion Checklist
1. `cargo nextest run -p <crate>` — Tests pass
2. `cargo clippy --all-targets` — No warnings
3. `cargo fmt` — Code formatted
