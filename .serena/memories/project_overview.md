# Project Overview

**tarjanize** analyzes Rust workspace dependency structures to identify crate splitting opportunities for improved build parallelism.

## Tech Stack
- Rust (edition 2024), virtual workspace under `crates/`
- `petgraph` for graph algorithms, `serde`/`serde_json` for serialization
- `axum` web framework for interactive server mode
- `askama` for HTML templates, `PixiJS` + `d3-force` for frontend
- `jj` (Jujutsu) for version control

## Workspace Crates
- `cargo-tarjanize` — Rustc driver for symbol extraction
- `tarjanize` — CLI binary
- `tarjanize-schemas` — SymbolGraph, CostModel types (crate name uses underscore: `tarjanize_schemas`)
- `tarjanize-schedule` — Schedule computation, split logic, target graph, heatmap
- `tarjanize-viz` — Web server + HTML visualization
- `tarjanize-cost` — Cost model fitting
- `tarjanize-magsac` — MAGSAC++ regression
- `tarjanize-condense` — SCC + partition

## Code Style
- Comments mandatory on all functions/structs (what + why)
- `#[expect]` over `#[allow]` for lint suppression
- All code must pass `cargo clippy --all-targets`
- 90%+ line coverage for non-main.rs modules
