# Interactive Split Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive web app where developers visually explore crate splits, see their effect on the build schedule in real time, and export the resulting symbol graph.

**Architecture:** Extract scheduling primitives into `tarjanize-schedule`, then extend `tarjanize-viz` from a static HTML generator into a local web app (axum backend + JS frontend). The backend handles all heavy computation (SCC condensation, schedule DP, convexity enforcement); the frontend renders a dual-panel UI (Gantt chart + force-directed compound DAG). Auto-fit cost model on startup.

**Tech Stack:** Rust (axum, tokio, petgraph, serde_json), JavaScript (PixiJS for Gantt, d3-force for compound DAG), esbuild bundling

---

## Task 1: Extract `tarjanize-schedule` crate — scaffolding

Create the new crate directory and Cargo.toml. Register it in the workspace.

**Files:**
- Create: `crates/tarjanize-schedule/Cargo.toml`
- Create: `crates/tarjanize-schedule/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

**Step 1: Create crate directory**

```bash
mkdir -p crates/tarjanize-schedule/src
```

**Step 2: Write `Cargo.toml`**

Create `crates/tarjanize-schedule/Cargo.toml`:

```toml
[package]
name = "tarjanize-schedule"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
keywords.workspace = true
categories.workspace = true
readme = "../../README.md"
description = "Build schedule computation: forward/backward DP, critical path, swim lanes"

[lints]
workspace = true

[dependencies]
indexmap.workspace = true
petgraph.workspace = true
serde.workspace = true
serde_json.workspace = true
```

**Step 3: Write initial `lib.rs`**

Create `crates/tarjanize-schedule/src/lib.rs` with just the module declarations (empty bodies — populated in next tasks):

```rust
//! Build schedule computation: forward/backward DP, critical path, swim lanes.
//!
//! Extracted from `tarjanize-viz` so both the static HTML visualization and the
//! interactive split explorer can share scheduling primitives.
```

**Step 4: Register in workspace**

Add `tarjanize-schedule = { path = "crates/tarjanize-schedule" }` to `[workspace.dependencies]` in the root `Cargo.toml`.

**Step 5: Run `cargo check` to verify the new crate compiles**

```bash
cargo check -p tarjanize-schedule
```

Expected: success (empty crate compiles fine).

**Step 6: Commit**

```bash
jj describe -m "Add tarjanize-schedule crate scaffold"
jj new
```

---

## Task 2: Move data types to `tarjanize-schedule`

Move `ScheduleData`, `Summary`, and `TargetData` from `tarjanize-viz/src/data.rs` to `tarjanize-schedule/src/lib.rs` (or a `data` submodule). Change visibility from `pub` to `pub`. Update `tarjanize-viz` to re-export or depend on the new crate.

**Files:**
- Modify: `crates/tarjanize-schedule/src/lib.rs`
- Create: `crates/tarjanize-schedule/src/data.rs`
- Modify: `crates/tarjanize-viz/src/data.rs`
- Modify: `crates/tarjanize-viz/Cargo.toml`

**Step 1: Create `crates/tarjanize-schedule/src/data.rs`**

Copy the contents of `crates/tarjanize-viz/src/data.rs` verbatim into `crates/tarjanize-schedule/src/data.rs`. All types are already `pub` and derive `Serialize` — no changes needed to the struct definitions.

**Step 2: Add `pub mod data;` to `crates/tarjanize-schedule/src/lib.rs`**

```rust
pub mod data;
```

**Step 3: Add `tarjanize-schedule` dependency to `tarjanize-viz/Cargo.toml`**

```toml
tarjanize-schedule.workspace = true
```

**Step 4: Replace `crates/tarjanize-viz/src/data.rs` with re-exports**

Replace the entire file with:

```rust
//! Schedule data types — re-exported from `tarjanize-schedule`.
//!
//! The canonical definitions live in `tarjanize_schedule::data`. This module
//! re-exports them so existing code in `tarjanize-viz` (html.rs, schedule.rs)
//! continues to compile without changing import paths.

pub use tarjanize_schedule::data::{ScheduleData, Summary, TargetData};
```

**Step 5: Run tests**

```bash
cargo nextest run -p tarjanize-viz
```

Expected: all existing tests pass.

**Step 6: Commit**

```bash
jj describe -m "Move schedule data types to tarjanize-schedule"
jj new
```

---

## Task 3: Move `TargetGraph` and scheduling functions to `tarjanize-schedule`

Move `TargetGraph`, `ForwardResult`, `BackwardResult`, `compute_schedule()`, `forward_pass()`, `backward_pass()`, and `pack_swim_lanes()` from `crates/tarjanize-viz/src/schedule.rs` to `crates/tarjanize-schedule/src/schedule.rs`. Make `TargetGraph` and `compute_schedule()` public. Update `tarjanize-viz` to import from the new location.

**Files:**
- Create: `crates/tarjanize-schedule/src/schedule.rs`
- Modify: `crates/tarjanize-schedule/src/lib.rs`
- Modify: `crates/tarjanize-viz/src/schedule.rs`
- Modify: `crates/tarjanize-viz/src/lib.rs`

**Step 1: Create `crates/tarjanize-schedule/src/schedule.rs`**

Copy the entire contents of `crates/tarjanize-viz/src/schedule.rs` (lines 1-683) into the new file. Make these changes:

1. Change `use crate::data::{...}` to `use crate::data::{ScheduleData, Summary, TargetData};`
2. Change `pub(crate) struct TargetGraph` to `pub struct TargetGraph`
3. Change all `pub(crate)` fields in `TargetGraph` to `pub`
4. Change `pub(crate) fn compute_schedule` to `pub fn compute_schedule`
5. Keep `ForwardResult`, `BackwardResult`, `forward_pass()`, `backward_pass()`, `pack_swim_lanes()` as private — only `TargetGraph` and `compute_schedule()` need to be public

**Step 2: Add `pub mod schedule;` to `crates/tarjanize-schedule/src/lib.rs`**

```rust
pub mod data;
pub mod schedule;
```

**Step 3: Replace `crates/tarjanize-viz/src/schedule.rs`**

Replace the entire file with re-exports plus the test helper:

```rust
//! Schedule computation — re-exported from `tarjanize-schedule`.
//!
//! The canonical implementations live in `tarjanize_schedule::schedule`.
//! Re-exported here so `lib.rs` continues to work without import changes.

pub use tarjanize_schedule::schedule::{TargetGraph, compute_schedule};
```

**Step 4: Update `crates/tarjanize-viz/src/lib.rs`**

The import `use crate::schedule::{TargetGraph, compute_schedule};` on line 29 should still work via the re-export. No change needed.

**Step 5: Move the tests**

The tests in the old `schedule.rs` (lines 348-683) should move to `crates/tarjanize-schedule/src/schedule.rs` since they test the scheduling logic. They should already be included in the copy from Step 1.

**Step 6: Run tests**

```bash
cargo nextest run -p tarjanize-schedule && cargo nextest run -p tarjanize-viz
```

Expected: all tests pass in both crates.

**Step 7: Commit**

```bash
jj describe -m "Move TargetGraph and schedule DP to tarjanize-schedule"
jj new
```

---

## Task 4: Move `build_target_graph()` to `tarjanize-schedule`

Move `build_target_graph()`, `collect_frontend_cost()`, and `count_symbols()` from `crates/tarjanize-viz/src/lib.rs` to `crates/tarjanize-schedule`. This function converts `SymbolGraph + CostModel` into `TargetGraph` and is needed by both viz and the split backend.

**Files:**
- Modify: `crates/tarjanize-schedule/src/lib.rs`
- Modify: `crates/tarjanize-schedule/Cargo.toml`
- Modify: `crates/tarjanize-viz/src/lib.rs`

**Step 1: Add `tarjanize-schemas` dependency to `tarjanize-schedule`**

Add to `crates/tarjanize-schedule/Cargo.toml` under `[dependencies]`:

```toml
tarjanize-schemas.workspace = true
```

**Step 2: Move the functions to `crates/tarjanize-schedule/src/lib.rs`**

Move `build_target_graph()`, `collect_frontend_cost()`, and `count_symbols()` from `crates/tarjanize-viz/src/lib.rs` (lines 56-211). In the new location:

1. Make `build_target_graph()` `pub`
2. Keep `collect_frontend_cost()` and `count_symbols()` as private helpers
3. Add the necessary imports (`tarjanize_schemas::{CostModel, Module, SymbolGraph, sum_event_times}`, `crate::schedule::TargetGraph`)

The `crates/tarjanize-schedule/src/lib.rs` should look like:

```rust
//! Build schedule computation: forward/backward DP, critical path, swim lanes.
//!
//! Extracted from `tarjanize-viz` so both the static HTML visualization and the
//! interactive split explorer can share scheduling primitives.

pub mod data;
pub mod schedule;

use tarjanize_schemas::{CostModel, Module, SymbolGraph, sum_event_times};

use crate::schedule::TargetGraph;

/// Builds a target graph from a `SymbolGraph` and optional `CostModel`.
///
/// For each target in the symbol graph:
/// 1. Computes the three regression predictors (attr, meta, other)
/// 2. Predicts cost via `CostModel::predict()`, or falls back to the
///    effective timing (wall-clock if available, per-symbol sum otherwise)
/// 3. Counts symbols recursively through the module tree
///
/// Test targets are augmented with their lib's per-symbol costs when
/// the test target has no wall-clock profiling data (same logic as
/// `tarjanize-cost`'s `build_target_graph`).
pub fn build_target_graph(
    symbol_graph: &SymbolGraph,
    cost_model: Option<&CostModel>,
) -> TargetGraph {
    // ... (copy body from tarjanize-viz/src/lib.rs lines 67-190)
}

/// Recursively sums all symbol `event_times_ms` in a module tree.
fn collect_frontend_cost(module: &Module) -> f64 {
    // ... (copy body)
}

/// Recursively counts symbols in a module tree.
fn count_symbols(module: &Module) -> usize {
    // ... (copy body)
}
```

**Step 3: Update `crates/tarjanize-viz/src/lib.rs`**

Replace the deleted functions with imports from `tarjanize-schedule`:

```rust
use tarjanize_schedule::build_target_graph;
use tarjanize_schedule::schedule::compute_schedule;
```

Remove the old `use crate::schedule::{TargetGraph, compute_schedule};` import. The `run()` function's body stays the same — it calls `build_target_graph()` and `compute_schedule()`.

**Step 4: Run tests**

```bash
cargo nextest run -p tarjanize-schedule && cargo nextest run -p tarjanize-viz
```

Expected: all tests pass.

**Step 5: Run clippy**

```bash
cargo clippy --all-targets
```

Expected: no warnings.

**Step 6: Commit**

```bash
jj describe -m "Move build_target_graph to tarjanize-schedule"
jj new
```

---

## Task 5: Add web server dependencies

Add axum, tokio, tower-http (for serving static files), and open (for launching browser) to the workspace and to the `tarjanize` binary crate.

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/tarjanize/Cargo.toml`

**Step 1: Add workspace dependencies**

Add to `[workspace.dependencies]` in root `Cargo.toml`:

```toml
# Web server
axum = "0.8"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
tower-http = { version = "0.6", features = ["fs"] }
open = "5"
```

**Step 2: Add dependencies to `crates/tarjanize/Cargo.toml`**

Add under `[dependencies]`:

```toml
axum.workspace = true
tokio.workspace = true
tower-http.workspace = true
open.workspace = true
tarjanize-schedule.workspace = true
```

Also add `tarjanize-cost` if not already present (needed for auto-fitting):

```toml
tarjanize-cost.workspace = true
```

**Step 3: Run `cargo check -p tarjanize`**

```bash
cargo check -p tarjanize
```

Expected: compiles (deps just added, not used yet).

**Step 4: Commit**

```bash
jj describe -m "Add axum/tokio/tower-http deps for web server"
jj new
```

---

## Task 6: Write the failing test for auto-fit cost model

The viz backend should automatically fit a cost model from the symbol graph's profiling data. Write a test that verifies `auto_fit_cost_model()` returns a valid `CostModel` when given a symbol graph with wall-clock data, and `None` when data is absent.

**Files:**
- Modify: `crates/tarjanize-schedule/src/lib.rs`

**Step 1: Write the test**

Add to `crates/tarjanize-schedule/src/lib.rs`:

```rust
/// Auto-fits a cost model from a `SymbolGraph`'s profiling data.
///
/// Uses lib targets with wall-clock profiling data (`wall_time_ms > 0`)
/// to fit the MAGSAC++ regression model. Returns `None` if insufficient
/// profiled targets are available (< 4 lib targets with data).
///
/// This replaces the manual `tarjanize cost --output-model` step for
/// interactive use.
pub fn auto_fit_cost_model(symbol_graph: &SymbolGraph) -> Option<CostModel> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_fit_returns_none_for_empty_graph() {
        let sg = SymbolGraph::default();
        assert!(auto_fit_cost_model(&sg).is_none());
    }
}
```

**Step 2: Run the test to verify it fails**

```bash
cargo nextest run -p tarjanize-schedule -E 'test(auto_fit)'
```

Expected: FAIL with `not yet implemented`.

**Step 3: Implement `auto_fit_cost_model()`**

This function delegates to `tarjanize_cost::fit()` and `tarjanize_cost::build_cost_model()`:

```rust
pub fn auto_fit_cost_model(symbol_graph: &SymbolGraph) -> Option<CostModel> {
    let options = tarjanize_cost::CostOptions {
        fit_libs_only: true,
    };
    let result = tarjanize_cost::fit(symbol_graph, options);
    tarjanize_cost::build_cost_model(&result)
}
```

Add `tarjanize-cost` to `crates/tarjanize-schedule/Cargo.toml`:

```toml
tarjanize-cost.workspace = true
```

**Step 4: Run the test to verify it passes**

```bash
cargo nextest run -p tarjanize-schedule -E 'test(auto_fit)'
```

Expected: PASS.

**Step 5: Commit**

```bash
jj describe -m "Add auto_fit_cost_model to tarjanize-schedule"
jj new
```

---

## Task 7: Convert `tarjanize viz` from static HTML to web server

Replace the current `viz` subcommand's file-based HTML generation with a local axum web server that serves the schedule data via JSON API. Keep the existing Gantt chart frontend working by serving it as a static page that fetches data from the API instead of reading embedded `window.DATA`.

**Files:**
- Create: `crates/tarjanize-viz/src/server.rs`
- Modify: `crates/tarjanize-viz/src/lib.rs`
- Modify: `crates/tarjanize-viz/Cargo.toml`
- Modify: `crates/tarjanize/src/main.rs`

**Step 1: Add web server deps to `tarjanize-viz`**

Add to `crates/tarjanize-viz/Cargo.toml`:

```toml
axum.workspace = true
tokio.workspace = true
tower-http.workspace = true
open.workspace = true
tarjanize-cost.workspace = true
tarjanize-schedule.workspace = true
```

**Step 2: Create `crates/tarjanize-viz/src/server.rs`**

This is the core web server module. It holds the `SymbolGraph` in an `Arc<AppState>` and serves API endpoints.

```rust
//! Local web server for the interactive split explorer.
//!
//! Holds the `SymbolGraph` and fitted `CostModel` in memory. Serves the
//! Gantt chart frontend as static HTML and provides JSON API endpoints
//! for schedule data, target drill-down, splits, and export.

use std::sync::Arc;

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::get,
};
use tarjanize_schedule::{
    auto_fit_cost_model, build_target_graph,
    data::ScheduleData,
    schedule::compute_schedule,
};
use tarjanize_schemas::SymbolGraph;

/// Shared application state, wrapped in `Arc` for cheap cloning across
/// axum handlers.
pub struct AppState {
    /// The loaded symbol graph (immutable for now; splits will mutate a
    /// working copy in a later task).
    pub symbol_graph: SymbolGraph,
    /// The schedule data computed on startup.
    pub schedule: ScheduleData,
}

/// Builds the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/api/schedule", get(schedule_handler))
        .with_state(state)
}

/// Serves the main HTML page (Gantt chart).
///
/// For now, serves a minimal page that fetches schedule data from
/// `/api/schedule`. The full interactive UI is built in later tasks.
async fn index_handler() -> Html<String> {
    // Temporary: serve a minimal page that loads schedule data via fetch.
    // This will be replaced with the full dual-panel UI in later tasks.
    Html(include_str!("../templates/app.html").to_string())
}

/// Returns the current schedule data as JSON.
async fn schedule_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ScheduleData> {
    Json(state.schedule.clone())
}
```

Note: `ScheduleData` needs to derive `Clone`. Add that in `tarjanize-schedule/src/data.rs` (add `Clone` to the existing derives for `ScheduleData`, `Summary`, and `TargetData`).

**Step 3: Create minimal `templates/app.html`**

Create `crates/tarjanize-viz/templates/app.html` — a minimal HTML page that fetches `/api/schedule` and renders the existing Gantt chart. This bridges the transition: same PixiJS renderer, but data comes from the API instead of embedded JSON.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>tarjanize — interactive split explorer</title>
  <style>{{ style_css }}</style>
  <script type="importmap">
    { "imports": { "pixi.js": "https://cdn.jsdelivr.net/npm/pixi.js@8.9.2/+esm" } }
  </script>
</head>
<body>
  <div id="sidebar">
    <h2>tarjanize</h2>
    <div id="info">Loading schedule...</div>
  </div>
  <canvas id="canvas"></canvas>
  <div id="tooltip"></div>
  <script type="module">
    // Fetch schedule data from the backend API.
    const resp = await fetch('/api/schedule');
    const DATA = await resp.json();
    window.DATA = DATA;
    // Import and initialize the renderer.
    // For now, reuse the existing bundle (adapted in a later task).
  </script>
</body>
</html>
```

This is a transitional placeholder. The full dual-panel UI is built in later tasks.

**Step 4: Add `run_server()` to `crates/tarjanize-viz/src/lib.rs`**

Add a new public function that starts the web server:

```rust
mod server;

/// Starts the interactive split explorer web server.
///
/// Loads the symbol graph from `input`, auto-fits a cost model, computes
/// the initial schedule, and starts a local HTTP server. Opens the
/// browser to the served page.
pub async fn run_server(mut input: impl Read) -> Result<(), VizError> {
    let mut json = String::new();
    input.read_to_string(&mut json)?;

    let symbol_graph: SymbolGraph =
        serde_json::from_str(&json).map_err(VizError::deserialize)?;

    // Auto-fit cost model from profiling data.
    let cost_model = tarjanize_schedule::auto_fit_cost_model(&symbol_graph);
    let target_graph = tarjanize_schedule::build_target_graph(
        &symbol_graph,
        cost_model.as_ref(),
    );
    let schedule = tarjanize_schedule::schedule::compute_schedule(&target_graph);

    let state = std::sync::Arc::new(server::AppState {
        symbol_graph,
        schedule,
    });

    let app = server::build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| VizError::io(e))?;
    let addr = listener.local_addr().map_err(|e| VizError::io(e))?;
    let url = format!("http://{addr}");

    eprintln!("Listening on {url}");
    let _ = open::that(&url);

    axum::serve(listener, app)
        .await
        .map_err(|e| VizError::io(e))?;

    Ok(())
}
```

**Step 5: Update CLI to use the web server**

Modify `crates/tarjanize/src/main.rs`:

1. Change the `Viz` command to remove the `--model` flag (cost model is now auto-fitted)
2. Change the handler to call `run_server()` via tokio runtime

```rust
Commands::Viz { io } => {
    let rt = tokio::runtime::Runtime::new()?;
    io.run(|r, _w| {
        rt.block_on(async {
            tarjanize_viz::run_server(r)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))
        })
    })
}
```

Keep the old `run()` function intact for now — it can be used as a `--static` export mode later. For now, the default `viz` command starts the web server.

**Step 6: Run `cargo check -p tarjanize`**

```bash
cargo check -p tarjanize
```

Expected: compiles.

**Step 7: Commit**

```bash
jj describe -m "Convert tarjanize viz to local web server with auto-fit"
jj new
```

---

## Task 8: Adapt Gantt chart frontend to fetch from API

Modify the existing PixiJS renderer to work with the web app by fetching schedule data from `/api/schedule` instead of reading embedded `window.DATA`. Create a proper `app.html` template with the same sidebar, canvas, and tooltip structure.

**Files:**
- Create: `crates/tarjanize-viz/templates/app.html`
- Create: `crates/tarjanize-viz/templates/app.js` (thin loader that fetches from API)
- Modify: `crates/tarjanize-viz/build.rs` (bundle app.js too)
- Modify: `crates/tarjanize-viz/src/server.rs` (serve bundled app)

**Step 1: Create `templates/app.js`**

```javascript
// Entry point for the interactive split explorer.
//
// Fetches schedule data from the backend API and initializes the
// PixiJS Gantt chart renderer. In later tasks, this will also
// initialize the left-panel compound DAG.

import { createRenderer } from './renderer.js';

async function main() {
  const resp = await fetch('/api/schedule');
  const data = await resp.json();
  window.DATA = data;

  // The renderer reads from window.DATA — same as the static HTML version.
  createRenderer();
}

main();
```

Note: `renderer.js` currently auto-initializes. We'll need to wrap its initialization in an exported `createRenderer()` function so the app.js can call it after the fetch completes. This requires a small refactor of `renderer.js` to export an init function instead of running on import.

**Step 2: Refactor `renderer.js` to export `createRenderer()`**

Wrap the top-level initialization code in `renderer.js` inside an exported function:

```javascript
export async function createRenderer() {
  // ... existing initialization code ...
}
```

The static `viz.html` template calls `createRenderer()` inline. The new `app.js` calls it after the fetch.

**Step 3: Update `build.rs` to bundle `app.js`**

Add a second esbuild invocation for `app.js`:

```rust
// Bundle app.js for the web server mode.
let app_entry = PathBuf::from(&manifest_dir).join("templates/app.js");
let app_output = PathBuf::from(&out_dir).join("app_bundle.js");

let status = Command::new(&esbuild)
    .arg(app_entry.to_str().expect("non-UTF8 path"))
    .arg("--bundle")
    .arg("--format=esm")
    .arg("--external:pixi.js")
    .arg(format!("--outfile={}", app_output.display()))
    .status()
    .unwrap_or_else(|e| {
        panic!("failed to run esbuild for app.js: {e}");
    });
assert!(status.success(), "esbuild bundling of app.js failed");

println!("cargo:rerun-if-changed=templates/app.js");
```

**Step 4: Create `templates/app.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>tarjanize — interactive split explorer</title>
  <style>/* CSS inlined at compile time */</style>
  <script type="importmap">
    { "imports": { "pixi.js": "https://cdn.jsdelivr.net/npm/pixi.js@8.9.2/+esm" } }
  </script>
</head>
<body>
  <div id="sidebar">
    <h2>tarjanize</h2>
    <div id="summary"></div>
    <div id="search-container">
      <input type="text" id="search" placeholder="Search targets..." />
    </div>
    <div id="info"></div>
  </div>
  <canvas id="canvas"></canvas>
  <div id="tooltip"></div>
  <script type="module">
    // Bundled app.js inlined at compile time
  </script>
</body>
</html>
```

The actual template will use Askama to inline the CSS and JS bundle, similar to the existing `viz.html`.

**Step 5: Update `server.rs` to serve the bundled HTML**

The `index_handler` serves the compiled app HTML with inlined CSS and JS bundle. Use `include_str!` similar to `html.rs`.

**Step 6: Test manually**

```bash
cargo run -p tarjanize -- viz -i /tmp/omicron-sg.json
```

Expected: browser opens, Gantt chart renders from API data.

**Step 7: Commit**

```bash
jj describe -m "Adapt Gantt chart frontend to fetch schedule from API"
jj new
```

---

## Task 9: Add `GET /api/target/{name}/graph` endpoint — SCC condensation

Implement the endpoint that returns the intra-target SCC DAG for a given target. This is the data the frontend needs to render the force-directed compound graph in the left panel.

**Files:**
- Create: `crates/tarjanize-schedule/src/target_graph.rs` (intra-target SCC computation)
- Modify: `crates/tarjanize-schedule/src/lib.rs`
- Modify: `crates/tarjanize-viz/src/server.rs`

**Step 1: Write the failing test for intra-target SCC condensation**

Create `crates/tarjanize-schedule/src/target_graph.rs`. The core function takes a target from the symbol graph and returns:
- List of SCCs (each SCC is a list of symbol paths)
- Edges between SCCs
- Module membership for each symbol
- Cost per SCC

```rust
//! Intra-target SCC condensation for the split explorer.
//!
//! Computes the SCC DAG within a single target, collapsing symbols into
//! strongly connected components. This is the graph the user sees when
//! drilling down into a target for splitting.

use std::collections::HashMap;

use serde::Serialize;

/// An SCC node in the intra-target condensation graph.
#[derive(Debug, Clone, Serialize)]
pub struct SccNode {
    /// Unique index of this SCC within the target.
    pub id: usize,
    /// Symbol paths belonging to this SCC.
    pub symbols: Vec<String>,
    /// Module path for the primary symbol (for clustering).
    pub module_path: String,
    /// Total attributed cost in milliseconds.
    pub cost_ms: f64,
    /// Number of symbols (excluding Use/ExternCrate items).
    pub symbol_count: usize,
}

/// An edge between two SCCs in the intra-target DAG.
#[derive(Debug, Clone, Serialize)]
pub struct SccEdge {
    pub from: usize,
    pub to: usize,
}

/// The intra-target SCC DAG, ready for frontend rendering.
#[derive(Debug, Clone, Serialize)]
pub struct IntraTargetGraph {
    /// SCC nodes, indexed by `SccNode.id`.
    pub nodes: Vec<SccNode>,
    /// Edges between SCCs (direction: dependency -> dependent).
    pub edges: Vec<SccEdge>,
    /// Module paths found in this target, for clustering metadata.
    pub modules: Vec<String>,
}

/// Computes the SCC DAG for a single target in the symbol graph.
///
/// Filters out Use/ExternCrate symbols (they have zero incoming edges
/// and don't constrain splits). Computes SCCs on the remaining symbol
/// dependency graph and returns the condensed DAG with module
/// membership annotations.
pub fn condense_target(
    symbol_graph: &tarjanize_schemas::SymbolGraph,
    target_id: &str,
) -> Option<IntraTargetGraph> {
    todo!()
}
```

Write a test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tarjanize_schemas::*;

    #[test]
    fn condense_target_filters_use_items() {
        // Build a small symbol graph with one Use item and two real
        // symbols. The Use item should be excluded from the SCC DAG.
        // ... (construct a SymbolGraph programmatically) ...
    }

    #[test]
    fn condense_target_finds_sccs() {
        // Two symbols with a cycle should form a single SCC.
        // ... (construct a SymbolGraph with A -> B and B -> A) ...
    }
}
```

**Step 2: Run the test to verify it fails**

```bash
cargo nextest run -p tarjanize-schedule -E 'test(condense_target)'
```

Expected: FAIL with `not yet implemented`.

**Step 3: Implement `condense_target()`**

The implementation:
1. Parse `target_id` as `{package}/{target}` to look up the target in the symbol graph
2. Walk the module tree, collecting all non-Use/non-ExternCrate symbols with their paths and module membership
3. Build a `DiGraph` of symbol dependencies (only intra-target edges)
4. Run `petgraph::algo::condensation()` to compute SCCs
5. Build `SccNode`s with costs and module paths
6. Extract inter-SCC edges
7. Return `IntraTargetGraph`

**Step 4: Run the tests to verify they pass**

```bash
cargo nextest run -p tarjanize-schedule -E 'test(condense_target)'
```

**Step 5: Add the API endpoint in `server.rs`**

```rust
/// Returns the intra-target SCC DAG for the specified target.
async fn target_graph_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<IntraTargetGraph>, StatusCode> {
    tarjanize_schedule::target_graph::condense_target(
        &state.symbol_graph,
        &name,
    )
    .map(Json)
    .ok_or(StatusCode::NOT_FOUND)
}
```

Register the route:

```rust
.route("/api/target/:name/graph", get(target_graph_handler))
```

Note: the target name contains `/` (e.g., `nexus-db-queries/lib`), so URL-encode it or use a different path scheme (e.g., `package` and `target` as separate path params: `/api/target/:package/:target/graph`).

**Step 6: Run clippy and tests**

```bash
cargo clippy --all-targets && cargo nextest run -p tarjanize-schedule
```

**Step 7: Commit**

```bash
jj describe -m "Add intra-target SCC condensation and API endpoint"
jj new
```

---

## Task 10: Frontend — left panel with force-directed compound DAG

Add the left panel to the web app that renders the intra-target SCC DAG as a force-directed graph with module clustering. This is the main interactive view for exploring splits.

**Files:**
- Create: `crates/tarjanize-viz/templates/dag.js` (d3-force compound DAG renderer)
- Modify: `crates/tarjanize-viz/templates/app.js`
- Modify: `crates/tarjanize-viz/templates/app.html`
- Modify: `crates/tarjanize-viz/templates/style.css`

**Step 1: Add d3-force to import map**

Add to the import map in `app.html`:

```json
{
  "imports": {
    "pixi.js": "https://cdn.jsdelivr.net/npm/pixi.js@8.9.2/+esm",
    "d3-force": "https://cdn.jsdelivr.net/npm/d3-force@3/+esm",
    "d3-selection": "https://cdn.jsdelivr.net/npm/d3-selection@3/+esm",
    "d3-zoom": "https://cdn.jsdelivr.net/npm/d3-zoom@3/+esm"
  }
}
```

**Step 2: Update `app.html` with dual-panel layout**

```html
<div id="left-panel">
  <div id="dag-header">
    <span id="dag-target-name">Click a target in the Gantt chart</span>
  </div>
  <svg id="dag-canvas"></svg>
</div>
<div id="right-panel">
  <div id="sidebar">...</div>
  <canvas id="canvas"></canvas>
</div>
```

**Step 3: Create `templates/dag.js`**

The d3-force renderer:

```javascript
// Force-directed compound DAG renderer for the target drill-down.
//
// Renders the intra-target SCC DAG with:
// - SCC nodes sized by attributed cost
// - Module clustering via translucent background regions
// - Edge bundling for readability
// - Pan/zoom via d3-zoom

import { forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide }
  from 'd3-force';
import { select } from 'd3-selection';
import { zoom } from 'd3-zoom';

export function renderDag(container, data) {
  // data: { nodes: SccNode[], edges: SccEdge[], modules: string[] }

  const svg = select(container);
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;

  // Clear previous content
  svg.selectAll('*').remove();

  const g = svg.append('g');

  // Set up zoom
  svg.call(zoom().on('zoom', (event) => {
    g.attr('transform', event.transform);
  }));

  // Module clustering: group nodes by module_path.
  // Draw translucent hulls around each module's nodes.
  // ... (implementation details)

  // Force simulation
  const simulation = forceSimulation(data.nodes)
    .force('link', forceLink(data.edges)
      .id(d => d.id)
      .source(d => d.from)
      .target(d => d.to)
      .distance(80))
    .force('charge', forceManyBody().strength(-200))
    .force('center', forceCenter(width / 2, height / 2))
    .force('collide', forceCollide(d => Math.sqrt(d.cost_ms) + 5));

  // Draw edges
  const link = g.selectAll('.edge')
    .data(data.edges)
    .join('line')
    .attr('class', 'edge')
    .attr('stroke', '#666')
    .attr('stroke-opacity', 0.4);

  // Draw SCC nodes
  const node = g.selectAll('.scc-node')
    .data(data.nodes)
    .join('circle')
    .attr('class', 'scc-node')
    .attr('r', d => Math.max(3, Math.sqrt(d.cost_ms / 10)))
    .attr('fill', d => moduleColor(d.module_path));

  // Tick update
  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    node
      .attr('cx', d => d.x)
      .attr('cy', d => d.y);
  });

  return simulation;
}

// Assign consistent colors to modules.
function moduleColor(modulePath) {
  // Hash-based coloring for stable module colors.
  // ... (implementation)
}
```

**Step 4: Wire up click handler in Gantt chart**

In `app.js`, add a click handler for Gantt chart target bars that fetches the target's SCC DAG and renders it in the left panel:

```javascript
window.addEventListener('target-click', async (event) => {
  const targetName = event.detail.name;
  const encoded = encodeURIComponent(targetName);
  const resp = await fetch(`/api/target/${encoded}/graph`);
  const data = await resp.json();
  renderDag(document.getElementById('dag-canvas'), data);
  document.getElementById('dag-target-name').textContent = targetName;
});
```

**Step 5: Update `build.rs` to bundle `dag.js`**

Add `cargo:rerun-if-changed=templates/dag.js`.

**Step 6: Test manually**

```bash
cargo run -p tarjanize -- viz -i /tmp/omicron-sg.json
```

Click a target in the Gantt chart; verify the left panel shows a force-directed graph.

**Step 7: Commit**

```bash
jj describe -m "Add force-directed compound DAG in left panel"
jj new
```

---

## Task 11: Split state management — backend

Add mutable split state to the backend. The user will create named crates and assign symbols to them. The backend must track the current split state, enforce convexity (downset constraint), recompute the schedule, and serve updated data.

**Files:**
- Create: `crates/tarjanize-schedule/src/split.rs`
- Modify: `crates/tarjanize-schedule/src/lib.rs`
- Modify: `crates/tarjanize-viz/src/server.rs`

**Step 1: Write the failing test for downset computation**

Create `crates/tarjanize-schedule/src/split.rs`:

```rust
//! Split state management and convexity enforcement.
//!
//! Tracks the user's split decisions: which symbols are assigned to which
//! new crate. Enforces the downset constraint (new crates must be
//! downward-closed sets in the SCC DAG) and computes the resulting
//! schedule impact.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// A proposed split: assigns symbols from one target to a new crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitOperation {
    /// The target being split (e.g., "nexus-db-queries/lib").
    pub source_target: String,
    /// Name for the new crate (e.g., "nexus-db-queries-storage").
    pub new_crate_name: String,
    /// Symbol paths the user selected. The backend will expand this to
    /// the full downset (transitive closure of dependencies within the
    /// target).
    pub selected_symbols: Vec<String>,
}

/// Result of applying a split, including the downset expansion.
#[derive(Debug, Clone, Serialize)]
pub struct SplitResult {
    /// All symbols in the new crate (selected + downset expansion).
    pub symbols_in_new_crate: Vec<String>,
    /// Symbols that were auto-included via downset expansion.
    pub auto_included: Vec<String>,
    /// Updated schedule data after applying the split.
    pub schedule: crate::data::ScheduleData,
}

/// Computes the downset (transitive dependency closure) of a set of
/// symbols within a target's SCC DAG.
///
/// Given selected symbol indices, returns the full downward-closed set:
/// all selected symbols plus all symbols they transitively depend on.
pub fn compute_downset(
    // Adjacency list: for each node, its dependencies (edges point to deps)
    deps: &[Vec<usize>],
    selected: &[usize],
) -> HashSet<usize> {
    todo!()
}
```

Write tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downset_of_leaf_is_just_itself() {
        // A -> B -> C, select C (leaf)
        let deps = vec![
            vec![],    // C: no deps
            vec![0],   // B: depends on C
            vec![1],   // A: depends on B
        ];
        let result = compute_downset(&deps, &[0]);
        assert_eq!(result, HashSet::from([0]));
    }

    #[test]
    fn downset_includes_transitive_deps() {
        // A -> B -> C, select A
        let deps = vec![
            vec![],    // C: no deps
            vec![0],   // B: depends on C
            vec![1],   // A: depends on B
        ];
        let result = compute_downset(&deps, &[2]);
        assert_eq!(result, HashSet::from([0, 1, 2]));
    }

    #[test]
    fn downset_of_parallel_nodes() {
        // A -> C, B -> C (diamond), select A and B
        let deps = vec![
            vec![],       // C: no deps
            vec![0],      // A: depends on C
            vec![0],      // B: depends on C
        ];
        let result = compute_downset(&deps, &[1, 2]);
        assert_eq!(result, HashSet::from([0, 1, 2]));
    }
}
```

**Step 2: Run the test to verify it fails**

```bash
cargo nextest run -p tarjanize-schedule -E 'test(downset)'
```

**Step 3: Implement `compute_downset()`**

Simple BFS/DFS over the dependency edges:

```rust
pub fn compute_downset(
    deps: &[Vec<usize>],
    selected: &[usize],
) -> HashSet<usize> {
    let mut result = HashSet::new();
    let mut stack: Vec<usize> = selected.to_vec();

    while let Some(node) = stack.pop() {
        if result.insert(node) {
            for &dep in &deps[node] {
                if !result.contains(&dep) {
                    stack.push(dep);
                }
            }
        }
    }

    result
}
```

**Step 4: Run tests**

```bash
cargo nextest run -p tarjanize-schedule -E 'test(downset)'
```

Expected: PASS.

**Step 5: Commit**

```bash
jj describe -m "Add downset computation for convexity enforcement"
jj new
```

---

## Task 12: Apply splits to the target graph

Implement the function that takes the current split state and produces an updated `TargetGraph`. When a target is split, replace it with N+1 nodes (N new crates + residual), recompute dependencies, predict costs for each fragment.

**Files:**
- Modify: `crates/tarjanize-schedule/src/split.rs`
- Modify: `crates/tarjanize-schedule/src/lib.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn apply_split_produces_two_targets() {
    // Build a SymbolGraph with a single target containing two independent
    // symbols. Split one into a new crate. Verify the resulting
    // TargetGraph has the original target count + 1.
    // ...
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `apply_splits()`**

This function takes a `SymbolGraph`, a `CostModel`, and a list of `SplitOperation`s, and returns a new `TargetGraph`:

1. For each split operation, identify which symbols move to the new crate (using downset expansion)
2. Compute the new crate's attr cost (sum of per-symbol attributed costs)
3. Distribute meta/other proportionally by attr share
4. Predict cost via the cost model
5. Compute dependencies: the new crate depends on external targets that its symbols depend on (resolved from symbol deps), plus any sibling split-crates
6. The residual target depends on external targets that its remaining symbols depend on, plus the new crate (if any remaining symbol depends on a moved symbol)
7. Build and return the modified `TargetGraph`

**Step 4: Run tests**

**Step 5: Commit**

```bash
jj describe -m "Add split application to produce modified TargetGraph"
jj new
```

---

## Task 13: Add `POST /api/split` and `POST /api/unsplit` endpoints

Wire the split logic into the web server. The backend holds mutable split state in an `RwLock` and recomputes the schedule on each split operation.

**Files:**
- Modify: `crates/tarjanize-viz/src/server.rs`

**Step 1: Add mutable state to `AppState`**

```rust
use std::sync::RwLock;

pub struct AppState {
    pub symbol_graph: SymbolGraph,
    pub cost_model: Option<CostModel>,
    /// Current split operations, guarded by RwLock for concurrent reads.
    pub splits: RwLock<Vec<SplitOperation>>,
    /// Current schedule (recomputed after each split).
    pub schedule: RwLock<ScheduleData>,
}
```

**Step 2: Implement `POST /api/split`**

```rust
async fn split_handler(
    State(state): State<Arc<AppState>>,
    Json(op): Json<SplitOperation>,
) -> Json<SplitResult> {
    // 1. Add the split operation to state.splits
    // 2. Recompute the target graph with all current splits
    // 3. Recompute the schedule
    // 4. Update state.schedule
    // 5. Return the result including the new schedule
}
```

**Step 3: Implement `POST /api/unsplit`**

```rust
/// Request to undo a split by crate name.
#[derive(Deserialize)]
struct UnsplitRequest {
    crate_name: String,
}

async fn unsplit_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnsplitRequest>,
) -> Json<ScheduleData> {
    // 1. Remove the split operation for the given crate name
    // 2. Recompute target graph and schedule
    // 3. Return updated schedule
}
```

**Step 4: Register routes**

```rust
.route("/api/split", post(split_handler))
.route("/api/unsplit", post(unsplit_handler))
```

**Step 5: Test manually**

Use `curl` to POST a split and verify the schedule changes:

```bash
curl -X POST http://localhost:PORT/api/split \
  -H 'Content-Type: application/json' \
  -d '{"source_target":"pkg/lib","new_crate_name":"pkg-core","selected_symbols":["[pkg/lib]::core::Foo"]}'
```

**Step 6: Commit**

```bash
jj describe -m "Add POST /api/split and /api/unsplit endpoints"
jj new
```

---

## Task 14: Frontend — split interaction UI

Add the crate creation and symbol assignment UI to the left panel. The user clicks "New Crate", names it, then clicks symbols/modules to assign them. Assigned symbols change color. The Gantt chart updates.

**Files:**
- Modify: `crates/tarjanize-viz/templates/dag.js`
- Modify: `crates/tarjanize-viz/templates/app.js`
- Modify: `crates/tarjanize-viz/templates/app.html`
- Modify: `crates/tarjanize-viz/templates/style.css`

**Step 1: Add "New Crate" button and name input to the left panel header**

```html
<div id="split-controls">
  <button id="new-crate-btn">New Crate</button>
  <input type="text" id="crate-name-input" placeholder="crate name" hidden />
  <div id="active-crates"></div>
</div>
```

**Step 2: Implement click-to-assign in `dag.js`**

When a new crate is being created:
- Clicking an SCC node sends a `POST /api/split` with the selected symbols
- The response includes `auto_included` symbols (downset expansion) — highlight those differently
- All assigned nodes change to the crate's color
- The Gantt chart re-renders with the updated schedule from the response

**Step 3: Show active crates list**

Display the list of created crates with an "undo" button for each (calls `POST /api/unsplit`).

**Step 4: Update Gantt chart on split**

When the split response arrives, update `window.DATA` with the new schedule and re-render the Gantt chart.

**Step 5: Test manually**

```bash
cargo run -p tarjanize -- viz -i /tmp/omicron-sg.json
```

1. Click a target bar in the Gantt chart
2. Click "New Crate", type a name
3. Click symbols in the DAG — they should change color, Gantt should update
4. Click "undo" on the crate — should revert

**Step 6: Commit**

```bash
jj describe -m "Add split interaction UI with crate creation and assignment"
jj new
```

---

## Task 15: Add heat map endpoint — internal critical path and per-SCC slack

Implement `GET /api/target/{name}/heatmap` that returns per-SCC slack and improvement potential within a target.

**Files:**
- Create: `crates/tarjanize-schedule/src/heatmap.rs`
- Modify: `crates/tarjanize-schedule/src/lib.rs`
- Modify: `crates/tarjanize-viz/src/server.rs`

**Step 1: Write the failing test**

```rust
//! Heat map computation for the intra-target SCC DAG.
//!
//! Runs forward/backward DP on the intra-target SCC DAG (analogous to
//! the target-level schedule) and computes per-SCC slack. Zero-slack
//! SCCs are on the internal critical path; high-slack SCCs are cool.

use serde::Serialize;

/// Heat map data for a single SCC.
#[derive(Debug, Clone, Serialize)]
pub struct SccHeat {
    /// SCC index.
    pub id: usize,
    /// Slack within the intra-target schedule (0 = on critical path).
    pub slack_ms: f64,
    /// Whether this SCC is on the internal critical path.
    pub on_critical_path: bool,
    /// Estimated improvement in ms if a downset split is made at this SCC.
    /// Only computed for zero-slack SCCs; `None` for others.
    pub improvement_ms: Option<f64>,
}

/// Computes heat map data for a target's intra-target SCC DAG.
pub fn compute_heatmap(
    intra_graph: &crate::target_graph::IntraTargetGraph,
) -> Vec<SccHeat> {
    todo!()
}
```

Test:

```rust
#[test]
fn heatmap_chain_all_on_critical_path() {
    // Chain of 3 SCCs: all should have zero slack.
}

#[test]
fn heatmap_parallel_has_slack() {
    // Fork: A -> B, A -> C where cost(B) > cost(C).
    // C should have nonzero slack.
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `compute_heatmap()`**

1. Run forward/backward DP on the `IntraTargetGraph` (reuse the same algorithm from `schedule.rs`, just on SCC costs instead of target costs)
2. For each zero-slack SCC, tentatively compute the downset and evaluate the split improvement by constructing a modified target graph and running the global schedule DP

**Step 4: Run tests**

**Step 5: Add the API endpoint**

```rust
.route("/api/target/:package/:target/heatmap", get(heatmap_handler))
```

**Step 6: Commit**

```bash
jj describe -m "Add heat map: internal critical path and per-SCC slack"
jj new
```

---

## Task 16: Frontend — heat map visualization

Apply heat map coloring to the force-directed DAG. Color SCC nodes by slack (red = critical, blue = high slack). Show improvement potential as tooltips or annotations on critical SCCs.

**Files:**
- Modify: `crates/tarjanize-viz/templates/dag.js`
- Modify: `crates/tarjanize-viz/templates/app.js`

**Step 1: Fetch heat map data alongside graph data**

In `app.js`, when a target is clicked, fetch both `/api/target/{name}/graph` and `/api/target/{name}/heatmap` in parallel.

**Step 2: Apply heat coloring in `dag.js`**

```javascript
// Color nodes by slack.
function slackColor(slack, maxSlack) {
  if (maxSlack === 0) return '#ff4444'; // all critical
  const t = Math.min(slack / maxSlack, 1);
  // Interpolate red -> blue
  const r = Math.round(255 * (1 - t));
  const b = Math.round(255 * t);
  return `rgb(${r}, 0, ${b})`;
}
```

**Step 3: Show improvement annotations**

For SCCs with `improvement_ms`, display a small label: "+X ms saved".

**Step 4: Test manually**

Verify that clicking a target shows the heat-colored DAG, with critical SCCs in red.

**Step 5: Commit**

```bash
jj describe -m "Add heat map coloring to DAG visualization"
jj new
```

---

## Task 17: Add `GET /api/export` endpoint — save modified SymbolGraph

Implement the export endpoint that writes the current split state as a modified `SymbolGraph` JSON.

**Files:**
- Create: `crates/tarjanize-schedule/src/export.rs`
- Modify: `crates/tarjanize-schedule/src/lib.rs`
- Modify: `crates/tarjanize-viz/src/server.rs`

**Step 1: Write the test for symbol graph export**

```rust
#[test]
fn export_with_no_splits_returns_original() {
    // Exporting without any splits should return the original SymbolGraph.
}

#[test]
fn export_with_split_adds_new_target() {
    // After one split, the exported graph should have an additional target.
}
```

**Step 2: Implement `export_symbol_graph()`**

This function takes the original `SymbolGraph` and the list of `SplitOperation`s and produces a new `SymbolGraph`:

1. For each split operation:
   - Create a new package/target for the new crate
   - Move the assigned symbols (with their module structure) from the source target to the new target
   - Update dependencies: the new target's dependencies come from its symbols' external deps; the residual target gains a dependency on the new target
   - Compute cost predictors for the new target and update the residual
2. Return the modified `SymbolGraph`

**Step 3: Run tests**

**Step 4: Add the API endpoint**

```rust
.route("/api/export", get(export_handler))

async fn export_handler(
    State(state): State<Arc<AppState>>,
) -> Json<SymbolGraph> {
    let splits = state.splits.read().unwrap();
    let exported = tarjanize_schedule::export::export_symbol_graph(
        &state.symbol_graph,
        &splits,
    );
    Json(exported)
}
```

**Step 5: Add "Save" button to the frontend**

Add a "Save" button that calls `GET /api/export` and triggers a file download of the JSON.

**Step 6: Commit**

```bash
jj describe -m "Add export endpoint and save button"
jj new
```

---

## Task 18: Preserve static HTML export

Keep the existing static HTML generation as a `--static` flag for sharing. When `--static` is passed, generate a self-contained HTML file (the old behavior). Without it, start the web server.

**Files:**
- Modify: `crates/tarjanize/src/main.rs`
- Modify: `crates/tarjanize-viz/src/lib.rs`

**Step 1: Add `--static` flag to the Viz command**

```rust
Viz {
    #[command(flatten)]
    io: IoArgs,

    /// Generate a static HTML file instead of starting the web server.
    /// Useful for sharing visualizations without running a server.
    #[arg(long, name = "static")]
    static_html: bool,

    /// Path to the fitted cost model (only for --static mode).
    #[arg(long, value_name = "PATH", requires = "static")]
    model: Option<PathBuf>,
},
```

**Step 2: Dispatch based on flag**

```rust
Commands::Viz { io, static_html, model } => {
    if static_html {
        let cost_model = model
            .map(|p| tarjanize_schemas::load_cost_model(&p))
            .transpose()?;
        io.run(|r, w| {
            tarjanize_viz::run(r, cost_model.as_ref(), w)
                .map_err(|e| anyhow::anyhow!("{e}"))
        })
    } else {
        let rt = tokio::runtime::Runtime::new()?;
        io.run(|r, _w| {
            rt.block_on(async {
                tarjanize_viz::run_server(r)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
        })
    }
}
```

**Step 3: Test both modes**

```bash
# Web server (default)
cargo run -p tarjanize -- viz -i /tmp/omicron-sg.json

# Static HTML
cargo run -p tarjanize -- viz --static -i /tmp/omicron-sg.json -o /tmp/test.html
```

**Step 4: Commit**

```bash
jj describe -m "Preserve static HTML export as --static flag"
jj new
```

---

## Task 19: Polish and integration testing

Final polish: error handling, edge cases, logging, and integration tests.

**Files:**
- Various

**Step 1: Add integration test for the full pipeline**

```rust
// In crates/tarjanize-schedule/tests/integration.rs
// Build a SymbolGraph programmatically, auto-fit cost model,
// apply a split, verify schedule improvement.
```

**Step 2: Run clippy**

```bash
cargo clippy --all-targets
```

Fix any warnings.

**Step 3: Run all tests with coverage**

```bash
cargo llvm-cov nextest
```

Verify >= 90% coverage on new modules.

**Step 4: Run doc tests**

```bash
cargo test --doc
```

**Step 5: Commit**

```bash
jj describe -m "Add integration tests and polish"
jj new
```

---

## Summary of new/modified crates

| Crate | Status | Purpose |
|---|---|---|
| `tarjanize-schedule` | **New** | Schedule DP, target graph, SCC condensation, split logic, heatmap, export |
| `tarjanize-viz` | **Modified** | Web server (axum), dual-panel frontend, re-exports from schedule |
| `tarjanize` | **Modified** | CLI: viz starts web server by default, `--static` for old behavior |
| `tarjanize-condense` | Unchanged (remove later) | Kept for now; removed once split explorer is validated |

## Dependency graph

```
tarjanize (binary)
  ├── tarjanize-viz
  │     ├── tarjanize-schedule
  │     │     ├── tarjanize-schemas
  │     │     └── tarjanize-cost
  │     ├── tarjanize-schemas
  │     ├── axum, tokio, tower-http
  │     └── askama (static HTML mode)
  └── tarjanize-cost (for --static mode cost model loading)
```
