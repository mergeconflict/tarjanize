//! Local web server for the interactive split explorer.
//!
//! Holds the computed `ScheduleData` in memory and serves it via JSON API.
//! The index page is a plain HTML file that loads CSS, the renderer bundle,
//! and the sidebar bundle as static assets. Schedule data is fetched from
//! `/api/schedule` at load time. The shatter endpoint lets users explore
//! horizon-grouped splits without mutating server state permanently.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::response::{Html, Json};
use axum::routing::{get, post};
use serde_with::{DurationMilliSecondsWithFrac, serde_as};
use tarjanize_schedule::data::ScheduleData;
use tarjanize_schedule::recommend::collect_symbol_external_targets;
use tarjanize_schedule::schedule::TargetGraph;
use tarjanize_schemas::{CostModel, SymbolGraph, TargetId};

/// Bundled JS produced by esbuild during `build.rs`. Contains renderer.ts
/// with logic.ts inlined, pixi.js kept as an external import.
const BUNDLE_JS: &str = include_str!(concat!(env!("OUT_DIR"), "/bundle.js"));

/// Raw CSS, included at compile time. A single file with no imports, so no
/// bundling needed.
const STYLE_CSS: &str = include_str!("../templates/style.css");

/// Sidebar JS bundle produced by esbuild during `build.rs`. Contains
/// sidebar.ts with tree.ts inlined, IIFE format to avoid variable
/// collisions with the ESM renderer bundle.
const SIDEBAR_JS: &str = include_str!(concat!(env!("OUT_DIR"), "/sidebar.js"));

/// The index HTML page, served as-is. Contains no template syntax -- all
/// assets are loaded via `<link>` and `<script>` tags pointing to the
/// `/static/` routes.
const INDEX_HTML: &str = include_str!("../templates/app.html");

/// Shared application state, wrapped in `Arc` for cheap cloning across
/// axum handlers.
///
/// Immutable data (`symbol_graph`, `base_target_graph`, `cost_model`) is
/// set once at startup. The `schedule` is guarded by `RwLock` for
/// concurrent read access from handlers.
///
/// Why: keeps handlers lightweight while sharing the precomputed schedule.
pub struct AppState {
    /// The original symbol graph, retained for intra-target SCC
    /// condensation when the user drills into a specific target.
    /// Wrapped in `RwLock` because `shatter_handler` updates it with
    /// group targets so that `/api/tree` can resolve shattered names.
    ///
    /// Why: tree rendering needs a mutable view that reflects shatters.
    pub symbol_graph: RwLock<SymbolGraph>,
    /// The base target graph (unsplit), built once at startup from the
    /// symbol graph and cost model.
    ///
    /// Why: shatter computations reuse the baseline graph to avoid drift.
    pub base_target_graph: TargetGraph,
    /// The auto-fitted cost model (if available). Stored so we don't
    /// need to re-fit when recomputing schedules.
    ///
    /// Why: re-fitting is expensive and must stay consistent during a session.
    pub cost_model: Option<CostModel>,
    /// Current schedule data, served by the schedule handler for the
    /// Gantt chart.
    ///
    /// Why: the UI polls this endpoint to render the timeline.
    pub schedule: RwLock<ScheduleData>,
}

// Manual Debug impl because `TargetGraph` derives Debug but we want to
// avoid printing the full graph contents in logs. Shows field presence
// rather than full data. Uses `finish_non_exhaustive()` because we
// intentionally omit the `schedule` field (it's large).
//
// Why: logs should remain readable and avoid dumping large graphs.
impl std::fmt::Debug for AppState {
    /// Formats a compact debug view that avoids dumping full graphs.
    ///
    /// Why: debug output should stay readable in logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field(
                "symbol_graph_packages",
                &self
                    .symbol_graph
                    .read()
                    .expect("symbol_graph lock poisoned")
                    .packages
                    .len(),
            )
            .field(
                "base_target_graph_targets",
                &self.base_target_graph.names.len(),
            )
            .field("cost_model", &self.cost_model.is_some())
            .finish_non_exhaustive()
    }
}

/// Request body for the shatter endpoint.
///
/// Identifies the target to shatter into its full SCC condensation
/// graph. Each SCC becomes an independent target with raw attr cost
/// (no per-target overhead), showing the theoretical maximum parallelism.
///
/// Why: the endpoint only needs a target id to recompute the schedule.
#[derive(Debug, serde::Deserialize)]
struct ShatterRequest {
    /// Target identifier in `{package}/{target}` format.
    ///
    /// Why: this matches the `TargetGraph` naming scheme.
    target_id: String,
}

/// Response for the tree endpoint.
///
/// Contains the full target structure and a map of per-symbol cost
/// breakdowns (attributed vs. shared meta/other costs).
///
/// Why: the sidebar needs both the tree structure and cost detail.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct TreeResponse {
    target: tarjanize_schemas::Target,
    /// Map from symbol path to cost breakdown.
    /// Key format matches `collect_symbol_external_targets` with empty prefix:
    /// e.g. `mod::SymbolName`.
    ///
    /// Why: the frontend uses paths as stable lookup keys for costs.
    symbol_costs: HashMap<String, SymbolCostBreakdown>,
}

/// Cost breakdown for a single symbol.
///
/// Why: the UI displays attr vs shared metadata/other contributions.
#[serde_as]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SymbolCostBreakdown {
    /// Direct attribute cost (sum of event times).
    ///
    /// Why: attr is the symbol's own compilation work.
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    attr: Duration,
    /// Share of metadata cost based on dependency ratio.
    ///
    /// Why: metadata work scales with dependency fanout.
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    meta_share: Duration,
    /// Share of other cost based on dependency ratio.
    ///
    /// Why: remaining compiler work needs proportional attribution.
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    other_share: Duration,
}

/// Context for cost distribution across a target's symbols.
///
/// Why: groups shared parameters so tree traversal stays readable.
struct CostContext<'a> {
    /// Total metadata cost for the target.
    meta: Duration,
    /// Total non-metadata cost for the target.
    other: Duration,
    /// Total number of target-level dependencies (for ratio scaling).
    total_deps: usize,
    /// Per-symbol external dependency lists (path -> deps).
    ext_deps: &'a HashMap<String, Vec<String>>,
    /// Target-level dependency set, used to bound metadata scaling.
    tgt_deps: &'a std::collections::HashSet<String>,
}

/// Sums event-time values into a `Duration`.
///
/// Why: we keep `Duration` as the internal representation while the
/// schema serializes milliseconds for JSON consumers.
fn sum_event_times_ms(map: &HashMap<String, Duration>) -> Duration {
    map.values()
        .copied()
        .fold(Duration::ZERO, |acc, next| acc + next)
}

/// Computes a proportional share of a total `Duration`.
///
/// Why: shared costs are distributed by dependency count without
/// introducing floating-point rounding errors in the attribution path.
fn proportional_share(total: Duration, part: usize, whole: usize) -> Duration {
    if part == 0 || whole == 0 {
        return Duration::ZERO;
    }

    let total_nanos = total.as_nanos();
    let share_nanos = total_nanos.saturating_mul(u128::from(part as u64))
        / u128::from(whole as u64);
    let share_nanos_u64 = u64::try_from(share_nanos)
        .expect("duration share fits in u64 nanoseconds");

    Duration::from_nanos(share_nanos_u64)
}

/// Builds a tree response with per-symbol cost breakdowns for a target.
///
/// Why: keeps handler logic small while centralizing cost attribution.
fn build_tree_response(tgt: &tarjanize_schemas::Target) -> TreeResponse {
    let meta = tgt
        .timings
        .event_times_ms
        .iter()
        .filter(|(k, _)| k.starts_with("metadata_decode_"))
        .map(|(_, v)| *v)
        .fold(Duration::ZERO, |acc, next| acc + next);
    let other = tgt
        .timings
        .event_times_ms
        .iter()
        .filter(|(k, _)| !k.starts_with("metadata_decode_"))
        .map(|(_, v)| *v)
        .fold(Duration::ZERO, |acc, next| acc + next);

    // Dependency counts are bounded by workspace size.
    // Why: metadata/other costs must be apportioned across symbols.
    let total_deps = tgt.dependencies.len();
    let symbol_ext_deps = collect_symbol_external_targets(&tgt.root, "", "");

    let ctx = CostContext {
        meta,
        other,
        total_deps,
        ext_deps: &symbol_ext_deps,
        tgt_deps: &tgt.dependencies,
    };
    let mut symbol_costs = HashMap::new();
    populate_costs(
        &tgt.root,
        &tarjanize_schemas::ModulePath::root(),
        &ctx,
        &mut symbol_costs,
    );

    TreeResponse {
        target: tgt.clone(),
        symbol_costs,
    }
}

/// Walks a module tree and records per-symbol cost breakdowns.
///
/// Recurses through submodules so the sidebar can resolve fully-qualified
/// symbol paths against the cost map.
///
/// Why: costs are computed top-down alongside dependency ratios.
fn populate_costs(
    module: &tarjanize_schemas::Module,
    path: &tarjanize_schemas::ModulePath,
    ctx: &CostContext<'_>,
    costs: &mut HashMap<String, SymbolCostBreakdown>,
) {
    for (name, sym) in &module.symbols {
        let full_path = if path.is_root() {
            name.clone()
        } else {
            format!("{}::{name}", path.as_str())
        };

        let attr = sum_event_times_ms(&sym.event_times_ms);
        let mut meta_share = Duration::ZERO;
        let mut other_share = Duration::ZERO;

        // Only allocate shared costs when dependency ratios are defined.
        if ctx.total_deps > 0
            && let Some(deps) = ctx.ext_deps.get(&full_path)
        {
            let count =
                deps.iter().filter(|d| ctx.tgt_deps.contains(*d)).count();
            meta_share = proportional_share(ctx.meta, count, ctx.total_deps);
            other_share = proportional_share(ctx.other, count, ctx.total_deps);
        }

        costs.insert(
            full_path,
            SymbolCostBreakdown {
                attr,
                meta_share,
                other_share,
            },
        );
    }

    for (sub_name, sub) in &module.submodules {
        populate_costs(sub, &path.child(sub_name), ctx, costs);
    }
}

/// Builds the axum router with all API routes.
///
/// Routes:
/// - `GET /` -- interactive Gantt chart (fetches data from the API)
/// - `GET /static/style.css` -- CSS stylesheet
/// - `GET /static/bundle.js` -- Gantt renderer (ESM)
/// - `GET /static/sidebar.js` -- Sidebar event wiring (IIFE)
/// - `GET /api/schedule` -- full schedule data as JSON
/// - `GET /api/tree/:package/*target` -- module/symbol tree for a target
/// - `POST /api/shatter` -- shatter a target into its SCCs (transient)
/// - `GET /api/export` -- export the `SymbolGraph` as JSON
///
/// Why: centralizes route wiring so tests can validate the API surface.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/static/style.css", get(css_handler))
        .route("/static/bundle.js", get(bundle_js_handler))
        .route("/static/sidebar.js", get(sidebar_js_handler))
        .route("/api/schedule", get(schedule_handler))
        .route("/api/tree/{package}/{*target}", get(tree_handler))
        .route("/api/shatter", post(shatter_handler))
        .route("/api/export", get(export_handler))
        .with_state(state)
}

/// Serves the interactive split explorer page as plain HTML.
///
/// Why: keeps the frontend static and self-contained for local use.
async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// Serves the CSS stylesheet with the correct content type.
///
/// Why: explicit content types avoid browser MIME sniffing issues.
async fn css_handler() -> ([(&'static str, &'static str); 1], &'static str) {
    ([("content-type", "text/css; charset=utf-8")], STYLE_CSS)
}

/// Serves the Gantt renderer JS bundle (ESM) with the correct content type.
///
/// Why: the renderer is an ESM bundle and must be delivered with JS MIME type.
async fn bundle_js_handler() -> ([(&'static str, &'static str); 1], &'static str)
{
    (
        [("content-type", "application/javascript; charset=utf-8")],
        BUNDLE_JS,
    )
}

/// Serves the sidebar JS bundle (IIFE) with the correct content type.
///
/// Why: the sidebar bundle is loaded separately from the renderer.
async fn sidebar_js_handler()
-> ([(&'static str, &'static str); 1], &'static str) {
    (
        [("content-type", "application/javascript; charset=utf-8")],
        SIDEBAR_JS,
    )
}

/// Returns the current schedule data as JSON.
///
/// The frontend fetches this on load to render the Gantt chart. The
/// response includes summary statistics, per-target data, and the
/// critical path.
///
/// Why: the UI polls this for a consistent schedule snapshot.
async fn schedule_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ScheduleData> {
    let schedule = state.schedule.read().expect("schedule lock poisoned");
    Json(schedule.clone())
}

/// Returns the full `Target` struct and per-symbol cost breakdowns.
///
/// The frontend uses this to render the module/symbol tree in the
/// sidebar. Returns `TreeResponse` containing the `Target` and a map
/// of distributed costs (meta/other shares based on dependencies).
///
/// Returns 404 if the package or target doesn't exist.
///
/// Why: the sidebar needs both structure and cost attribution.
async fn tree_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((package, target)): axum::extract::Path<(
        String,
        String,
    )>,
) -> Result<Json<TreeResponse>, axum::http::StatusCode> {
    let sg = state
        .symbol_graph
        .read()
        .expect("symbol_graph lock poisoned");
    let pkg = sg
        .packages
        .get(&package)
        .ok_or(axum::http::StatusCode::NOT_FOUND)?;
    let tgt = pkg
        .targets
        .get(&target)
        .ok_or(axum::http::StatusCode::NOT_FOUND)?;
    Ok(Json(build_tree_response(tgt)))
}

/// Shatters a target by horizon, persisting the result.
///
/// Groups the target's SCCs by effective horizon and replaces it with
/// one target per horizon group. The new schedule is written back to
/// the server state, so subsequent shatters build on top of previous
/// ones. The frontend should update its baseline after a successful
/// shatter.
///
/// Returns 404 if the target doesn't exist or has no symbols.
///
/// Why: enables iterative what-if splitting without restarting the server.
async fn shatter_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShatterRequest>,
) -> Result<Json<ScheduleData>, axum::http::StatusCode> {
    let (new_schedule, new_graph) = {
        let sg = state
            .symbol_graph
            .read()
            .expect("symbol_graph lock poisoned");
        let schedule = state.schedule.read().expect("schedule lock poisoned");
        let target_id = TargetId::parse(&req.target_id)
            .ok_or(axum::http::StatusCode::BAD_REQUEST)?;
        tarjanize_schedule::recommend::shatter_target(
            &sg,
            &target_id,
            &schedule,
            state.cost_model.as_ref(),
        )
        .ok_or(axum::http::StatusCode::NOT_FOUND)?
    };

    // Persist both the shattered schedule and the updated symbol graph
    // so subsequent operations (including /api/tree) resolve group names.
    // Why: UI endpoints must observe the same shatter state.
    let mut schedule = state.schedule.write().expect("schedule lock poisoned");
    *schedule = new_schedule.clone();
    let mut sg = state
        .symbol_graph
        .write()
        .expect("symbol_graph lock poisoned");
    *sg = new_graph;

    Ok(Json(new_schedule))
}

/// Exports the current `SymbolGraph` as JSON.
///
/// Returns the full `SymbolGraph` so the frontend can download it as a
/// file that can be used as input to subsequent pipeline stages.
///
/// Why: export is the handoff between visualization and CLI stages.
async fn export_handler(
    State(state): State<Arc<AppState>>,
) -> Json<SymbolGraph> {
    let sg = state
        .symbol_graph
        .read()
        .expect("symbol_graph lock poisoned");
    Json(sg.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: building the router must not panic.
    ///
    /// Axum validates route patterns eagerly on `Router::route()`. If we
    /// accidentally use the pre-0.8 `:param` syntax instead of `{param}`,
    /// the panic happens here rather than silently passing all unit tests
    /// and blowing up at runtime.
    ///
    /// Why: route validation is the first failure point for bad patterns.
    #[test]
    #[expect(clippy::default_trait_access)]
    fn build_router_does_not_panic() {
        let empty_schedule = tarjanize_schedule::data::ScheduleData {
            targets: Vec::new(),
            critical_path: Vec::new(),
            summary: tarjanize_schedule::data::Summary {
                critical_path: Duration::ZERO,
                total_cost: Duration::ZERO,
                parallelism_ratio: 0.0,
                target_count: 0,
                symbol_count: 0,
                lane_count: 0,
            },
        };
        let state = Arc::new(AppState {
            symbol_graph: RwLock::new(SymbolGraph::default()),
            base_target_graph: tarjanize_schedule::schedule::TargetGraph {
                names: Default::default(),
                costs: Vec::new(),
                symbol_counts: Vec::new(),
                graph: Default::default(),
            },
            cost_model: None,
            schedule: RwLock::new(empty_schedule),
        });
        let _router = build_router(state);
    }

    // =================================================================
    // API contract tests -- exercises each endpoint via tower::oneshot
    // =================================================================

    use std::collections::HashSet;

    use http::Request;
    use http_body_util::BodyExt;
    use tarjanize_schemas::*;
    use tower::ServiceExt;

    /// Build a test `AppState` with a small but valid `SymbolGraph`.
    ///
    /// Contains a single package "test-pkg" with target "lib" holding
    /// two symbols (Foo and Bar) where Foo depends on Bar. This gives
    /// us two SCCs and a non-trivial schedule.
    ///
    /// Why: keeps API tests small while exercising non-trivial graph paths.
    fn test_state() -> Arc<AppState> {
        let prefix = "[test-pkg/lib]::";

        let bar = Symbol {
            file: "lib.rs".into(),
            event_times_ms: HashMap::from([(
                "typeck".into(),
                Duration::from_millis(5),
            )]),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Struct".into(),
                visibility: Visibility::Public,
            },
        };
        let foo = Symbol {
            file: "lib.rs".into(),
            event_times_ms: HashMap::from([(
                "typeck".into(),
                Duration::from_millis(10),
            )]),
            dependencies: HashSet::from([format!("{prefix}Bar")]),
            kind: SymbolKind::ModuleDef {
                kind: "Function".into(),
                visibility: Visibility::Public,
            },
        };

        let root = Module {
            symbols: HashMap::from([("Foo".into(), foo), ("Bar".into(), bar)]),
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root,
        };
        let mut targets = HashMap::new();
        targets.insert("lib".into(), target);
        let mut packages = HashMap::new();
        packages.insert("test-pkg".into(), Package { targets });
        let sg = SymbolGraph { packages };

        let tg = tarjanize_schedule::build_target_graph(&sg, None);
        let schedule = tarjanize_schedule::schedule::compute_schedule(&tg);

        Arc::new(AppState {
            symbol_graph: RwLock::new(sg),
            base_target_graph: tg,
            cost_model: None,
            schedule: RwLock::new(schedule),
        })
    }

    /// GET /api/schedule should return 200 with valid JSON containing
    /// the schedule data (targets, `critical_path`, summary).
    ///
    /// Why: the frontend assumes this endpoint is a stable schedule source.
    #[tokio::test]
    async fn api_schedule_returns_valid_json() {
        let state = test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/schedule")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let schedule: tarjanize_schedule::data::ScheduleData =
            serde_json::from_slice(&body)
                .expect("response should be valid ScheduleData JSON");

        assert_eq!(schedule.targets.len(), 1, "should have 1 target");
        assert!(
            !schedule.summary.critical_path.is_zero(),
            "critical path should be positive"
        );
    }

    /// GET /api/export should return a valid `SymbolGraph` JSON.
    ///
    /// Why: export is the handoff point for downstream CLI stages.
    #[tokio::test]
    async fn api_export_returns_valid_symbol_graph() {
        let state = test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/export")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let sg: SymbolGraph = serde_json::from_slice(&body)
            .expect("response should be valid SymbolGraph JSON");

        assert!(
            sg.packages.contains_key("test-pkg"),
            "exported graph should contain test-pkg"
        );
    }

    /// POST /api/shatter should replace a target with its SCC DAG,
    /// returning a schedule with more targets than the original.
    ///
    /// Why: shatter must update the schedule for interactive exploration.
    #[tokio::test]
    async fn api_shatter_returns_expanded_schedule() {
        let state = test_state();

        let shatter_body = serde_json::json!({
            "target_id": "test-pkg/lib",
        });

        let app = build_router(Arc::clone(&state));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/shatter")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&shatter_body).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let schedule: tarjanize_schedule::data::ScheduleData =
            serde_json::from_slice(&body)
                .expect("response should be valid ScheduleData JSON");

        // The test fixture has 2 symbols (Foo depends on Bar) but no
        // external deps, so both SCCs share horizon 0.0 and merge into
        // one group. The original target is replaced by that single group.
        assert!(
            !schedule.targets.is_empty(),
            "shattered schedule should have targets"
        );
        // The shattered target should be named with a group suffix.
        assert!(
            schedule.targets.iter().any(|t| t.name.contains("::group_")),
            "shattered schedule should contain group targets"
        );
    }

    /// POST /api/shatter should return 404 for a nonexistent target.
    ///
    /// Why: the UI treats missing targets as a recoverable error.
    #[tokio::test]
    async fn api_shatter_404_for_missing_target() {
        let state = test_state();

        let shatter_body = serde_json::json!({
            "target_id": "nonexistent/lib",
        });

        let app = build_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/shatter")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&shatter_body).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 404);
    }

    /// GET /api/tree/{package}/{target} should return the Target struct
    /// from the `SymbolGraph`, including the full module/symbol tree.
    ///
    /// Why: the sidebar depends on accurate tree data for symbol display.
    #[tokio::test]
    async fn api_tree_returns_target() {
        let state = test_state();
        let app = build_router(Arc::clone(&state));

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/tree/test-pkg/lib")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let response: TreeResponse = serde_json::from_slice(&body)
            .expect("response should be valid TreeResponse JSON");

        let target = response.target;

        // The test fixture has 2 symbols (Foo and Bar) in the root module.
        assert_eq!(target.root.symbols.len(), 2);
        assert!(
            target.root.symbols.contains_key("Foo"),
            "root module should contain Foo"
        );
        assert!(
            target.root.symbols.contains_key("Bar"),
            "root module should contain Bar"
        );
    }

    /// GET /api/tree/{package}/{target} should return 404 for a
    /// nonexistent package or target.
    ///
    /// Why: missing targets should not crash the UI.
    #[tokio::test]
    async fn api_tree_404_for_missing_target() {
        let state = test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/tree/nonexistent/lib")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 404);
    }

    /// GET /api/tree/{package}/{*target} should handle target names
    /// containing slashes (e.g., `bin/main`). The catch-all `{*target}`
    /// parameter captures the entire remaining path.
    ///
    /// Why: binary targets often include slashes and must resolve correctly.
    #[tokio::test]
    async fn api_tree_handles_slash_in_target_name() {
        // Build a state with a "bin/main" target to exercise the
        // catch-all path parameter.
        let sym = Symbol {
            file: "main.rs".into(),
            event_times_ms: HashMap::from([(
                "typeck".into(),
                Duration::from_millis(3),
            )]),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".into(),
                visibility: Visibility::Public,
            },
        };
        let root = Module {
            symbols: HashMap::from([("main".into(), sym)]),
            submodules: HashMap::new(),
        };
        let target = Target {
            timings: TargetTimings::default(),
            dependencies: HashSet::new(),
            root,
        };
        let mut targets = HashMap::new();
        targets.insert("bin/main".into(), target);
        let mut packages = HashMap::new();
        packages.insert("my-pkg".into(), Package { targets });
        let sg = SymbolGraph { packages };

        let tg = tarjanize_schedule::build_target_graph(&sg, None);
        let schedule = tarjanize_schedule::schedule::compute_schedule(&tg);

        let state = Arc::new(AppState {
            symbol_graph: RwLock::new(sg),
            base_target_graph: tg,
            cost_model: None,
            schedule: RwLock::new(schedule),
        });

        let app = build_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/tree/my-pkg/bin/main")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let response: TreeResponse = serde_json::from_slice(&body)
            .expect("response should be valid TreeResponse JSON");

        let tgt = response.target;

        assert!(
            tgt.root.symbols.contains_key("main"),
            "should return the bin/main target's symbols"
        );
    }

    /// After shattering a target, `GET /api/tree/{pkg}/{target}::group_N`
    /// should return 200 with the symbols that belong to that group.
    ///
    /// This is the core regression test: before the fix, shattered group
    /// names were not present in the `SymbolGraph`, causing 404 errors
    /// when the frontend tried to display the module tree for a group.
    ///
    /// Why: group targets are the primary UI drill-down after shatter.
    #[tokio::test]
    async fn api_tree_returns_group_after_shatter() {
        let state = test_state();

        // Step 1: Shatter test-pkg/lib into horizon groups.
        let shatter_body = serde_json::json!({
            "target_id": "test-pkg/lib",
        });
        let app = build_router(Arc::clone(&state));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/shatter")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&shatter_body).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "shatter should succeed");

        // Step 2: Find the group target names from the shattered schedule.
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let schedule: tarjanize_schedule::data::ScheduleData =
            serde_json::from_slice(&body).unwrap();
        let group_targets: Vec<&str> = schedule
            .targets
            .iter()
            .filter(|t| t.name.contains("::group_"))
            .map(|t| t.name.as_str())
            .collect();
        assert!(
            !group_targets.is_empty(),
            "shattered schedule must contain group targets"
        );

        // Step 3: Request /api/tree for the first group target.
        // Group names are like "test-pkg/lib::group_0". The URL
        // splits as package="test-pkg", target="lib::group_0".
        let group_name = group_targets[0];
        let tree_uri = format!("/api/tree/{group_name}");
        let app = build_router(Arc::clone(&state));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&tree_uri)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            200,
            "GET /api/tree for group target '{group_name}' should return 200"
        );

        // Step 4: Verify the response is a valid Target with symbols.
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let response: TreeResponse = serde_json::from_slice(&body)
            .expect("response should be valid TreeResponse JSON");

        let target = response.target;

        // The group should have at least one symbol (the test fixture
        // has Foo and Bar; they share horizon 0 so both land in group_0).
        let sym_count = target.root.count_symbols();
        assert!(sym_count > 0, "group target should contain symbols, got 0");
    }
}
