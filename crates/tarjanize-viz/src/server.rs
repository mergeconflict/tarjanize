//! Local web server for the interactive split explorer.
//!
//! Holds the computed `ScheduleData` in memory and serves it via JSON API.
//! The index page renders the full `PixiJS` Gantt chart, fetching schedule
//! data from `/api/schedule` at load time. Split recommendations and preview
//! endpoints let users explore split candidates without mutating server state.

use std::sync::{Arc, RwLock};

use askama::Template;
use axum::Router;
use axum::extract::State;
use axum::response::{Html, Json};
use axum::routing::{get, post};
use tarjanize_schedule::data::ScheduleData;
use tarjanize_schedule::schedule::TargetGraph;
use tarjanize_schedule::split::SplitOperation;
use tarjanize_schemas::{CostModel, SymbolGraph};

/// Bundled JS produced by esbuild during `build.rs`. Contains renderer.ts
/// with logic.ts inlined, pixi.js kept as an external import. Shared with
/// the static HTML mode (same bundle).
const BUNDLE_JS: &str = include_str!(concat!(env!("OUT_DIR"), "/bundle.js"));

/// Raw CSS, included at compile time. A single file with no imports, so no
/// bundling needed. Shared with the static HTML mode (same stylesheet).
const STYLE_CSS: &str = include_str!("../templates/style.css");

/// Askama template for the interactive web app.
///
/// Unlike the static `VizTemplate` (which embeds schedule data inline),
/// this template fetches data from `/api/schedule` at load time via
/// top-level `await`. The bundle JS runs only after the fetch completes,
/// so `window.DATA` is populated before the renderer reads it.
#[derive(Template)]
#[template(path = "app.html")]
struct AppTemplate {
    /// Inlined CSS (dark theme, sidebar layout, tooltip styling).
    style_css: String,
    /// Inlined esbuild bundle (renderer.ts + logic.ts, pixi.js external).
    bundle_js: String,
}

/// Shared application state, wrapped in `Arc` for cheap cloning across
/// axum handlers.
///
/// Immutable data (`symbol_graph`, `base_target_graph`, `cost_model`) is
/// set once at startup. The `schedule` is guarded by `RwLock` for
/// concurrent read access from handlers.
pub struct AppState {
    /// The original symbol graph, retained for intra-target SCC
    /// condensation when the user drills into a specific target.
    pub symbol_graph: SymbolGraph,
    /// The base target graph (unsplit), built once at startup from the
    /// symbol graph and cost model.
    pub base_target_graph: TargetGraph,
    /// The auto-fitted cost model (if available). Stored so we don't
    /// need to re-fit when recomputing schedules.
    pub cost_model: Option<CostModel>,
    /// Current schedule data, served by the schedule handler for the
    /// Gantt chart.
    pub schedule: RwLock<ScheduleData>,
}

// Manual Debug impl because `TargetGraph` derives Debug but we want to
// avoid printing the full graph contents in logs. Shows field presence
// rather than full data. Uses `finish_non_exhaustive()` because we
// intentionally omit the `schedule` field (it's large).
impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("symbol_graph_packages", &self.symbol_graph.packages.len())
            .field(
                "base_target_graph_targets",
                &self.base_target_graph.names.len(),
            )
            .field("cost_model", &self.cost_model.is_some())
            .finish_non_exhaustive()
    }
}

/// Request body for the preview-split endpoint.
///
/// Specifies a target and a horizon threshold from the split
/// recommendations. The handler finds the matching candidate, applies
/// the split transiently (without persisting), and returns the modified
/// schedule for Gantt chart preview.
#[derive(Debug, serde::Deserialize)]
struct PreviewSplitRequest {
    /// Target identifier in `{package}/{target}` format.
    target_id: String,
    /// The horizon threshold (ms) identifying the candidate to preview.
    threshold_ms: f64,
}

/// Request body for the shatter endpoint.
///
/// Identifies the target to shatter into its full SCC condensation
/// graph. Each SCC becomes an independent target with raw attr cost
/// (no per-target overhead), showing the theoretical maximum parallelism.
#[derive(Debug, serde::Deserialize)]
struct ShatterRequest {
    /// Target identifier in `{package}/{target}` format.
    target_id: String,
}

/// Builds the axum router with all API routes.
///
/// Routes:
/// - `GET /` -- interactive Gantt chart (fetches data from the API)
/// - `GET /api/schedule` -- full schedule data as JSON
/// - `GET /api/splits/:package/:target` -- ranked split recommendations
/// - `POST /api/preview-split` -- preview a split candidate (transient)
/// - `POST /api/shatter` -- shatter a target into its SCCs (transient)
/// - `GET /api/export` -- export the `SymbolGraph` as JSON
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/api/schedule", get(schedule_handler))
        .route("/api/splits/{package}/{target}", get(splits_handler))
        .route("/api/preview-split", post(preview_split_handler))
        .route("/api/shatter", post(shatter_handler))
        .route("/api/export", get(export_handler))
        .with_state(state)
}

/// Serves the interactive split explorer page.
///
/// Renders the `app.html` Askama template with inlined CSS and JS bundle.
/// The template fetches schedule data from `/api/schedule` at load time,
/// then initializes the `PixiJS` Gantt chart renderer.
async fn index_handler() -> Html<String> {
    let template = AppTemplate {
        style_css: STYLE_CSS.to_owned(),
        bundle_js: BUNDLE_JS.to_owned(),
    };
    // Template rendering is infallible for static string inputs. Panic on
    // failure because a broken template is a programming error, not a
    // runtime condition the server can recover from.
    let rendered = template.render().expect("app.html template render failed");
    Html(rendered)
}

/// Returns the current schedule data as JSON.
///
/// The frontend fetches this on load to render the Gantt chart. The
/// response includes summary statistics, per-target data, and the
/// critical path.
async fn schedule_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ScheduleData> {
    let schedule = state.schedule.read().expect("schedule lock poisoned");
    Json(schedule.clone())
}

/// Returns ranked split recommendations for a target.
///
/// Computes effective horizons across the target's intra-target SCC DAG
/// and evaluates threshold cuts to find beneficial split points. Returns
/// a `SplitRecommendation` with candidates sorted by global improvement.
/// Always returns 200 (empty candidates list for non-existent targets).
async fn splits_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((package, target)): axum::extract::Path<(
        String,
        String,
    )>,
) -> Json<tarjanize_schedule::recommend::SplitRecommendation> {
    let target_id = format!("{package}/{target}");
    let schedule = state.schedule.read().expect("schedule lock poisoned");
    Json(
        tarjanize_schedule::recommend::compute_split_recommendations(
            &state.symbol_graph,
            &target_id,
            &schedule,
            state.cost_model.as_ref(),
        ),
    )
}

/// Previews a split candidate without persisting it.
///
/// Takes a target and threshold, computes the corresponding split
/// candidate from the recommendations, determines the downset SCC IDs,
/// applies the split transiently, and returns the modified schedule for
/// Gantt chart preview. Does not mutate the server's split state.
///
/// Returns 404 if the target doesn't exist or no candidate matches
/// the given threshold.
async fn preview_split_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PreviewSplitRequest>,
) -> Result<Json<ScheduleData>, axum::http::StatusCode> {
    let schedule = state.schedule.read().expect("schedule lock poisoned");

    // Condense the target into its SCC DAG.
    let intra = tarjanize_schedule::target_graph::condense_target(
        &state.symbol_graph,
        &req.target_id,
    )
    .ok_or(axum::http::StatusCode::NOT_FOUND)?;

    // Compute effective horizons to determine which SCCs fall in the downset.
    let horizons = tarjanize_schedule::recommend::compute_effective_horizons(
        &intra,
        &state.symbol_graph,
        &req.target_id,
        &schedule,
    );

    // Collect SCC IDs where horizon <= threshold (the downset).
    // Convert the f64 ms threshold from the request to Duration for
    // comparison with the Duration-based horizons.
    let threshold =
        std::time::Duration::from_secs_f64(req.threshold_ms / 1000.0);
    let downset_scc_ids: Vec<usize> = intra
        .nodes
        .iter()
        .filter(|node| horizons[node.id] <= threshold)
        .map(|node| node.id)
        .collect();

    if downset_scc_ids.is_empty() {
        return Err(axum::http::StatusCode::NOT_FOUND);
    }

    // Build a synthetic split operation for the downset.
    let preview_op = SplitOperation {
        source_target: req.target_id.clone(),
        new_crate_name: format!("{}::preview", req.target_id),
        selected_sccs: downset_scc_ids,
    };

    // Apply the preview split transiently (no persistent state).
    let (target_graph, _) = tarjanize_schedule::split::apply_splits(
        &state.base_target_graph,
        &state.symbol_graph,
        &[preview_op],
    );

    let new_schedule =
        tarjanize_schedule::schedule::compute_schedule(&target_graph);

    Ok(Json(new_schedule))
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
async fn shatter_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShatterRequest>,
) -> Result<Json<ScheduleData>, axum::http::StatusCode> {
    let new_schedule = {
        let schedule = state.schedule.read().expect("schedule lock poisoned");
        tarjanize_schedule::recommend::shatter_target(
            &state.symbol_graph,
            &req.target_id,
            &schedule,
        )
        .ok_or(axum::http::StatusCode::NOT_FOUND)?
    };

    // Persist the shattered schedule so subsequent operations build on it.
    let mut schedule = state.schedule.write().expect("schedule lock poisoned");
    *schedule = new_schedule.clone();

    Ok(Json(new_schedule))
}

/// Exports the current `SymbolGraph` as JSON.
///
/// Returns the full `SymbolGraph` so the frontend can download it as a
/// file that can be used as input to subsequent pipeline stages.
async fn export_handler(
    State(state): State<Arc<AppState>>,
) -> Json<SymbolGraph> {
    Json(state.symbol_graph.clone())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// Smoke test: building the router must not panic.
    ///
    /// Axum validates route patterns eagerly on `Router::route()`. If we
    /// accidentally use the pre-0.8 `:param` syntax instead of `{param}`,
    /// the panic happens here rather than silently passing all unit tests
    /// and blowing up at runtime.
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
            symbol_graph: SymbolGraph::default(),
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

    use std::collections::{HashMap, HashSet};

    use http::Request;
    use http_body_util::BodyExt;
    use tarjanize_schemas::*;
    use tower::ServiceExt;

    /// Build a test `AppState` with a small but valid `SymbolGraph`.
    ///
    /// Contains a single package "test-pkg" with target "lib" holding
    /// two symbols (Foo and Bar) where Foo depends on Bar. This gives
    /// us two SCCs and a non-trivial schedule.
    fn test_state() -> Arc<AppState> {
        let prefix = "[test-pkg/lib]::";

        let bar = Symbol {
            file: "lib.rs".into(),
            event_times_ms: HashMap::from([("typeck".into(), 5.0)]),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Struct".into(),
                visibility: Visibility::Public,
            },
        };
        let foo = Symbol {
            file: "lib.rs".into(),
            event_times_ms: HashMap::from([("typeck".into(), 10.0)]),
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
            symbol_graph: sg,
            base_target_graph: tg,
            cost_model: None,
            schedule: RwLock::new(schedule),
        })
    }

    /// GET /api/schedule should return 200 with valid JSON containing
    /// the schedule data (targets, `critical_path`, summary).
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

    /// GET /api/splits/{package}/{target} should return split
    /// recommendations with target, candidates, and `current_cost_ms`.
    #[tokio::test]
    async fn api_splits_returns_recommendations() {
        let state = test_state();
        let app = build_router(Arc::clone(&state));

        // Use the known test target "test-pkg/lib".
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/splits/test-pkg/lib")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let rec: serde_json::Value = serde_json::from_slice(&body)
            .expect("response should be valid JSON");

        assert!(rec.get("target").is_some(), "must have target field");
        assert!(
            rec.get("candidates").is_some(),
            "must have candidates field"
        );
        assert!(
            rec.get("current_cost_ms").is_some(),
            "must have current_cost_ms field"
        );
    }

    /// POST /api/preview-split should return a modified schedule
    /// without mutating the server's schedule state.
    #[tokio::test]
    async fn api_preview_split_returns_modified_schedule() {
        let state = test_state();

        // Preview a split at threshold 0.0. Since the test fixture has
        // no external dependencies, all horizons are 0.0 and the
        // downset includes all SCCs at this threshold.
        let preview_body = serde_json::json!({
            "target_id": "test-pkg/lib",
            "threshold_ms": 0.0,
        });

        let app = build_router(Arc::clone(&state));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/preview-split")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&preview_body).unwrap(),
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

        // The preview should produce a schedule (may have 2 targets
        // from the split). The exact count depends on how the split
        // partitions the SCCs.
        assert!(
            !schedule.targets.is_empty(),
            "preview schedule should have targets"
        );
    }

    /// POST /api/preview-split should return 404 for a nonexistent
    /// target.
    #[tokio::test]
    async fn api_preview_split_404_for_missing_target() {
        let state = test_state();

        let preview_body = serde_json::json!({
            "target_id": "nonexistent/lib",
            "threshold_ms": 0.0,
        });

        let app = build_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/preview-split")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&preview_body).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 404);
    }

    /// GET /api/export should return a valid `SymbolGraph` JSON.
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
}
