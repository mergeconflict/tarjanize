//! Local web server for the interactive split explorer.
//!
//! Holds the computed `ScheduleData` in memory and serves it via JSON API.
//! The index page is a plain HTML file that loads CSS, the renderer bundle,
//! and the sidebar bundle as static assets. Schedule data is fetched from
//! `/api/schedule` at load time. The shatter endpoint lets users explore
//! horizon-grouped splits without mutating server state permanently.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::Router;
use axum::extract::State;
use axum::response::{Html, Json};
use axum::routing::{get, post};
use tarjanize_schedule::data::ScheduleData;
use tarjanize_schedule::recommend::collect_symbol_external_targets;
use tarjanize_schedule::schedule::TargetGraph;
use tarjanize_schemas::{CostModel, SymbolGraph};

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
pub struct AppState {
    /// The original symbol graph, retained for intra-target SCC
    /// condensation when the user drills into a specific target.
    /// Wrapped in `RwLock` because `shatter_handler` updates it with
    /// group targets so that `/api/tree` can resolve shattered names.
    pub symbol_graph: RwLock<SymbolGraph>,
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
#[derive(Debug, serde::Deserialize)]
struct ShatterRequest {
    /// Target identifier in `{package}/{target}` format.
    target_id: String,
}

/// Response for the tree endpoint.
///
/// Contains the full target structure and a map of per-symbol cost
/// breakdowns (attributed vs. shared meta/other costs).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct TreeResponse {
    target: tarjanize_schemas::Target,
    /// Map from symbol path to cost breakdown.
    /// Key format matches `collect_symbol_external_targets` with empty prefix:
    /// e.g. `mod::SymbolName`.
    symbol_costs: HashMap<String, SymbolCostBreakdown>,
}

/// Cost breakdown for a single symbol.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SymbolCostBreakdown {
    /// Direct attribute cost (sum of event times).
    attr: f64,
    /// Share of metadata cost based on dependency ratio.
    meta_share: f64,
    /// Share of other cost based on dependency ratio.
    other_share: f64,
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
async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// Serves the CSS stylesheet with the correct content type.
async fn css_handler() -> ([(&'static str, &'static str); 1], &'static str) {
    ([("content-type", "text/css; charset=utf-8")], STYLE_CSS)
}

/// Serves the Gantt renderer JS bundle (ESM) with the correct content type.
async fn bundle_js_handler() -> ([(&'static str, &'static str); 1], &'static str)
{
    (
        [("content-type", "application/javascript; charset=utf-8")],
        BUNDLE_JS,
    )
}

/// Serves the sidebar JS bundle (IIFE) with the correct content type.
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
#[expect(
    clippy::too_many_lines,
    reason = "handler keeps parsing, cost aggregation, and response assembly in one place"
)]
async fn tree_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((package, target)): axum::extract::Path<(
        String,
        String,
    )>,
) -> Result<Json<TreeResponse>, axum::http::StatusCode> {
    /// Walks a module tree and records per-symbol cost breakdowns.
    ///
    /// Recurses through submodules so the sidebar can resolve fully-qualified
    /// symbol paths against the cost map.
    #[expect(
        clippy::too_many_arguments,
        reason = "local helper keeps tree traversal close to handler data flow"
    )]
    fn populate_costs(
        module: &tarjanize_schemas::Module,
        path: &str,
        meta: f64,
        other: f64,
        total_deps: u32,
        ext_deps: &HashMap<String, Vec<String>>,
        tgt_deps: &std::collections::HashSet<String>,
        costs: &mut HashMap<String, SymbolCostBreakdown>,
    ) {
        for (name, sym) in &module.symbols {
            let full_path = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}::{name}")
            };

            let attr: f64 = sym.event_times_ms.values().sum();

            let mut meta_share = 0.0;
            let mut other_share = 0.0;

            // Only allocate shared costs when dependency ratios are defined.
            if total_deps > 0
                && let Some(deps) = ext_deps.get(&full_path)
            {
                let count_u32 = u32::try_from(
                    deps.iter().filter(|d| tgt_deps.contains(*d)).count(),
                )
                .expect("dependency count fits u32");
                let ratio = f64::from(count_u32) / f64::from(total_deps);
                meta_share = meta * ratio;
                other_share = other * ratio;
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

        for (name, sub) in &module.submodules {
            let child_path = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}::{name}")
            };
            populate_costs(
                sub,
                &child_path,
                meta,
                other,
                total_deps,
                ext_deps,
                tgt_deps,
                costs,
            );
        }
    }

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

    // Calculate total meta/other costs for the target.
    let meta: f64 = tgt
        .timings
        .event_times_ms
        .iter()
        .filter(|(k, _)| k.starts_with("metadata_decode_"))
        .map(|(_, v)| v)
        .sum();
    let other: f64 = tgt
        .timings
        .event_times_ms
        .iter()
        .filter(|(k, _)| !k.starts_with("metadata_decode_"))
        .map(|(_, v)| v)
        .sum();

    // Calculate dependency ratios and distribute costs.
    // Dependency counts are bounded by workspace size; convert through u32
    // to avoid clippy's precision-loss lint while keeping ratios stable.
    let total_deps = u32::try_from(tgt.dependencies.len())
        .expect("dependency count fits u32");
    let symbol_ext_deps = collect_symbol_external_targets(&tgt.root, "", "");

    let mut symbol_costs = HashMap::new();

    populate_costs(
        &tgt.root,
        "",
        meta,
        other,
        total_deps,
        &symbol_ext_deps,
        &tgt.dependencies,
        &mut symbol_costs,
    );

    Ok(Json(TreeResponse {
        target: tgt.clone(),
        symbol_costs,
    }))
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
    let (new_schedule, new_graph) = {
        let sg = state
            .symbol_graph
            .read()
            .expect("symbol_graph lock poisoned");
        let schedule = state.schedule.read().expect("schedule lock poisoned");
        tarjanize_schedule::recommend::shatter_target(
            &sg,
            &req.target_id,
            &schedule,
            state.cost_model.as_ref(),
        )
        .ok_or(axum::http::StatusCode::NOT_FOUND)?
    };

    // Persist both the shattered schedule and the updated symbol graph
    // so subsequent operations (including /api/tree) resolve group names.
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
            symbol_graph: RwLock::new(sg),
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

    /// GET /api/tree/{package}/{target} should return the Target struct
    /// from the `SymbolGraph`, including the full module/symbol tree.
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
    #[tokio::test]
    async fn api_tree_handles_slash_in_target_name() {
        // Build a state with a "bin/main" target to exercise the
        // catch-all path parameter.
        let sym = Symbol {
            file: "main.rs".into(),
            event_times_ms: HashMap::from([("typeck".into(), 3.0)]),
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
        let sym_count = count_symbols_in_module(&target.root);
        assert!(sym_count > 0, "group target should contain symbols, got 0");
    }

    /// Recursively counts symbols in a module tree.
    fn count_symbols_in_module(module: &tarjanize_schemas::Module) -> usize {
        module.symbols.len()
            + module
                .submodules
                .values()
                .map(count_symbols_in_module)
                .sum::<usize>()
    }
}
