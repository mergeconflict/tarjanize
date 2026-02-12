//! Interactive build schedule visualization web server.
//!
//! Starts a local HTTP server with an interactive split explorer for
//! analyzing build schedules from a `SymbolGraph`.
//!
//! ## Pipeline
//!
//! ```text
//! SymbolGraph + CostModel -> TargetGraph -> ScheduleData -> JSON API
//! ```
//!
//! The `SymbolGraph` provides the dependency structure and per-symbol costs.
//! The `CostModel` predicts compilation wall time from three predictors
//! (attributed, metadata, other). When no `CostModel` is available, falls
//! back to per-symbol cost sums.

pub mod data;
mod error;
pub mod server;

use std::io::Read;
use std::sync::{Arc, RwLock};

pub use error::VizError;
use tarjanize_schemas::SymbolGraph;

/// Starts the interactive split explorer web server.
///
/// Loads the symbol graph from `input`, auto-fits a cost model from
/// profiling data, computes the initial schedule, and starts a local
/// HTTP server on an ephemeral port. Opens the default browser to the
/// served page automatically.
///
/// The server runs until terminated (Ctrl+C).
pub async fn run_server(mut input: impl Read) -> Result<(), VizError> {
    let mut json = String::new();
    input.read_to_string(&mut json)?;

    let symbol_graph: SymbolGraph =
        serde_json::from_str(&json).map_err(VizError::deserialize)?;

    // Auto-fit cost model from profiling data. Returns `None` if there
    // are fewer than 4 lib targets with wall-clock data, in which case
    // we fall back to per-symbol cost sums.
    let cost_model = tarjanize_schedule::auto_fit_cost_model(&symbol_graph);
    let target_graph = tarjanize_schedule::build_target_graph(
        &symbol_graph,
        cost_model.as_ref(),
    );
    let schedule =
        tarjanize_schedule::schedule::compute_schedule(&target_graph);

    let state = Arc::new(server::AppState {
        symbol_graph: RwLock::new(symbol_graph),
        base_target_graph: target_graph,
        cost_model,
        schedule: RwLock::new(schedule),
    });
    let app = server::build_router(state);

    // Bind to an ephemeral port on localhost. The OS assigns a free port,
    // which we then print so the user knows where to connect.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(VizError::io)?;
    let addr = listener.local_addr().map_err(VizError::io)?;
    let url = format!("http://{addr}");

    eprintln!("Listening on {url}");
    let _ = open::that(&url);

    axum::serve(listener, app).await.map_err(VizError::io)?;

    Ok(())
}
