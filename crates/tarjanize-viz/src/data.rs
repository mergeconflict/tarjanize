//! Schedule data types serialized to JSON for the HTML template.
//!
//! These structs are embedded as JSON in the generated HTML file and
//! consumed by the `PixiJS` renderer. Field names are chosen for JS
//! ergonomics (short, camelCase via serde rename).

use serde::Serialize;

/// Top-level schedule data embedded in the HTML visualization.
#[derive(Debug, Serialize)]
pub struct ScheduleData {
    /// Aggregate statistics about the build schedule.
    pub summary: Summary,
    /// Per-target scheduling details, indexed by position.
    pub targets: Vec<TargetData>,
    /// Indices into `targets` forming the global critical path.
    pub critical_path: Vec<usize>,
}

/// Summary statistics for the build schedule.
#[derive(Debug, Serialize)]
pub struct Summary {
    /// Critical path length in milliseconds (minimum build time).
    pub critical_path_ms: f64,
    /// Sum of all target costs (theoretical sequential build time).
    pub total_cost_ms: f64,
    /// `total_cost_ms / critical_path_ms` — effective parallelism.
    pub parallelism_ratio: f64,
    /// Number of compilation targets in the graph.
    pub target_count: usize,
    /// Total symbol count across all targets.
    pub symbol_count: usize,
    /// Number of swim lanes (peak parallelism).
    pub lane_count: usize,
}

/// Scheduling details for a single compilation target.
#[derive(Debug, Serialize)]
pub struct TargetData {
    /// Target identifier in `{package}/{target}` format.
    pub name: String,
    /// Earliest start time in milliseconds.
    pub start: f64,
    /// Completion time (`start + cost`).
    pub finish: f64,
    /// Predicted compilation cost in milliseconds.
    pub cost: f64,
    /// How much this target can slip without affecting critical path.
    pub slack: f64,
    /// Swim lane index for vertical positioning in the Gantt chart.
    pub lane: usize,
    /// Number of symbols in this target.
    pub symbol_count: usize,
    /// Indices of direct dependencies (into the `targets` array).
    pub deps: Vec<usize>,
    /// Indices of direct dependents (into the `targets` array).
    pub dependents: Vec<usize>,
    /// Whether this target lies on the global critical path.
    pub on_critical_path: bool,
    /// Predecessor on the longest path TO this target (for JS path walk).
    pub forward_pred: Option<usize>,
    /// Successor on the longest path FROM this target (for JS path walk).
    pub backward_succ: Option<usize>,
}
