//! Schedule data types serialized to JSON for the HTML template.
//!
//! These structs are embedded as JSON in the generated HTML file and
//! consumed by the `PixiJS` renderer. Field names are chosen for JS
//! ergonomics (short, camelCase via serde rename).

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_with::{DurationMilliSecondsWithFrac, serde_as};
use ts_rs::TS;

/// Top-level schedule data embedded in the HTML visualization.
///
/// Why: the viz frontend consumes a single JSON payload, so we bundle all
/// schedule state into one serializable struct.
#[derive(Clone, Debug, Deserialize, Serialize, TS)]
#[ts(export)]
pub struct ScheduleData {
    /// Aggregate statistics about the build schedule.
    pub summary: Summary,
    /// Per-target scheduling details, indexed by position.
    pub targets: Vec<TargetData>,
    /// Indices into `targets` forming the global critical path.
    pub critical_path: Vec<usize>,
}

/// Summary statistics for the build schedule.
///
/// Why: these aggregate metrics drive the sidebar and allow quick comparisons
/// without recomputing from per-target data.
#[serde_as]
#[derive(Clone, Debug, Default, Deserialize, Serialize, TS)]
#[ts(export)]
pub struct Summary {
    /// Critical path length (minimum build time).
    #[serde(rename = "critical_path_ms")]
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    #[ts(as = "f64")]
    pub critical_path: Duration,
    /// Sum of all target costs (theoretical sequential build time).
    #[serde(rename = "total_cost_ms")]
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    #[ts(as = "f64")]
    pub total_cost: Duration,
    /// `total_cost / critical_path` — effective parallelism.
    pub parallelism_ratio: f64,
    /// Number of compilation targets in the graph.
    pub target_count: usize,
    /// Total symbol count across all targets.
    pub symbol_count: usize,
    /// Number of swim lanes (peak parallelism).
    pub lane_count: usize,
}

/// Scheduling details for a single compilation target.
///
/// Why: Gantt rendering and interactions need per-target timing, deps, and
/// critical-path metadata.
#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize, TS)]
#[ts(export)]
pub struct TargetData {
    /// Target identifier in `{package}/{target}` format.
    pub name: String,
    /// Earliest start time.
    #[serde(rename = "start")]
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    #[ts(as = "f64")]
    pub start: Duration,
    /// Completion time (`start + cost`).
    #[serde(rename = "finish")]
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    #[ts(as = "f64")]
    pub finish: Duration,
    /// Predicted compilation cost.
    #[serde(rename = "cost")]
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    #[ts(as = "f64")]
    pub cost: Duration,
    /// How much this target can slip without affecting critical path.
    #[serde(rename = "slack")]
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    #[ts(as = "f64")]
    pub slack: Duration,
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
