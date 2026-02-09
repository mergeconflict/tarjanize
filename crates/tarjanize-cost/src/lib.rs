//! Cost analysis for symbol graphs.
//!
//! This crate computes build cost metrics for `SymbolGraph` structures,
//! including critical path analysis. The critical path represents the
//! minimum build time achievable with infinite parallelism.
//!
//! ## Cost Model
//!
//! For crate splitting, only **frontend time** matters — it's serial within a
//! crate, determines rmeta readiness, and gates downstream crates. Backend time
//! (LLVM codegen) is parallel via CGUs and doesn't meaningfully affect the
//! critical path.
//!
//! ```text
//! crate_cost = wall_time_ms = Σ sum(symbol.event_times_ms)
//! ```
//!
//! ## Target-Level Analysis
//!
//! The critical path is computed at the **target** level, not package level.
//! Each compilation target (lib, test, bin/cli, etc.) is a separate node in
//! the dependency graph. This is important because:
//!
//! - Dev-dependencies only affect test targets, not lib targets
//! - The lib→test dependency is explicit (tests depend on their lib)
//! - No artificial cycles from dev-dependencies
//!
//! Target identifiers use the format `{package}/{target}`:
//! - `my-package/lib` - library target
//! - `my-package/test` - unit test target
//! - `my-package/bin/cli` - binary target named "cli"
//!
//! ## Algorithm
//!
//! The critical path is computed using dynamic programming in topological order:
//!
//! 1. Build a directed graph of target dependencies
//! 2. Process targets in topological order (dependencies before dependents)
//! 3. For each target:
//!    - `start[t] = max(finish[dep] for dep in deps)`
//!    - `finish[t] = start[t] + wall_time_ms[t]`
//! 4. The critical path length is the maximum `finish[t]` across all targets

use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;

use indexmap::IndexSet;
use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use tarjanize_magsac::{
    LinearModel, fit_magsac, r_squared_no_intercept_inliers,
};
use tarjanize_schemas::{Module, SymbolGraph, TargetTimings, sum_event_times};

/// Serializable cost model for predicting synthetic crate wall times.
///
/// Stores the coefficients from the fitted 3-variable no-intercept model:
/// ```text
/// wall = coeff_attr · attr + coeff_meta · meta + coeff_other · other
/// ```
///
/// For synthetic crates (after condensation), the caller computes:
/// - `attr`: sum of per-symbol event times across the partition
/// - `meta`: max of constituent targets' `metadata_decode_*` event sums
/// - `other`: max of constituent targets' non-metadata event sums
///
/// The max-constituent heuristic works because a synthetic crate inherits
/// at least as much metadata/other overhead as its largest constituent.
///
/// Named fields ensure schema mismatches fail at deserialization time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Coefficient for symbol-attributed cost.
    pub coeff_attr: f64,
    /// Coefficient for metadata decode cost (`metadata_decode_*` events).
    pub coeff_meta: f64,
    /// Coefficient for remaining unattributed cost (non-metadata events).
    pub coeff_other: f64,
    /// R² of the model fit.
    pub r_squared: f64,
    /// MAGSAC++ inlier threshold for classifying outliers.
    pub inlier_threshold: f64,
}

impl CostModel {
    /// Predicts wall time for a synthetic crate.
    ///
    /// The three inputs mirror the original model's predictors:
    /// - `attr`: sum of symbol-attributed event times
    /// - `meta`: metadata decode cost (max of constituent targets)
    /// - `other`: non-metadata unattributed cost (max of constituent targets)
    pub fn predict(&self, attr: f64, meta: f64, other: f64) -> f64 {
        self.coeff_attr * attr
            + self.coeff_meta * meta
            + self.coeff_other * other
    }
}

/// Loads a `CostModel` from a JSON file.
pub fn load_cost_model(path: &Path) -> std::io::Result<CostModel> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Extracts a `CostModel` from a [`CriticalPathResult`].
///
/// Returns `None` if no model was fitted (insufficient profiling data).
/// The `SymbolGraph` is not needed — the coefficients come entirely from
/// the MAGSAC++ regression fitted during critical path computation.
pub fn build_cost_model(result: &CriticalPathResult) -> Option<CostModel> {
    let fit = result.model.as_ref()?;
    let coeffs = &fit.model.coeffs;

    Some(CostModel {
        coeff_attr: *coeffs.first().unwrap_or(&0.0),
        coeff_meta: *coeffs.get(1).unwrap_or(&0.0),
        coeff_other: *coeffs.get(2).unwrap_or(&0.0),
        r_squared: fit.r_squared,
        inlier_threshold: fit.inlier_threshold,
    })
}

/// Details about a single target on the critical path.
#[derive(Debug, Clone)]
pub struct TargetOnPath {
    /// Target identifier in `{package}/{target}` format.
    /// Examples: `my-package/lib`, `my-package/test`, `my-package/bin/cli`.
    pub name: String,

    /// Frontend cost in milliseconds (type checking, borrow checking).
    /// This is the only cost dimension — backend time is parallel via CGUs
    /// and doesn't affect the critical path.
    pub frontend_cost: f64,

    /// When this target can start (max of dependencies' `finish_time`).
    /// With infinite parallelism, this is the earliest possible start.
    pub start_time: f64,

    /// When this target fully completes (`start_time + frontend_cost`).
    pub finish_time: f64,

    /// Direct dependencies of this target (target identifiers).
    pub dependencies: Vec<String>,

    /// Raw wall-clock timings from profiler (zeros if no profiling data).
    /// These are the actual measured compilation times, unmodified.
    pub wall_timings: TargetTimings,

    /// Per-symbol model timings (frontend = sum of symbol frontend costs).
    /// Computed from symbol-level cost estimates, independent of whether
    /// profiling data was available.
    pub symbol_timings: TargetTimings,
}

/// Result of critical path analysis.
#[derive(Debug, Clone)]
pub struct CriticalPathResult {
    /// Minimum build time with infinite parallelism (max finish time).
    /// Only frontend time matters — backend is parallel via CGUs.
    pub critical_path_ms: f64,

    /// Targets on the critical path, from deepest dependency to top-level.
    /// Each entry is a target identifier like `my-package/lib`.
    pub path: Vec<String>,

    /// Detailed information for each target on the critical path.
    pub path_details: Vec<TargetOnPath>,

    /// All targets with their timing details, sorted by `finish_time` descending.
    pub all_targets: Vec<TargetOnPath>,

    /// Sum of all target costs (theoretical sequential build time).
    pub total_cost: f64,

    /// Number of targets in the graph.
    pub target_count: usize,

    /// Number of symbols in the graph.
    pub symbol_count: usize,

    /// Fitted regression model, if enough profiled targets were available.
    /// Reused by both the modeled scheduling pass and the validation table.
    pub model: Option<ModelFit>,

    /// Modeled critical path length (using model-predicted timings).
    /// `None` when no model could be fitted.
    pub modeled_critical_path_ms: Option<f64>,

    /// Target names on the modeled critical path.
    pub modeled_path: Option<Vec<String>>,

    /// Detailed timing info for targets on the modeled critical path.
    pub modeled_path_details: Option<Vec<TargetOnPath>>,

    /// All targets with modeled timing details, sorted by `finish_time`
    /// descending.
    pub modeled_all_targets: Option<Vec<TargetOnPath>>,

    /// Sum of all modeled target costs.
    pub modeled_total_cost: Option<f64>,
}

/// Configuration options for cost reporting.
#[derive(Debug, Clone, Copy, Default)]
pub struct CostOptions {
    /// Fit the regression model using lib targets only.
    pub fit_libs_only: bool,
}

impl CriticalPathResult {
    /// Writes a human-readable report to the given writer.
    ///
    /// The report includes summary statistics and timing tables for all targets.
    pub fn write_report(&self, mut w: impl Write) -> std::io::Result<()> {
        self.write_report_with_options(&mut w, CostOptions::default())
    }

    /// Writes a human-readable report to the given writer with custom options.
    ///
    /// Shows both wall-clock and modeled critical paths when a model is
    /// available. The wall-clock path uses effective timings (profiled
    /// values or per-symbol fallback). The modeled path uses regression-
    /// predicted timings, enabling apples-to-apples comparison before/after
    /// crate condensation.
    pub fn write_report_with_options(
        &self,
        mut w: impl Write,
        _options: CostOptions,
    ) -> std::io::Result<()> {
        // Summary statistics — show both actual and modeled when available.
        writeln!(
            w,
            "Critical path (wall-clock): {}",
            format_duration_ms(self.critical_path_ms)
        )?;
        if let Some(modeled_ms) = self.modeled_critical_path_ms {
            writeln!(
                w,
                "Critical path (modeled):    {}",
                format_duration_ms(modeled_ms)
            )?;
        }
        writeln!(
            w,
            "Total cost (wall-clock):    {}",
            format_duration_ms(self.total_cost)
        )?;
        if let Some(modeled_total) = self.modeled_total_cost {
            writeln!(
                w,
                "Total cost (modeled):       {}",
                format_duration_ms(modeled_total)
            )?;
        }

        let parallelism = if self.critical_path_ms > 0.0 {
            self.total_cost / self.critical_path_ms
        } else {
            1.0
        };
        writeln!(w, "Parallelism ratio:          {parallelism:.2}x")?;
        writeln!(w, "Target count:               {}", self.target_count)?;
        writeln!(w, "Symbol count:               {}", self.symbol_count)?;

        // Table header for timing details.
        let header = format!(
            "{:>10}  {:>10}  {:>10}  {:<40}  Dependencies",
            "Start", "Finish", "Cost", "Target"
        );
        let separator = "-".repeat(header.len() + 20);

        // --- Actual (wall-clock) tables ---
        if !self.path_details.is_empty() {
            writeln!(
                w,
                "\nCritical path — wall-clock ({} targets):\n",
                self.path.len()
            )?;
            writeln!(w, "{header}")?;
            writeln!(w, "{separator}")?;

            for target in &self.path_details {
                writeln!(w, "{target}")?;
            }
        }

        if !self.all_targets.is_empty() {
            writeln!(
                w,
                "\nAll targets by finish time — wall-clock ({} targets):\n",
                self.all_targets.len()
            )?;
            writeln!(w, "{header}")?;
            writeln!(w, "{separator}")?;

            for target in &self.all_targets {
                writeln!(w, "{target}")?;
            }
        }

        // --- Modeled tables (only when model was fitted) ---
        if let Some(ref modeled_path_details) = self.modeled_path_details {
            let modeled_path = self.modeled_path.as_ref().unwrap();
            if !modeled_path_details.is_empty() {
                writeln!(
                    w,
                    "\nCritical path — modeled ({} targets):\n",
                    modeled_path.len()
                )?;
                writeln!(w, "{header}")?;
                writeln!(w, "{separator}")?;

                for target in modeled_path_details {
                    writeln!(w, "{target}")?;
                }
            }
        }

        if let Some(ref modeled_all) = self.modeled_all_targets
            && !modeled_all.is_empty()
        {
            writeln!(
                w,
                "\nAll targets by finish time — modeled ({} targets):\n",
                modeled_all.len()
            )?;
            writeln!(w, "{header}")?;
            writeln!(w, "{separator}")?;

            for target in modeled_all {
                writeln!(w, "{target}")?;
            }
        }

        self.write_validation_table(&mut w)?;

        Ok(())
    }

    /// Writes a model validation table showing actual vs predicted compilation
    /// times.
    ///
    /// Only emitted when at least one target has non-zero wall-clock data
    /// from profiling. Reuses `self.model` (fitted during critical path
    /// computation) to show per-target predictions and percentage errors.
    /// Sorted by absolute error descending so the worst predictions appear
    /// first.
    ///
    /// Falls back to showing just Target + Actual when no model was fitted
    /// (insufficient profiled targets).
    #[expect(
        clippy::too_many_lines,
        reason = "formatting-heavy table output, splitting would scatter layout logic"
    )]
    fn write_validation_table(&self, mut w: impl Write) -> std::io::Result<()> {
        // Collect targets that have wall-clock profiling data.
        let validated: Vec<&TargetOnPath> = self
            .all_targets
            .iter()
            .filter(|t| t.wall_timings.wall_time_ms > 0.0)
            .collect();

        if validated.is_empty() {
            return Ok(());
        }

        // Build regression data per target: [attr, meta, other, wall].
        // Used for predictions even though the model is already fitted.
        let regression_data: Vec<Vec<f64>> = validated
            .iter()
            .map(|t| {
                let attr = t.symbol_timings.wall_time_ms;
                let meta: f64 = t
                    .wall_timings
                    .event_times_ms
                    .iter()
                    .filter(|(k, _)| k.starts_with("metadata_decode_"))
                    .map(|(_, v)| v)
                    .sum();
                let other: f64 = t
                    .wall_timings
                    .event_times_ms
                    .iter()
                    .filter(|(k, _)| !k.starts_with("metadata_decode_"))
                    .map(|(_, v)| v)
                    .sum();
                let wall = t.wall_timings.wall_time_ms;
                vec![attr, meta, other, wall]
            })
            .collect();

        writeln!(
            w,
            "\nModel validation ({} targets with profiling):\n",
            validated.len(),
        )?;

        let mut outlier_count_for_summary = None;

        if let Some(ref fit) = self.model {
            // Build rows with predictions and sort by absolute error.
            // Predictions use the fitted model for all targets. Mark
            // outliers using MAGSAC's inlier threshold — points whose
            // absolute residual exceeds the threshold are outliers that
            // the robust estimator down-weighted during fitting.
            let mut rows: Vec<(&str, f64, f64, f64, bool)> = validated
                .iter()
                .zip(&regression_data)
                .map(|(t, row)| {
                    let actual = t.wall_timings.wall_time_ms;
                    let predicted = fit.model.predict(&row[..row.len() - 1]);
                    let delta_ms = predicted - actual;
                    let is_outlier =
                        (actual - predicted).abs() > fit.inlier_threshold;
                    (t.name.as_str(), actual, predicted, delta_ms, is_outlier)
                })
                .collect();
            // Sort by absolute error descending so the worst predictions
            // appear at the top of the table.
            rows.sort_by(|a, b| {
                b.3.abs()
                    .partial_cmp(&a.3.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let outlier_count = rows.iter().filter(|r| r.4).count();

            // Table header: actual, predicted, delta, outlier marker.
            writeln!(
                w,
                "{:<35} {:>9} {:>9} {:>9}  Out",
                "Target", "Actual", "Predicted", "Delta",
            )?;
            writeln!(w, "{}", "-".repeat(70))?;

            for (name, actual, predicted, delta_ms, is_outlier) in &rows {
                let marker = if *is_outlier { " *" } else { "" };
                writeln!(
                    w,
                    "{:<35} {:>9} {:>9} {:>9} {:>3}",
                    truncate_name(name, 35),
                    format_duration_ms(*actual),
                    format_duration_ms(*predicted),
                    format_duration_ms(*delta_ms),
                    marker,
                )?;
            }

            // Store outlier count for the summary section below.
            outlier_count_for_summary = Some(outlier_count);
        } else {
            // No model — show actuals only.
            writeln!(w, "{:<35} {:>9}", "Target", "Actual")?;
            writeln!(w, "{}", "-".repeat(46))?;

            for target in &validated {
                writeln!(
                    w,
                    "{:<35} {:>9}",
                    truncate_name(&target.name, 35),
                    format_duration_ms(target.wall_timings.wall_time_ms),
                )?;
            }
        }

        // Summary statistics.
        writeln!(w)?;
        writeln!(w, "Targets compared:    {}", validated.len())?;

        if let Some(ref fit) = self.model {
            let coeffs = &fit.model.coeffs;
            // Display the model equation with named predictors.
            // Coefficients: [0]=attr, [1]=meta, [2]=other.
            if coeffs.len() >= 3 {
                writeln!(
                    w,
                    "Model fit:           wall = {:.2} * attr + \
                     {:.2} * meta + {:.2} * other",
                    coeffs[0], coeffs[1], coeffs[2],
                )?;
            } else if coeffs.len() == 2 {
                writeln!(
                    w,
                    "Model fit:           wall = {:.2} * attr + \
                     {:.2} * meta",
                    coeffs[0], coeffs[1],
                )?;
            }
            writeln!(w, "R²:                  {:.4}", fit.r_squared)?;

            if let Some(n) = outlier_count_for_summary {
                writeln!(w, "Outliers excluded:   {n}")?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for TargetOnPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format dependencies: show first few, then count if many.
        let deps_str = if self.dependencies.is_empty() {
            "(none)".to_string()
        } else if self.dependencies.len() <= 3 {
            self.dependencies.join(", ")
        } else {
            format!(
                "{}, ... (+{} more)",
                self.dependencies[..3].join(", "),
                self.dependencies.len() - 3
            )
        };

        write!(
            f,
            "{:>10}  {:>10}  {:>10}  {:<40}  {}",
            format_duration_ms(self.start_time),
            format_duration_ms(self.finish_time),
            format_duration_ms(self.frontend_cost),
            self.name,
            deps_str
        )
    }
}

/// Intermediate representation of the target dependency graph.
///
/// Contains the dependency graph plus three timing views per target:
/// - `effective`: what the scheduler uses (wall-clock if available,
///   per-symbol fallback)
/// - `wall`: raw profiler wall-clock times (zeros if no profiling)
/// - `symbol_model`: per-symbol cost model (frontend = sum of symbol costs)
struct TargetGraph {
    /// Target identifiers in `{package}/{target}` format.
    names: IndexSet<String>,
    /// Timings used by the scheduler (wall-clock when available, else
    /// per-symbol). This is what determines scheduling and critical path.
    effective: Vec<TargetTimings>,
    /// Raw profiler wall-clock times. Zeros when no profiling data exists.
    /// Never augmented — represents exactly what the profiler measured.
    wall: Vec<TargetTimings>,
    /// Per-symbol cost model: frontend = sum of symbol frontend costs.
    /// Always computed from symbols.
    symbol_model: Vec<TargetTimings>,
    /// Total number of symbols across all targets.
    symbol_count: usize,
    /// Dependency graph where edges point from dependency to dependent.
    graph: DiGraph<usize, ()>,
}

/// Result of N-variable no-intercept regression.
///
/// Fits `y = Σ βᵢ·xᵢ` (no intercept). Used to model
/// `wall_time = a·attr + b·meta + c·other` where:
/// - `attr` = sum of symbol-attributed event times
/// - `meta` = sum of `metadata_decode_*` event times
/// - `other` = sum of remaining unattributed event times
#[derive(Debug, Clone)]
pub struct ModelFit {
    /// The fitted linear model (coefficients in predictor order).
    pub model: LinearModel,
    /// Coefficient of determination (R²).
    pub r_squared: f64,
    /// Residual threshold from MAGSAC++ for classifying outliers.
    /// Points with `|y - predicted| > inlier_threshold` are outliers.
    pub inlier_threshold: f64,
}

/// Fits `y = Σ βᵢ·xᵢ` (no intercept) using MAGSAC++ for robustness.
///
/// The no-intercept constraint is physically motivated: zero code plus zero
/// dependencies should produce zero compilation time.
///
/// Each row in `data` is `[x1, x2, ..., xn, y]`. Returns `None` if
/// insufficient data points or if the robust estimator can't produce a
/// stable model.
pub fn model_fit(data: &[Vec<f64>]) -> Option<ModelFit> {
    if data.is_empty() {
        return None;
    }
    let n_vars = data[0].len().checked_sub(1)?;
    if n_vars == 0 || data.len() < n_vars + 1 {
        return None;
    }

    let result = fit_magsac(data)?;
    let r_squared = r_squared_no_intercept_inliers(
        data,
        &result.model,
        result.inlier_threshold,
    );

    Some(ModelFit {
        model: result.model,
        r_squared,
        inlier_threshold: result.inlier_threshold,
    })
}

/// Result of scheduling targets in topological order using DP.
///
/// Contains timing arrays and the reconstructed critical path. Reusable
/// for both wall-clock and model-predicted scheduling passes.
struct ScheduleResult {
    /// Start time for each target (indexed by node index).
    start: Vec<f64>,
    /// Finish time for each target (indexed by node index).
    finish: Vec<f64>,
    /// Minimum build time with infinite parallelism (max finish time).
    critical_path_ms: f64,
    /// Nodes on the critical path, from deepest dependency to top-level.
    path_nodes: Vec<NodeIndex>,
    /// Sum of all target costs (theoretical sequential build time).
    total_cost: f64,
}

/// Schedules targets in topological order using DP.
///
/// For each target t:
///   `start[t] = max(finish[dep] for dep in dependencies)`
///   `finish[t] = start[t] + cost[t]`
///
/// The critical path is the chain of dependencies that determines the
/// latest finish time. This is factored out so we can run it twice:
/// once with wall-clock timings, once with model-predicted timings.
fn schedule(
    costs: &[f64],
    graph: &DiGraph<usize, ()>,
    sorted: &[NodeIndex],
) -> ScheduleResult {
    let n = costs.len();
    let mut start = vec![0.0; n];
    let mut finish = vec![0.0; n];
    let mut predecessor: Vec<Option<NodeIndex>> = vec![None; n];

    for &node in sorted {
        let idx = node.index();

        // Find max finish time among dependencies.
        let mut max_dep_finish = 0.0f64;
        let mut max_dep_node: Option<NodeIndex> = None;

        for dep in graph.neighbors_directed(node, Direction::Incoming) {
            let dep_idx = dep.index();
            if finish[dep_idx] > max_dep_finish {
                max_dep_finish = finish[dep_idx];
                max_dep_node = Some(dep);
            }
        }

        start[idx] = max_dep_finish;
        finish[idx] = max_dep_finish + costs[idx];
        predecessor[idx] = max_dep_node;
    }

    // Critical path ends at the target with latest finish time.
    let (max_node, &critical_path_ms) = finish
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // Reconstruct the critical path by following predecessors.
    let mut path_nodes = Vec::new();
    let mut current = Some(NodeIndex::new(max_node));
    while let Some(node) = current {
        path_nodes.push(node);
        current = predecessor[node.index()];
    }
    path_nodes.reverse();

    let total_cost: f64 = costs.iter().sum();

    ScheduleResult {
        start,
        finish,
        critical_path_ms,
        path_nodes,
        total_cost,
    }
}

/// Computes the critical path of a symbol graph at the target level.
///
/// Only frontend time matters for scheduling — backend time (LLVM codegen)
/// is parallel via CGUs and doesn't affect the critical path.
///
/// Each compilation target (lib, test, bin, etc.) is a separate node. This
/// naturally resolves dev-dependency "cycles" because test targets depend on
/// lib targets, not vice versa.
///
/// Uses default options (no model fitting). For model-predicted scheduling,
/// use [`critical_path_with_options`].
pub fn critical_path(symbol_graph: &SymbolGraph) -> CriticalPathResult {
    critical_path_with_options(symbol_graph, CostOptions::default())
}

/// Computes the critical path with custom options.
///
/// When profiling data is available for enough targets, fits a regression
/// model and runs scheduling twice: once with wall-clock/effective timings
/// (the "actual" critical path) and once with model-predicted timings
/// (the "modeled" critical path). This enables apples-to-apples comparison
/// before/after crate condensation.
#[expect(
    clippy::too_many_lines,
    reason = "core algorithm with model fitting + dual scheduling, \
              splitting would scatter the scheduling logic"
)]
pub fn critical_path_with_options(
    symbol_graph: &SymbolGraph,
    options: CostOptions,
) -> CriticalPathResult {
    let TargetGraph {
        names: target_names,
        effective: target_timings,
        wall: wall_timings,
        symbol_model: symbol_timings,
        symbol_count,
        graph,
    } = build_target_graph(symbol_graph);

    let n = target_names.len();
    if n == 0 {
        return CriticalPathResult {
            critical_path_ms: 0.0,
            path: Vec::new(),
            path_details: Vec::new(),
            all_targets: Vec::new(),
            total_cost: 0.0,
            target_count: 0,
            symbol_count: 0,
            model: None,
            modeled_critical_path_ms: None,
            modeled_path: None,
            modeled_path_details: None,
            modeled_all_targets: None,
            modeled_total_cost: None,
        };
    }

    // Topological sort. With target-level analysis, the graph should always
    // be a DAG (no cycles from dev-dependencies). If there are cycles, it
    // indicates a real circular dependency which is a configuration error.
    let Ok(sorted) = toposort(&graph, None) else {
        eprintln!(
            "WARNING: cycle detected in target dependency graph, \
             skipping critical path computation"
        );
        // Return early with just the target timings (no scheduling computed).
        let total_cost: f64 =
            target_timings.iter().map(|t| t.wall_time_ms).sum();
        let mut all_targets: Vec<TargetOnPath> = target_names
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                let node = NodeIndex::new(idx);
                let timings = &target_timings[idx];
                let dependencies: Vec<String> = graph
                    .neighbors_directed(node, Direction::Incoming)
                    .map(|dep| {
                        target_names.get_index(dep.index()).unwrap().clone()
                    })
                    .collect();
                TargetOnPath {
                    name: name.clone(),
                    frontend_cost: timings.wall_time_ms,
                    start_time: 0.0,
                    finish_time: timings.wall_time_ms,
                    dependencies,
                    wall_timings: wall_timings[idx].clone(),
                    symbol_timings: symbol_timings[idx].clone(),
                }
            })
            .collect();
        all_targets
            .sort_by(|a, b| b.finish_time.partial_cmp(&a.finish_time).unwrap());

        return CriticalPathResult {
            critical_path_ms: total_cost,
            path: Vec::new(),
            path_details: Vec::new(),
            all_targets,
            total_cost,
            target_count: n,
            symbol_count,
            model: None,
            modeled_critical_path_ms: None,
            modeled_path: None,
            modeled_path_details: None,
            modeled_all_targets: None,
            modeled_total_cost: None,
        };
    };

    // Extract effective costs for the actual scheduling pass.
    let effective_costs: Vec<f64> =
        target_timings.iter().map(|t| t.wall_time_ms).collect();

    // --- Actual scheduling pass (wall-clock / per-symbol fallback) ---
    let actual = schedule(&effective_costs, &graph, &sorted);

    // --- Model fitting ---
    // Build regression data from targets with wall-clock profiling data.
    // Each row: [attr, meta, other, wall].
    let regression_data: Vec<(usize, Vec<f64>)> = (0..n)
        .filter(|&idx| wall_timings[idx].wall_time_ms > 0.0)
        .map(|idx| {
            let attr = symbol_timings[idx].wall_time_ms;
            let meta: f64 = wall_timings[idx]
                .event_times_ms
                .iter()
                .filter(|(k, _)| k.starts_with("metadata_decode_"))
                .map(|(_, v)| v)
                .sum();
            let other: f64 = wall_timings[idx]
                .event_times_ms
                .iter()
                .filter(|(k, _)| !k.starts_with("metadata_decode_"))
                .map(|(_, v)| v)
                .sum();
            let wall = wall_timings[idx].wall_time_ms;
            (idx, vec![attr, meta, other, wall])
        })
        .collect();

    // Filter to lib targets only when requested, for model fitting.
    let fit_data: Vec<Vec<f64>> = if options.fit_libs_only {
        regression_data
            .iter()
            .filter(|(idx, _)| {
                target_names
                    .get_index(*idx)
                    .is_some_and(|n| n.ends_with("/lib"))
            })
            .map(|(_, row)| row.clone())
            .collect()
    } else {
        regression_data.iter().map(|(_, row)| row.clone()).collect()
    };

    let model = model_fit(&fit_data);

    // --- Modeled scheduling pass ---
    // Build modeled costs: use model prediction for profiled targets,
    // fall back to effective cost for targets without profiling data.
    // This ensures the modeled critical path uses predictions where
    // available while keeping unprofiled targets at their best estimate.
    let modeled_result = model.as_ref().map(|fit| {
        let modeled_costs: Vec<f64> = (0..n)
            .map(|idx| {
                // Look up this target's regression row if it was profiled.
                regression_data
                    .iter()
                    .find(|(i, _)| *i == idx)
                    .map_or(effective_costs[idx], |(_, row)| {
                        fit.model.predict(&row[..row.len() - 1])
                    })
            })
            .collect();
        schedule(&modeled_costs, &graph, &sorted)
    });

    // --- Build result ---
    // Helper to build TargetOnPath from a ScheduleResult and index.
    let build_target_on_path =
        |sched: &ScheduleResult, costs: &[f64], idx: usize| {
            let node = NodeIndex::new(idx);
            let name = target_names.get_index(idx).unwrap().clone();

            let dependencies: Vec<String> = graph
                .neighbors_directed(node, Direction::Incoming)
                .map(|dep| target_names.get_index(dep.index()).unwrap().clone())
                .collect();

            TargetOnPath {
                name,
                frontend_cost: costs[idx],
                start_time: sched.start[idx],
                finish_time: sched.finish[idx],
                dependencies,
                wall_timings: wall_timings[idx].clone(),
                symbol_timings: symbol_timings[idx].clone(),
            }
        };

    // Build actual path names and details.
    let path: Vec<String> = actual
        .path_nodes
        .iter()
        .map(|node| target_names.get_index(node.index()).unwrap().clone())
        .collect();

    let path_details: Vec<TargetOnPath> = actual
        .path_nodes
        .iter()
        .map(|&node| {
            build_target_on_path(&actual, &effective_costs, node.index())
        })
        .collect();

    let mut all_targets: Vec<TargetOnPath> = (0..n)
        .map(|idx| build_target_on_path(&actual, &effective_costs, idx))
        .collect();
    all_targets
        .sort_by(|a, b| b.finish_time.partial_cmp(&a.finish_time).unwrap());

    // Build modeled path names and details (if model was fitted).
    let (
        modeled_critical_path_ms,
        modeled_path,
        modeled_path_details,
        modeled_all_targets,
        modeled_total_cost,
    ) = if let Some(ref modeled) = modeled_result {
        let modeled_costs: Vec<f64> = (0..n)
            .map(|idx| {
                regression_data.iter().find(|(i, _)| *i == idx).map_or(
                    effective_costs[idx],
                    |(_, row)| {
                        model
                            .as_ref()
                            .unwrap()
                            .model
                            .predict(&row[..row.len() - 1])
                    },
                )
            })
            .collect();

        let m_path: Vec<String> = modeled
            .path_nodes
            .iter()
            .map(|node| target_names.get_index(node.index()).unwrap().clone())
            .collect();

        let m_path_details: Vec<TargetOnPath> = modeled
            .path_nodes
            .iter()
            .map(|&node| {
                build_target_on_path(modeled, &modeled_costs, node.index())
            })
            .collect();

        let mut m_all_targets: Vec<TargetOnPath> = (0..n)
            .map(|idx| build_target_on_path(modeled, &modeled_costs, idx))
            .collect();
        m_all_targets
            .sort_by(|a, b| b.finish_time.partial_cmp(&a.finish_time).unwrap());

        (
            Some(modeled.critical_path_ms),
            Some(m_path),
            Some(m_path_details),
            Some(m_all_targets),
            Some(modeled.total_cost),
        )
    } else {
        (None, None, None, None, None)
    };

    CriticalPathResult {
        critical_path_ms: actual.critical_path_ms,
        path,
        path_details,
        all_targets,
        total_cost: actual.total_cost,
        target_count: n,
        symbol_count,
        model,
        modeled_critical_path_ms,
        modeled_path,
        modeled_path_details,
        modeled_all_targets,
        modeled_total_cost,
    }
}

/// Reads a symbol graph, computes critical path, and writes a report.
///
/// This is the main entry point for the cost analysis phase. It mirrors the
/// API of `tarjanize_condense::run` for consistency.
pub fn run(input: impl Read, output: impl Write) -> std::io::Result<()> {
    run_with_options(input, output, CostOptions::default())
}

/// Reads a symbol graph, computes critical path, and writes a report with
/// custom options.
pub fn run_with_options(
    mut input: impl Read,
    output: impl Write,
    options: CostOptions,
) -> std::io::Result<()> {
    let mut json = String::new();
    input.read_to_string(&mut json)?;

    let symbol_graph: SymbolGraph = serde_json::from_str(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let result = critical_path_with_options(&symbol_graph, options);
    result.write_report_with_options(output, options)
}

/// Convenience function to compute critical path from JSON input.
///
/// Use [`run`] for the standard read-compute-write workflow. This function
/// is useful when you need to inspect the result programmatically.
pub fn critical_path_from_reader(
    mut input: impl Read,
) -> Result<CriticalPathResult, serde_json::Error> {
    let mut json = String::new();
    input
        .read_to_string(&mut json)
        .map_err(serde_json::Error::io)?;
    let symbol_graph: SymbolGraph = serde_json::from_str(&json)?;
    Ok(critical_path(&symbol_graph))
}

/// Builds a target-level dependency graph with three timing views.
///
/// Returns a [`TargetGraph`] containing:
/// - `names`: `IndexSet` of target identifiers (`{package}/{target}`)
/// - `effective`: timings used by the scheduler (wall-clock when profiled,
///   per-symbol fallback otherwise)
/// - `wall`: raw profiler wall-clock times (zeros if not profiled)
/// - `symbol_model`: per-symbol model (frontend = sum of symbol costs)
/// - `symbol_count`: total symbols across all targets
/// - `graph`: `DiGraph` where edges point from dependency to dependent
fn build_target_graph(symbol_graph: &SymbolGraph) -> TargetGraph {
    // Build target name index. Each target gets a unique identifier:
    // `{package}/{target}`.
    // Example: "my-package/lib", "my-package/test", "my-package/bin/cli".
    let mut target_names: IndexSet<String> = IndexSet::new();
    let mut effective_map: HashMap<String, TargetTimings> = HashMap::new();
    let mut wall_map: HashMap<String, TargetTimings> = HashMap::new();
    let mut symbol_model_map: HashMap<String, TargetTimings> = HashMap::new();
    let mut symbol_count = 0;

    for (package_name, package) in &symbol_graph.packages {
        for (target_key, target_data) in &package.targets {
            let target_id = format!("{package_name}/{target_key}");
            target_names.insert(target_id.clone());

            symbol_count += count_symbols_in_module(&target_data.root);

            // Wall-clock: raw profiler data, may be zeros.
            let wall = target_data.timings.clone();

            // Per-symbol model: built purely from per-symbol costs.
            // frontend = sum of all symbol event_times_ms values
            //
            // The validation table compares this against wall-clock to
            // measure how much per-symbol attribution captures. The gap
            // reveals unattributed time (crate-level work, external
            // queries, etc.) tracked in event_times_ms but not assigned
            // to individual symbols.
            let symbol_model = TargetTimings {
                wall_time_ms: collect_frontend_cost(&target_data.root),
                event_times_ms: target_data.timings.event_times_ms.clone(),
            };

            // Effective: wall-clock when profiled, per-symbol fallback
            // otherwise. The scheduler uses these for critical path
            // computation.
            let effective = if wall.wall_time_ms > 0.0 {
                wall.clone()
            } else {
                symbol_model.clone()
            };

            wall_map.insert(target_id.clone(), wall);
            symbol_model_map.insert(target_id.clone(), symbol_model);
            effective_map.insert(target_id, effective);
        }
    }

    // Test targets recompile the lib code (cargo builds with --test flag),
    // so add the lib's frontend costs to test targets. This models the fact
    // that `cargo test` doesn't reuse the lib compilation — it's a fresh
    // build.
    //
    // Augmentation rules:
    // - `symbol_model`: always augmented (per-symbol costs only cover
    //   test-specific code)
    // - `effective`: augmented only when test lacks wall-clock times
    //   (otherwise profiled time already includes lib recompilation)
    // - `wall`: never augmented (raw profiler data, unmodified)
    for (package_name, package) in &symbol_graph.packages {
        if !package.targets.contains_key("test") {
            continue;
        }
        let test_id = format!("{package_name}/test");
        let lib_id = format!("{package_name}/lib");

        let test_has_wall_times = package
            .targets
            .get("test")
            .is_some_and(|t| t.timings.wall_time_ms > 0.0);

        // Always augment symbol_model for test targets — per-symbol costs
        // from the test target only cover test-specific symbols, not the lib
        // code that gets recompiled with `--test`.
        if let Some(lib_sym) = symbol_model_map.get(&lib_id).cloned()
            && let Some(test_sym) = symbol_model_map.get_mut(&test_id)
        {
            test_sym.wall_time_ms += lib_sym.wall_time_ms;
        }

        // Augment effective only when test lacks profiling data.
        if !test_has_wall_times
            && let Some(lib_eff) = effective_map.get(&lib_id).cloned()
            && let Some(test_eff) = effective_map.get_mut(&test_id)
        {
            test_eff.wall_time_ms += lib_eff.wall_time_ms;
        }
    }

    // Collect timing vectors in target_names order.
    let effective: Vec<TargetTimings> = target_names
        .iter()
        .map(|name| effective_map.get(name).cloned().unwrap_or_default())
        .collect();
    let wall: Vec<TargetTimings> = target_names
        .iter()
        .map(|name| wall_map.get(name).cloned().unwrap_or_default())
        .collect();
    let symbol_model: Vec<TargetTimings> = target_names
        .iter()
        .map(|name| symbol_model_map.get(name).cloned().unwrap_or_default())
        .collect();

    // Build target dependency graph.
    let mut graph = DiGraph::<usize, ()>::with_capacity(target_names.len(), 0);
    for i in 0..target_names.len() {
        graph.add_node(i);
    }

    // For each target, add edges from its dependencies.
    // Dependencies come from cargo metadata (target_data.dependencies) in
    // "package/target" format (e.g., "tokio/lib", "serde/lib").
    for (package_name, package) in &symbol_graph.packages {
        for (target_key, target_data) in &package.targets {
            let target_id = format!("{package_name}/{target_key}");
            let Some(target_idx) = target_names.get_index_of(&target_id) else {
                continue;
            };

            for dep_target in &target_data.dependencies {
                // Skip self-dependencies and unknown targets.
                if dep_target == &target_id {
                    continue;
                }
                if let Some(dep_idx) = target_names.get_index_of(dep_target) {
                    // Edge from dependency → dependent.
                    graph.add_edge(
                        NodeIndex::new(dep_idx),
                        NodeIndex::new(target_idx),
                        (),
                    );
                }
            }
        }
    }

    TargetGraph {
        names: target_names,
        effective,
        wall,
        symbol_model,
        symbol_count,
        graph,
    }
}

/// Collects total frontend cost across all symbols in a module tree.
/// Frontend work is serial, so we sum all `event_times_ms` values per symbol.
fn collect_frontend_cost(module: &Module) -> f64 {
    let mut total = 0.0;

    for symbol in module.symbols.values() {
        total += sum_event_times(&symbol.event_times_ms);
    }

    for submodule in module.submodules.values() {
        total += collect_frontend_cost(submodule);
    }

    total
}

/// Formats a millisecond duration as a human-readable string.
///
/// Uses `jiff::SignedDuration` with `SpanPrinter` configured for sign-prefix
/// style (`Direction::Sign`), producing `"-1m 10s"` instead of `"1m 10s ago"`.
///
/// Values ≥ 1s are rounded to the nearest second for cleaner output.
/// Sub-second values keep millisecond precision (rounding would lose all
/// information).
fn format_duration_ms(ms: f64) -> String {
    use jiff::fmt::friendly::{Direction, SpanPrinter};

    #[expect(
        clippy::cast_possible_truncation,
        reason = "practical durations are well within i64 range"
    )]
    let mut dur = jiff::SignedDuration::from_millis(ms.round() as i64);

    // Round to seconds when ≥ 1s; keep milliseconds for sub-second values.
    if dur.abs() >= jiff::SignedDuration::from_secs(1) {
        dur = dur.round(jiff::Unit::Second).unwrap();
    }
    SpanPrinter::new()
        .direction(Direction::Sign)
        .duration_to_string(&dur)
}

/// Truncates a target name to fit within `max_len` characters.
/// If truncation is needed, replaces the last 3 characters with "...".
fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("{}...", &name[..max_len - 3])
    }
}

/// Counts symbols in a module tree.
fn count_symbols_in_module(module: &Module) -> usize {
    let mut count = module.symbols.len();
    for submodule in module.submodules.values() {
        count += count_symbols_in_module(submodule);
    }
    count
}

#[cfg(test)]
mod tests {
    use tarjanize_schemas::{Package, Symbol, SymbolKind, Visibility};

    use super::*;

    /// Creates a symbol with the given frontend cost.
    /// The cost is stored in `event_times_ms` under the `"typeck"` key.
    fn make_symbol(frontend_cost: f64) -> Symbol {
        Symbol {
            file: "test.rs".to_string(),
            event_times_ms: HashMap::from([(
                "typeck".to_string(),
                frontend_cost,
            )]),
            dependencies: std::collections::HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        }
    }

    /// Creates a crate with the given root module and target-level dependencies.
    fn make_crate_with_deps(
        root: Module,
        deps: &[&str],
    ) -> tarjanize_schemas::Crate {
        tarjanize_schemas::Crate {
            root,
            dependencies: deps.iter().map(|&s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    /// Creates a crate with the given root module (no dependencies).
    fn make_crate(root: Module) -> tarjanize_schemas::Crate {
        make_crate_with_deps(root, &[])
    }

    /// Creates a crate with the given root module and wall-clock timings.
    fn make_crate_with_timings(
        root: Module,
        timings: TargetTimings,
    ) -> tarjanize_schemas::Crate {
        tarjanize_schemas::Crate {
            timings,
            root,
            ..Default::default()
        }
    }

    /// Creates a package with a single "lib" target containing the given crate.
    fn make_package(crate_data: tarjanize_schemas::Crate) -> Package {
        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), crate_data);
        Package { targets }
    }

    /// Creates a `SymbolGraph` from a map of package names to crates.
    /// Each crate becomes the "lib" target of a package with the same name.
    fn make_graph(
        crates: HashMap<String, tarjanize_schemas::Crate>,
    ) -> SymbolGraph {
        let packages = crates
            .into_iter()
            .map(|(name, crate_data)| (name, make_package(crate_data)))
            .collect();
        SymbolGraph { packages }
    }

    /// Helper to create a target identifier.
    /// `target_id("pkg", "lib")` returns `"pkg/lib"`.
    fn target_id(package: &str, target: &str) -> String {
        format!("{package}/{target}")
    }

    #[test]
    fn test_single_target_frontend_only() {
        // One target with two symbols, frontend-only: cost = 30
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));
        symbols.insert("bar".to_string(), make_symbol(20.0));

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate(Module {
                symbols,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Frontend is summed: 10 + 20 = 30
        assert!((result.critical_path_ms - 30.0).abs() < f64::EPSILON);
        assert_eq!(result.path, vec![target_id("my-pkg", "lib")]);
        assert_eq!(result.target_count, 1);
        assert_eq!(result.symbol_count, 2);
    }

    #[test]
    fn test_single_target_with_event_times_ms() {
        // Symbol model uses per-symbol costs, not event_times_ms.
        // event_times_ms are preserved for validation/reporting but don't
        // affect scheduling.
        //
        // Per-symbol frontend: foo = 10.
        // event_times_ms: typeck=10, check_mod_type_wf=3 (total 13).
        // Critical path uses per-symbol sum = 10.
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let mut event_times_ms = HashMap::new();
        event_times_ms.insert("typeck".to_string(), 10.0);
        event_times_ms.insert("check_mod_type_wf".to_string(), 3.0);

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate_with_timings(
                Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                TargetTimings {
                    event_times_ms,
                    ..Default::default()
                },
            ),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Per-symbol frontend sum = 10 (event_times_ms don't affect scheduling).
        assert!((result.critical_path_ms - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_target() {
        // One target with two symbols: total cost = 30
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));
        symbols.insert("bar".to_string(), make_symbol(20.0));

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate(Module {
                symbols,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Single target, so critical path = total cost = 30
        assert!((result.critical_path_ms - 30.0).abs() < f64::EPSILON);
        assert_eq!(result.path, vec![target_id("my-pkg", "lib")]);
        assert_eq!(result.target_count, 1);
        assert_eq!(result.symbol_count, 2);
    }

    #[test]
    fn test_two_independent_targets() {
        // Two independent targets: can build in parallel
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a.insert("foo".to_string(), make_symbol(100.0));
        crates.insert(
            "pkg-a".to_string(),
            make_crate(Module {
                symbols: symbols_a,
                submodules: HashMap::new(),
            }),
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert("bar".to_string(), make_symbol(50.0));
        crates.insert(
            "pkg-b".to_string(),
            make_crate(Module {
                symbols: symbols_b,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Critical path is max(100, 50) = 100
        assert!((result.critical_path_ms - 100.0).abs() < f64::EPSILON);
        // Total cost is 150 (sequential)
        assert!((result.total_cost - 150.0).abs() < f64::EPSILON);
        // Parallelism ratio = 150/100 = 1.5x
        assert_eq!(result.target_count, 2);
    }

    #[test]
    fn test_target_chain() {
        // pkg-a/lib (100) depends on pkg-b/lib (50)
        // Critical path = 150
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a.insert("foo".to_string(), make_symbol(100.0));
        crates.insert(
            "pkg-a".to_string(),
            make_crate_with_deps(
                Module {
                    symbols: symbols_a,
                    submodules: HashMap::new(),
                },
                &["pkg-b/lib"],
            ),
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert("bar".to_string(), make_symbol(50.0));
        crates.insert(
            "pkg-b".to_string(),
            make_crate(Module {
                symbols: symbols_b,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // With frontend-only costs, pipelining doesn't help (no backend to overlap).
        // Critical path = pkg-b/lib + pkg-a/lib = 150
        assert!((result.critical_path_ms - 150.0).abs() < f64::EPSILON);
        assert_eq!(
            result.path,
            vec![target_id("pkg-b", "lib"), target_id("pkg-a", "lib")]
        );
        // Total cost = 150 (no parallelism possible)
        assert!((result.total_cost - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diamond_targets() {
        // pkg-a/lib depends on pkg-b/lib (100) and pkg-c/lib (50)
        // pkg-b/lib and pkg-c/lib both depend on pkg-d/lib (30)
        // With frontend-only costs, critical path: d(30) → b(100) → a(10) = 140
        let mut crates = HashMap::new();

        let mut symbols_a = HashMap::new();
        symbols_a.insert("a".to_string(), make_symbol(10.0));
        crates.insert(
            "pkg-a".to_string(),
            make_crate_with_deps(
                Module {
                    symbols: symbols_a,
                    submodules: HashMap::new(),
                },
                &["pkg-b/lib", "pkg-c/lib"],
            ),
        );

        let mut symbols_b = HashMap::new();
        symbols_b.insert("b".to_string(), make_symbol(100.0));
        crates.insert(
            "pkg-b".to_string(),
            make_crate_with_deps(
                Module {
                    symbols: symbols_b,
                    submodules: HashMap::new(),
                },
                &["pkg-d/lib"],
            ),
        );

        let mut symbols_c = HashMap::new();
        symbols_c.insert("c".to_string(), make_symbol(50.0));
        crates.insert(
            "pkg-c".to_string(),
            make_crate_with_deps(
                Module {
                    symbols: symbols_c,
                    submodules: HashMap::new(),
                },
                &["pkg-d/lib"],
            ),
        );

        let mut symbols_d = HashMap::new();
        symbols_d.insert("d".to_string(), make_symbol(30.0));
        crates.insert(
            "pkg-d".to_string(),
            make_crate(Module {
                symbols: symbols_d,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Critical path = d(30) + b(100) + a(10) = 140
        assert!((result.critical_path_ms - 140.0).abs() < f64::EPSILON);
        assert_eq!(
            result.path,
            vec![
                target_id("pkg-d", "lib"),
                target_id("pkg-b", "lib"),
                target_id("pkg-a", "lib")
            ]
        );
        // Total cost = 190
        assert!((result.total_cost - 190.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_graph() {
        let graph = SymbolGraph {
            packages: HashMap::new(),
        };
        let result = critical_path(&graph);

        assert!((result.critical_path_ms).abs() < f64::EPSILON);
        assert!(result.path.is_empty());
        assert_eq!(result.target_count, 0);
        assert_eq!(result.symbol_count, 0);
    }

    #[test]
    fn test_lib_and_test_targets() {
        // Test that lib and test targets are handled separately.
        // pkg-a has both lib (100ms frontend) and test (50ms frontend) targets.
        // The test target depends on the lib target.
        //
        // Test targets include lib costs because `cargo test` recompiles the lib
        // with --test flag. So test's effective cost = 50 + 100 = 150.
        // Critical path: lib(100) → test(150) = 250

        let mut lib_symbols = HashMap::new();
        lib_symbols.insert("lib_fn".to_string(), make_symbol(100.0));

        let mut test_symbols = HashMap::new();
        test_symbols.insert("test_fn".to_string(), make_symbol(50.0));

        let mut targets = HashMap::new();
        targets.insert(
            "lib".to_string(),
            make_crate(Module {
                symbols: lib_symbols,
                submodules: HashMap::new(),
            }),
        );
        targets.insert(
            "test".to_string(),
            make_crate_with_deps(
                Module {
                    symbols: test_symbols,
                    submodules: HashMap::new(),
                },
                &["pkg-a/lib"],
            ),
        );

        let mut packages = HashMap::new();
        packages.insert("pkg-a".to_string(), Package { targets });

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        // Critical path: lib(100) → test(150) = 250
        // (test includes lib's 100ms frontend cost)
        assert!((result.critical_path_ms - 250.0).abs() < f64::EPSILON);
        assert_eq!(
            result.path,
            vec![target_id("pkg-a", "lib"), target_id("pkg-a", "test")]
        );
        assert_eq!(result.target_count, 2);
    }

    #[test]
    #[expect(
        clippy::similar_names,
        reason = "lib_a/lib_b and pkg_a/pkg_b are intentional"
    )]
    fn test_dev_dependency_no_cycle() {
        // This test verifies that dev-dependencies don't create cycles
        // at the target level.
        //
        // Setup:
        // - pkg-a/lib (100ms)
        // - pkg-a/test (50ms + 100ms lib cost = 150ms) depends on pkg-a/lib and pkg-b/lib
        // - pkg-b/lib (30ms) depends on pkg-a/lib
        //
        // At the package level, this would look like a cycle:
        //   pkg-a → pkg-b → pkg-a (via dev-deps)
        //
        // At the target level, there's no cycle:
        //   pkg-a/lib → pkg-b/lib → pkg-a/test
        //   pkg-a/lib → pkg-a/test
        //
        // Critical path: pkg-a/lib(100) → pkg-b/lib(30) → pkg-a/test(150) = 280
        // (test includes lib's 100ms frontend cost)

        // pkg-a/lib
        let mut lib_a_symbols = HashMap::new();
        lib_a_symbols.insert("lib_a_fn".to_string(), make_symbol(100.0));

        // pkg-a/test (depends on pkg-a/lib and pkg-b/lib)
        let mut test_a_symbols = HashMap::new();
        test_a_symbols.insert("test_a_fn".to_string(), make_symbol(50.0));

        // pkg-b/lib (depends on pkg-a/lib)
        let mut lib_b_symbols = HashMap::new();
        lib_b_symbols.insert("lib_b_fn".to_string(), make_symbol(30.0));

        let mut pkg_a_targets = HashMap::new();
        pkg_a_targets.insert(
            "lib".to_string(),
            make_crate(Module {
                symbols: lib_a_symbols,
                submodules: HashMap::new(),
            }),
        );
        pkg_a_targets.insert(
            "test".to_string(),
            make_crate_with_deps(
                Module {
                    symbols: test_a_symbols,
                    submodules: HashMap::new(),
                },
                &["pkg-a/lib", "pkg-b/lib"],
            ),
        );

        let mut pkg_b_targets = HashMap::new();
        pkg_b_targets.insert(
            "lib".to_string(),
            make_crate_with_deps(
                Module {
                    symbols: lib_b_symbols,
                    submodules: HashMap::new(),
                },
                &["pkg-a/lib"],
            ),
        );

        let mut packages = HashMap::new();
        packages.insert(
            "pkg-a".to_string(),
            Package {
                targets: pkg_a_targets,
            },
        );
        packages.insert(
            "pkg-b".to_string(),
            Package {
                targets: pkg_b_targets,
            },
        );

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        // Should compute successfully (no cycle)
        assert!(!result.path.is_empty(), "path should not be empty");

        // With frontend-only costs (test includes lib cost):
        // Critical path: pkg-a/lib(100) → pkg-b/lib(30) → pkg-a/test(150) = 280
        assert!((result.critical_path_ms - 280.0).abs() < f64::EPSILON);
        assert_eq!(
            result.path,
            vec![
                target_id("pkg-a", "lib"),
                target_id("pkg-b", "lib"),
                target_id("pkg-a", "test")
            ]
        );

        // Total cost = 100 + 30 + 150 = 280
        assert!((result.total_cost - 280.0).abs() < f64::EPSILON);
        assert_eq!(result.target_count, 3);
    }

    #[test]
    fn test_wall_clock_chain() {
        // When wall-clock times are available, scheduling uses them directly.
        //
        // Setup:
        // - pkg-a/lib: frontend=120
        // - pkg-b/lib: frontend=30, depends on pkg-a/lib
        //
        // Critical path: pkg-a(120) → pkg-b(30) = 150

        let mut symbols_a = HashMap::new();
        symbols_a.insert("a".to_string(), make_symbol(1.0));

        let mut symbols_b = HashMap::new();
        symbols_b.insert("b".to_string(), make_symbol(1.0));

        let mut crates = HashMap::new();
        crates.insert(
            "pkg-a".to_string(),
            make_crate_with_timings(
                Module {
                    symbols: symbols_a,
                    submodules: HashMap::new(),
                },
                TargetTimings {
                    wall_time_ms: 120.0,
                    ..Default::default()
                },
            ),
        );
        crates.insert(
            "pkg-b".to_string(),
            tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time_ms: 30.0,
                    ..Default::default()
                },
                root: Module {
                    symbols: symbols_b,
                    submodules: HashMap::new(),
                },
                dependencies: ["pkg-a/lib".to_string()].into_iter().collect(),
            },
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Critical path = 120 + 30 = 150
        assert!((result.critical_path_ms - 150.0).abs() < f64::EPSILON);

        let pkg_a = result
            .all_targets
            .iter()
            .find(|t| t.name == "pkg-a/lib")
            .unwrap();
        assert!((pkg_a.finish_time - 120.0).abs() < f64::EPSILON);

        // pkg-b starts when pkg-a finishes (120), finishes at 150
        let pkg_b = result
            .all_targets
            .iter()
            .find(|t| t.name == "pkg-b/lib")
            .unwrap();
        assert!((pkg_b.start_time - 120.0).abs() < f64::EPSILON);
        assert!((pkg_b.finish_time - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wall_time_used_when_available() {
        // When TargetTimings has non-zero wall_time_ms, wall-clock times
        // should be used directly instead of computing from per-symbol costs.
        //
        // Setup: one target with per-symbol costs that differ from wall-clock.
        // - Per-symbol: frontend=10 (sum)
        // - Wall-clock: frontend=25
        //
        // The cost model should use wall-clock frontend (25), not per-symbol (10).

        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let crate_data = tarjanize_schemas::Crate {
            timings: TargetTimings {
                wall_time_ms: 25.0,
                ..Default::default()
            },
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        };

        let mut crates = HashMap::new();
        crates.insert("my-pkg".to_string(), crate_data);

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Should use wall-clock frontend (25), not per-symbol (10).
        assert!((result.critical_path_ms - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_zero_wall_time_falls_back_to_per_symbol() {
        // When wall_time_ms is 0 (synthetic crate or no profiling),
        // the cost model falls back to per-symbol computation.
        //
        // Setup: one target with wall_time_ms=0 but has symbol costs.
        // - Per-symbol: frontend=25+15=40
        // - Expected: frontend=40

        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(25.0));
        symbols.insert("bar".to_string(), make_symbol(15.0));

        let crate_data = tarjanize_schemas::Crate {
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        };

        let mut crates = HashMap::new();
        crates.insert("my-pkg".to_string(), crate_data);

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Fallback: frontend=40
        assert!((result.critical_path_ms - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wall_time_test_target_no_double_count() {
        // When wall-clock times are available from profiling, test targets
        // should NOT have lib costs added, because `rustc --test` recompiles
        // the entire crate (lib + test code). The profiled wall-clock time
        // for the test target already includes lib code compilation.
        //
        // Setup:
        // - pkg-a/lib: wall-clock frontend=100
        // - pkg-a/test: wall-clock frontend=150 (already includes lib's 100),
        //   depends on pkg-a/lib
        //
        // Expected: test effective frontend = 150 (NOT 150 + 100 = 250)
        // Critical path: lib(100) → test(150) = 250

        let mut lib_symbols = HashMap::new();
        lib_symbols.insert("lib_fn".to_string(), make_symbol(1.0));

        let mut test_symbols = HashMap::new();
        test_symbols.insert("test_fn".to_string(), make_symbol(1.0));

        let mut targets = HashMap::new();
        targets.insert(
            "lib".to_string(),
            tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time_ms: 100.0,
                    ..Default::default()
                },
                root: Module {
                    symbols: lib_symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            },
        );
        targets.insert(
            "test".to_string(),
            tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time_ms: 150.0, // Already includes lib's 100ms
                    ..Default::default()
                },
                root: Module {
                    symbols: test_symbols,
                    submodules: HashMap::new(),
                },
                dependencies: ["pkg-a/lib".to_string()].into_iter().collect(),
            },
        );

        let mut packages = HashMap::new();
        packages.insert("pkg-a".to_string(), Package { targets });

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        // Test's wall-clock frontend is 150 (already includes lib).
        // It should NOT be augmented to 250.
        //
        // lib: start=0, finish=100
        // test: start=100 (after lib finishes), finish=100+150=250
        // Critical path = 250
        let test_target = result
            .all_targets
            .iter()
            .find(|t| t.name == "pkg-a/test")
            .unwrap();

        assert!(
            (test_target.frontend_cost - 150.0).abs() < f64::EPSILON,
            "Test frontend should be 150 (wall-clock, already includes lib), \
             got {} (likely double-counted)",
            test_target.frontend_cost
        );
        assert!(
            (result.critical_path_ms - 250.0).abs() < f64::EPSILON,
            "Critical path should be 250, got {}",
            result.critical_path_ms
        );
    }

    #[test]
    fn test_lib_wall_clock_test_per_symbol_augments() {
        // When lib has wall-clock times but test uses per-symbol fallback,
        // the test target SHOULD be augmented with lib costs. The test
        // target's per-symbol costs only cover test-specific code, not the
        // lib code that gets recompiled with `--test`.
        //
        // Setup:
        // - pkg-a/lib: wall-clock frontend=100
        // - pkg-a/test: per-symbol frontend=20 (test-only code)
        //   After augmentation: frontend=20+100=120

        let mut lib_symbols = HashMap::new();
        lib_symbols.insert("lib_fn".to_string(), make_symbol(1.0));

        let mut test_symbols = HashMap::new();
        test_symbols.insert("test_fn".to_string(), make_symbol(20.0));

        let mut targets = HashMap::new();
        targets.insert(
            "lib".to_string(),
            tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time_ms: 100.0,
                    ..Default::default()
                },
                root: Module {
                    symbols: lib_symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            },
        );
        targets.insert(
            "test".to_string(),
            // No wall-clock times → per-symbol fallback.
            make_crate_with_deps(
                Module {
                    symbols: test_symbols,
                    submodules: HashMap::new(),
                },
                &["pkg-a/lib"],
            ),
        );

        let mut packages = HashMap::new();
        packages.insert("pkg-a".to_string(), Package { targets });

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        let test_target = result
            .all_targets
            .iter()
            .find(|t| t.name == "pkg-a/test")
            .unwrap();

        // Per-symbol frontend (20) + lib wall-clock frontend (100) = 120.
        assert!(
            (test_target.frontend_cost - 120.0).abs() < f64::EPSILON,
            "Test frontend should be augmented to 120, got {}",
            test_target.frontend_cost
        );
    }

    #[test]
    fn test_scheduling_uses_wall_time_ms() {
        // Verify that scheduling uses wall_time_ms:
        //   finish = start + wall_time_ms
        //
        // Setup: single target with wall-clock frontend=40.

        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(1.0));

        let crate_data = tarjanize_schemas::Crate {
            timings: TargetTimings {
                wall_time_ms: 40.0,
                ..Default::default()
            },
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        };

        let mut crates = HashMap::new();
        crates.insert("my-pkg".to_string(), crate_data);

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        let target = &result.all_targets[0];

        assert!(
            (target.start_time).abs() < f64::EPSILON,
            "start should be 0, got {}",
            target.start_time
        );
        assert!(
            (target.finish_time - 40.0).abs() < f64::EPSILON,
            "finish should be frontend = 40, got {}",
            target.finish_time
        );
        assert!(
            (target.frontend_cost - 40.0).abs() < f64::EPSILON,
            "frontend_cost should be 40, got {}",
            target.frontend_cost
        );
    }

    #[test]
    fn test_validation_both_timings_populated() {
        // When a target has both wall-clock and per-symbol data, both
        // should be available in TargetOnPath for comparison. The
        // effective timings (used by the scheduler) should use wall-clock,
        // but the per-symbol model timings should also be populated.
        //
        // Setup: one target with wall-clock frontend=25 and per-symbol
        // frontend=10.

        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let crate_data = tarjanize_schemas::Crate {
            timings: TargetTimings {
                wall_time_ms: 25.0,
                ..Default::default()
            },
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        };

        let mut crates = HashMap::new();
        crates.insert("my-pkg".to_string(), crate_data);

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        let target = &result.all_targets[0];

        // Effective (scheduler) uses wall-clock frontend.
        assert!(
            (target.frontend_cost - 25.0).abs() < f64::EPSILON,
            "effective frontend should be wall-clock 25, got {}",
            target.frontend_cost
        );

        // Wall timings should match profiler data exactly.
        assert!(
            (target.wall_timings.wall_time_ms - 25.0).abs() < f64::EPSILON,
            "wall frontend should be 25, got {}",
            target.wall_timings.wall_time_ms
        );

        // Symbol model timings should be computed from per-symbol costs.
        assert!(
            (target.symbol_timings.wall_time_ms - 10.0).abs() < f64::EPSILON,
            "symbol frontend should be 10, got {}",
            target.symbol_timings.wall_time_ms
        );
    }

    #[test]
    fn test_validation_table_appears_with_wall_data() {
        // Verify that the validation table is emitted when wall-clock
        // data is present.

        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let crate_data = tarjanize_schemas::Crate {
            timings: TargetTimings {
                wall_time_ms: 50.0,
                ..Default::default()
            },
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        };

        let mut crates = HashMap::new();
        crates.insert("my-pkg".to_string(), crate_data);

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        let mut output = Vec::new();
        result.write_report(&mut output).unwrap();
        let report = String::from_utf8(output).unwrap();

        assert!(
            report.contains("Model validation"),
            "report should contain validation table when wall data exists"
        );
        assert!(
            report.contains("my-pkg/lib"),
            "validation table should contain the target name"
        );
        assert!(
            report.contains("Actual"),
            "validation table should have Actual column"
        );
    }

    #[test]
    fn test_validation_table_absent_without_wall_data() {
        // No wall-clock data → no validation table.

        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate(Module {
                symbols,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        let mut output = Vec::new();
        result.write_report(&mut output).unwrap();
        let report = String::from_utf8(output).unwrap();

        assert!(
            !report.contains("Model validation"),
            "report should NOT contain validation table without wall data"
        );
    }

    #[test]
    fn test_event_times_ms_dont_affect_scheduling() {
        // event_times_ms are preserved for validation but the symbol model
        // uses per-symbol costs for scheduling, regardless of what
        // event_times_ms contains.
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let mut event_times_ms = HashMap::new();
        // Frontend events total 15, backend events total 28.
        // None of these affect scheduling — only per-symbol costs do.
        event_times_ms.insert("typeck".to_string(), 10.0);
        event_times_ms.insert("check_mod_type_wf".to_string(), 5.0);
        event_times_ms.insert("LLVM_module_codegen".to_string(), 20.0);
        event_times_ms.insert("link_crate".to_string(), 8.0);

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate_with_timings(
                Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                TargetTimings {
                    event_times_ms,
                    ..Default::default()
                },
            ),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Per-symbol: frontend=10, backend=0. event_times_ms ignored.
        assert!(
            (result.critical_path_ms - 10.0).abs() < f64::EPSILON,
            "Expected 10 (per-symbol only), got {}",
            result.critical_path_ms
        );
    }

    #[test]
    fn test_event_times_ms_dont_override_per_symbol_sum() {
        // Even when event_times_ms contains more time than per-symbol
        // costs (due to unattributed work), the symbol model always
        // uses per-symbol costs. The gap between event_times_ms total
        // and per-symbol sum is visible in the validation table.
        let mut symbols = HashMap::new();
        // Per-symbol total: 10 + 20 = 30.
        symbols.insert("foo".to_string(), make_symbol(10.0));
        symbols.insert("bar".to_string(), make_symbol(20.0));

        let mut event_times_ms = HashMap::new();
        // event_times_ms total: 30 + 15 = 45 (includes 15ms of
        // unattributed check_mod_type_wf time).
        event_times_ms.insert("typeck".to_string(), 30.0);
        event_times_ms.insert("check_mod_type_wf".to_string(), 15.0);

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate_with_timings(
                Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                TargetTimings {
                    event_times_ms,
                    ..Default::default()
                },
            ),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // Per-symbol sum (30) is used, not event_times_ms total (45).
        assert!(
            (result.critical_path_ms - 30.0).abs() < f64::EPSILON,
            "Expected per-symbol sum = 30, got {} \
             (event_times_ms total would be 45)",
            result.critical_path_ms
        );
    }

    #[test]
    fn test_event_times_ms_with_chain() {
        // event_times_ms don't affect scheduling — only per-symbol frontend
        // costs matter. This test verifies scheduling with per-symbol
        // costs even when event_times_ms are present.
        //
        // pkg-a/lib: per-symbol frontend=30
        //            event_times_ms frontend=50 (ignored for scheduling)
        // pkg-b/lib: per-symbol frontend=10, depends on pkg-a/lib
        //
        // Critical path: pkg-a(30) → pkg-b(10) = 40
        let mut symbols_a = HashMap::new();
        symbols_a.insert("a".to_string(), make_symbol(30.0));

        let mut event_times_ms_a = HashMap::new();
        event_times_ms_a.insert("typeck".to_string(), 35.0);
        event_times_ms_a.insert("resolve_instance".to_string(), 15.0);

        let mut symbols_b = HashMap::new();
        symbols_b.insert("b".to_string(), make_symbol(10.0));

        let mut crates = HashMap::new();
        crates.insert(
            "pkg-a".to_string(),
            make_crate_with_timings(
                Module {
                    symbols: symbols_a,
                    submodules: HashMap::new(),
                },
                TargetTimings {
                    event_times_ms: event_times_ms_a,
                    ..Default::default()
                },
            ),
        );
        crates.insert(
            "pkg-b".to_string(),
            tarjanize_schemas::Crate {
                root: Module {
                    symbols: symbols_b,
                    submodules: HashMap::new(),
                },
                dependencies: ["pkg-a/lib".to_string()].into_iter().collect(),
                ..Default::default()
            },
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        // pkg-a: frontend=30 (per-symbol)
        // pkg-b: frontend=10, starts at 30 (after pkg-a finishes)
        // Critical path = 30 + 10 = 40
        assert!(
            (result.critical_path_ms - 40.0).abs() < f64::EPSILON,
            "Expected 40, got {}",
            result.critical_path_ms
        );

        let pkg_a = result
            .all_targets
            .iter()
            .find(|t| t.name == "pkg-a/lib")
            .unwrap();
        assert!(
            (pkg_a.finish_time - 30.0).abs() < f64::EPSILON,
            "pkg-a finish should be at 30 (per-symbol frontend)",
        );
    }

    // =========================================================================
    // model_fit tests
    // =========================================================================

    #[test]
    fn test_validation_includes_model_fit() {
        // Build 5 targets with wall-clock data, metadata decode events, and
        // other events. The model fit and per-target predictions should appear.
        let mut packages = HashMap::new();
        for (pkg, sym_cost, meta_cost, other_cost, wall) in [
            ("a", 100.0, 20.0, 10.0, 440.0),
            ("b", 50.0, 40.0, 20.0, 340.0),
            ("c", 200.0, 10.0, 5.0, 725.0),
            ("d", 30.0, 80.0, 15.0, 405.0),
            ("e", 150.0, 30.0, 25.0, 625.0),
        ] {
            let mut symbols = HashMap::new();
            symbols.insert("fn1".to_string(), make_symbol(sym_cost));

            let mut event_times_ms = HashMap::new();
            event_times_ms.insert(
                "metadata_decode_entry_generics_of".to_string(),
                meta_cost,
            );
            event_times_ms.insert("typeck".to_string(), other_cost);

            let crate_data = tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time_ms: wall,
                    event_times_ms,
                },
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            };

            let mut targets = HashMap::new();
            targets.insert("lib".to_string(), crate_data);
            packages.insert(pkg.to_string(), Package { targets });
        }

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        let mut output = Vec::new();
        result.write_report(&mut output).unwrap();
        let report = String::from_utf8(output).unwrap();

        assert!(
            report.contains("Model fit:"),
            "report should contain model fit when profiling data \
             has events. Report:\n{report}",
        );
        assert!(report.contains("R²:"), "report should contain R²",);
        // The table should show prediction columns.
        assert!(
            report.contains("Predicted"),
            "report should contain Predicted column. Report:\n{report}",
        );
        assert!(
            report.contains("Delta"),
            "report should contain Delta column. Report:\n{report}",
        );
        assert!(
            report.contains("Out"),
            "report should contain Out column. Report:\n{report}",
        );
        assert!(
            report.contains("Outliers excluded:"),
            "report should contain outlier count. Report:\n{report}",
        );
    }

    #[test]
    fn test_model_fit_perfect_3var() {
        // wall = 3·attr + 2·meta + 4·other, 5 data points
        let data: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0, 1.0, 9.0],   // 3+2+4
            vec![2.0, 3.0, 0.5, 14.0],  // 6+6+2
            vec![4.0, 1.0, 2.0, 22.0],  // 12+2+8
            vec![0.5, 5.0, 1.0, 15.5],  // 1.5+10+4
            vec![10.0, 2.0, 3.0, 46.0], // 30+4+12
        ];
        let fit = model_fit(&data).expect("should fit");
        let c = &fit.model.coeffs;
        assert!((c[0] - 3.0).abs() < 1e-6, "Expected a=3.0, got {}", c[0],);
        assert!((c[1] - 2.0).abs() < 1e-6, "Expected b=2.0, got {}", c[1],);
        assert!((c[2] - 4.0).abs() < 1e-6, "Expected c=4.0, got {}", c[2],);
        assert!(
            (fit.r_squared - 1.0).abs() < 1e-6,
            "Expected R²=1.0, got {}",
            fit.r_squared,
        );
    }

    #[test]
    fn test_model_fit_perfect_2var() {
        // wall = 3·attr + 2·meta, 5 data points (2-var)
        let data: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0, 5.0],   // 3*1 + 2*1 = 5
            vec![2.0, 3.0, 12.0],  // 3*2 + 2*3 = 12
            vec![4.0, 1.0, 14.0],  // 3*4 + 2*1 = 14
            vec![0.5, 5.0, 11.5],  // 3*0.5 + 2*5 = 11.5
            vec![10.0, 2.0, 34.0], // 3*10 + 2*2 = 34
        ];
        let fit = model_fit(&data).expect("should fit");
        let c = &fit.model.coeffs;
        assert!((c[0] - 3.0).abs() < 1e-10, "Expected a=3.0, got {}", c[0],);
        assert!((c[1] - 2.0).abs() < 1e-10, "Expected b=2.0, got {}", c[1],);
        assert!(
            (fit.r_squared - 1.0).abs() < 1e-10,
            "Expected R²=1.0, got {}",
            fit.r_squared,
        );
    }

    #[test]
    fn test_model_fit_zero_metadata() {
        // All metadata values zero → degenerates to wall = a·attr.
        let data: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 3.0],
            vec![2.0, 0.0, 6.0],
            vec![5.0, 0.0, 15.0],
            vec![10.0, 0.0, 30.0],
        ];
        let fit = model_fit(&data).expect("should fit");
        let c = &fit.model.coeffs;
        assert!((c[0] - 3.0).abs() < 1e-10, "Expected a=3.0, got {}", c[0],);
        assert!(
            (fit.r_squared - 1.0).abs() < 1e-10,
            "Expected R²=1.0, got {}",
            fit.r_squared,
        );
    }

    #[test]
    fn test_model_fit_zero_attribution() {
        // All attr values zero → degenerates to wall = b·meta.
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 1.0, 2.0],
            vec![0.0, 3.0, 6.0],
            vec![0.0, 5.0, 10.0],
            vec![0.0, 10.0, 20.0],
        ];
        let fit = model_fit(&data).expect("should fit");
        let c = &fit.model.coeffs;
        assert!((c[1] - 2.0).abs() < 1e-10, "Expected b=2.0, got {}", c[1],);
        assert!(
            (fit.r_squared - 1.0).abs() < 1e-10,
            "Expected R²=1.0, got {}",
            fit.r_squared,
        );
    }

    #[test]
    fn test_model_fit_insufficient_data() {
        // Fewer than n_vars + 1 points → None.
        let empty: Vec<Vec<f64>> = vec![];
        assert!(model_fit(&empty).is_none());
        assert!(model_fit(&[vec![1.0, 2.0, 3.0]]).is_none());
        assert!(
            model_fit(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).is_none()
        );
    }

    #[test]
    fn test_model_fit_realistic() {
        // Simulated data with a≈3.4, b≈2.8 and some noise.
        let data: Vec<Vec<f64>> = vec![
            vec![100.0, 20.0, 398.0],   // 3.4*100 + 2.8*20 = 396
            vec![50.0, 40.0, 283.0],    // 3.4*50 + 2.8*40 = 282
            vec![200.0, 10.0, 710.0],   // 3.4*200 + 2.8*10 = 708
            vec![30.0, 80.0, 325.0],    // 3.4*30 + 2.8*80 = 326
            vec![150.0, 50.0, 651.0],   // 3.4*150 + 2.8*50 = 650
            vec![500.0, 100.0, 1982.0], // 3.4*500 + 2.8*100 = 1980
            vec![10.0, 5.0, 49.0],      // 3.4*10 + 2.8*5 = 48
        ];
        let fit = model_fit(&data).expect("should fit");
        assert!(
            fit.r_squared > 0.99,
            "Expected R² > 0.99, got {}",
            fit.r_squared,
        );
        let c = &fit.model.coeffs;
        assert!((c[0] - 3.4).abs() < 0.1, "Expected a ≈ 3.4, got {}", c[0],);
        assert!((c[1] - 2.8).abs() < 0.1, "Expected b ≈ 2.8, got {}", c[1],);
    }

    #[test]
    fn test_validation_table_marks_outliers() {
        // Build targets where one is a clear outlier. The model should
        // detect it via MAGSAC's inlier threshold and mark it with `*`.
        //
        // 7 targets that follow wall ≈ 3·attr + 2·meta + 1·other,
        // plus one egregious outlier whose wall time is 10× the model.
        let mut packages = HashMap::new();
        for (pkg, sym_cost, meta_cost, other_cost, wall) in [
            ("a", 100.0, 20.0, 10.0, 350.0), // 3*100+2*20+1*10=350
            ("b", 50.0, 40.0, 20.0, 250.0),  // 3*50+2*40+1*20=250
            ("c", 200.0, 10.0, 5.0, 625.0),  // 3*200+2*10+1*5=625
            ("d", 30.0, 80.0, 15.0, 265.0),  // 3*30+2*80+1*15=265
            ("e", 150.0, 30.0, 25.0, 535.0), // 3*150+2*30+1*25=535
            ("f", 80.0, 50.0, 30.0, 370.0),  // 3*80+2*50+1*30=370
            ("g", 120.0, 15.0, 10.0, 400.0), // 3*120+2*15+1*10=400
            ("outlier", 10.0, 5.0, 2.0, 5000.0), // model predicts ~42
        ] {
            let mut symbols = HashMap::new();
            symbols.insert("fn1".to_string(), make_symbol(sym_cost));

            let mut event_times_ms = HashMap::new();
            event_times_ms
                .insert("metadata_decode_entry_foo".to_string(), meta_cost);
            event_times_ms.insert("typeck".to_string(), other_cost);

            let crate_data = tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time_ms: wall,
                    event_times_ms,
                },
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            };

            let mut targets = HashMap::new();
            targets.insert("lib".to_string(), crate_data);
            packages.insert(pkg.to_string(), Package { targets });
        }

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        let mut output = Vec::new();
        result.write_report(&mut output).unwrap();
        let report = String::from_utf8(output).unwrap();

        // The outlier target should have an asterisk in its validation
        // table row. Only look at lines after the "Out" header to
        // avoid matching the timing detail table.
        let after_header = report
            .split_once("  Out")
            .expect("report should contain Out header")
            .1;
        let outlier_line = after_header
            .lines()
            .find(|l| l.contains("outlier/lib"))
            .expect("outlier should appear in validation table");
        assert!(
            outlier_line.contains('*'),
            "outlier row should be marked with *. Line: {outlier_line}",
        );

        // Summary should show at least 1 outlier excluded.
        assert!(
            report.contains("Outliers excluded:   1"),
            "summary should show 1 outlier. Report:\n{report}",
        );
    }

    // =========================================================================
    // format_duration_ms tests
    // =========================================================================

    #[test]
    fn test_format_duration_ms() {
        // Zero and sub-second.
        assert_eq!(format_duration_ms(0.0), "0s");
        assert_eq!(format_duration_ms(499.0), "499ms");

        // >= 1s: rounded to nearest second.
        assert_eq!(format_duration_ms(1_000.0), "1s");
        assert_eq!(format_duration_ms(1_500.0), "2s");
        assert_eq!(format_duration_ms(14_300.0), "14s");

        // Minutes and seconds.
        assert_eq!(format_duration_ms(60_000.0), "1m");
        assert_eq!(format_duration_ms(70_000.0), "1m 10s");
        assert_eq!(format_duration_ms(116_000.0), "1m 56s");

        // Hours.
        assert_eq!(format_duration_ms(3_600_000.0), "1h");
        assert_eq!(format_duration_ms(3_661_000.0), "1h 1m 1s");
        assert_eq!(format_duration_ms(7_320_000.0), "2h 2m");

        // Negative values (deltas) — sign prefix, not "ago".
        assert_eq!(format_duration_ms(-1_500.0), "-2s");
        assert_eq!(format_duration_ms(-70_000.0), "-1m 10s");
    }

    // =========================================================================
    // lib-only fit tests
    // =========================================================================

    #[test]
    fn test_validation_fit_lib_only() {
        // Mix of lib and non-lib targets. Model should fit on lib targets
        // only, but predictions should appear for all targets.
        let mut packages = HashMap::new();

        // 5 lib targets — enough for fitting (3-var model needs 4+).
        for (pkg, sym_cost, meta_cost, other_cost, wall) in [
            ("a", 100.0, 20.0, 10.0, 440.0),
            ("b", 50.0, 40.0, 20.0, 340.0),
            ("c", 200.0, 10.0, 5.0, 725.0),
            ("d", 30.0, 80.0, 15.0, 405.0),
            ("e", 150.0, 30.0, 25.0, 625.0),
        ] {
            let mut symbols = HashMap::new();
            symbols.insert("fn1".to_string(), make_symbol(sym_cost));

            let mut event_times_ms = HashMap::new();
            event_times_ms.insert(
                "metadata_decode_entry_generics_of".to_string(),
                meta_cost,
            );
            event_times_ms.insert("typeck".to_string(), other_cost);

            let crate_data = tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time_ms: wall,
                    event_times_ms,
                },
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            };

            let mut targets = HashMap::new();
            targets.insert("lib".to_string(), crate_data);
            packages.insert(pkg.to_string(), Package { targets });
        }

        // Add 1 test target (non-lib) with profiling data to package "a".
        let mut test_symbols = HashMap::new();
        test_symbols.insert("test_fn".to_string(), make_symbol(60.0));

        let mut test_event_times = HashMap::new();
        test_event_times
            .insert("metadata_decode_entry_generics_of".to_string(), 15.0);
        test_event_times.insert("typeck".to_string(), 5.0);

        let test_crate = tarjanize_schemas::Crate {
            timings: TargetTimings {
                wall_time_ms: 250.0,
                event_times_ms: test_event_times,
            },
            root: Module {
                symbols: test_symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        };

        packages
            .get_mut("a")
            .unwrap()
            .targets
            .insert("test".to_string(), test_crate);

        let graph = SymbolGraph { packages };
        let result = critical_path_with_options(
            &graph,
            CostOptions {
                fit_libs_only: true,
            },
        );

        // Model should be fitted (enough lib targets).
        assert!(
            result.model.is_some(),
            "model should be fitted with 5 lib targets",
        );

        // Modeled critical path should be populated.
        assert!(
            result.modeled_critical_path_ms.is_some(),
            "modeled critical path should exist when model is fitted",
        );

        let mut output = Vec::new();
        result
            .write_report_with_options(
                &mut output,
                CostOptions {
                    fit_libs_only: true,
                },
            )
            .unwrap();
        let report = String::from_utf8(output).unwrap();

        // Non-lib target should still appear in predictions.
        assert!(
            report.contains("a/test"),
            "non-lib target should appear in table. Report:\n{report}",
        );

        // Delta column should be present (model was fitted).
        assert!(
            report.contains("Delta"),
            "Delta column should appear. Report:\n{report}",
        );

        // Both critical paths should be shown.
        assert!(
            report.contains("Critical path (wall-clock)"),
            "should show wall-clock critical path. Report:\n{report}",
        );
        assert!(
            report.contains("Critical path (modeled)"),
            "should show modeled critical path. Report:\n{report}",
        );
    }

    // =========================================================================
    // CostModel tests
    // =========================================================================

    #[test]
    fn test_cost_model_json_roundtrip() {
        // Verify CostModel serializes to JSON and deserializes back
        // with all fields intact.
        let model = CostModel {
            coeff_attr: 1.5,
            coeff_meta: 2.3,
            coeff_other: 0.8,
            r_squared: 0.95,
            inlier_threshold: 100.0,
        };

        let json = serde_json::to_string_pretty(&model).unwrap();
        let roundtripped: CostModel = serde_json::from_str(&json).unwrap();

        assert!(
            (roundtripped.coeff_attr - 1.5).abs() < f64::EPSILON,
            "coeff_attr mismatch",
        );
        assert!(
            (roundtripped.coeff_meta - 2.3).abs() < f64::EPSILON,
            "coeff_meta mismatch",
        );
        assert!(
            (roundtripped.coeff_other - 0.8).abs() < f64::EPSILON,
            "coeff_other mismatch",
        );
        assert!(
            (roundtripped.r_squared - 0.95).abs() < f64::EPSILON,
            "r_squared mismatch",
        );
        assert!(
            (roundtripped.inlier_threshold - 100.0).abs() < f64::EPSILON,
            "inlier_threshold mismatch",
        );
    }

    #[test]
    fn test_cost_model_predict() {
        // Verify CostModel::predict applies the 3-variable model directly.
        //
        // wall = 2.0 * attr + 3.0 * meta + 1.5 * other
        //      = 2.0 * 100 + 3.0 * 20 + 1.5 * 10 = 275
        let model = CostModel {
            coeff_attr: 2.0,
            coeff_meta: 3.0,
            coeff_other: 1.5,
            r_squared: 1.0,
            inlier_threshold: 100.0,
        };

        let result = model.predict(100.0, 20.0, 10.0);
        assert!(
            (result - 275.0).abs() < 1e-6,
            "Expected 275.0, got {result}",
        );
    }

    #[test]
    fn test_cost_model_predict_zero_meta_other() {
        // With zero meta and other, only attr matters.
        // wall = 2.0 * 200 + 3.0 * 0 + 1.5 * 0 = 400
        let model = CostModel {
            coeff_attr: 2.0,
            coeff_meta: 3.0,
            coeff_other: 1.5,
            r_squared: 1.0,
            inlier_threshold: 100.0,
        };

        let result = model.predict(200.0, 0.0, 0.0);
        assert!(
            (result - 400.0).abs() < 1e-6,
            "Expected 400.0, got {result}",
        );
    }

    #[test]
    fn test_build_cost_model_basic() {
        // Build a SymbolGraph with 5 packages that have profiling data,
        // run critical_path to get the ModelFit, then build_cost_model
        // and verify the extracted coefficients match the fitted model.
        //
        // Data (wall = 3·attr + 2·meta + 1·other):
        //   pkg  attr  meta   other  wall
        //   a    100   20     10     350
        //   b    50    40     20     250
        //   c    200   10     5      625
        //   d    30    80     15     265
        //   e    150   30     25     535
        let mut packages = HashMap::new();
        for (pkg, sym_cost, meta_cost, other_cost) in [
            ("a", 100.0, 20.0, 10.0),
            ("b", 50.0, 40.0, 20.0),
            ("c", 200.0, 10.0, 5.0),
            ("d", 30.0, 80.0, 15.0),
            ("e", 150.0, 30.0, 25.0),
        ] {
            let wall = 3.0 * sym_cost + 2.0 * meta_cost + 1.0 * other_cost;
            let mut symbols = HashMap::new();
            symbols.insert("fn1".to_string(), make_symbol(sym_cost));

            let mut event_times_ms = HashMap::new();
            event_times_ms
                .insert("metadata_decode_entry_foo".to_string(), meta_cost);
            event_times_ms.insert("typeck".to_string(), other_cost);

            let crate_data = tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time_ms: wall,
                    event_times_ms,
                },
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            };

            let mut targets = HashMap::new();
            targets.insert("lib".to_string(), crate_data);
            packages.insert(pkg.to_string(), Package { targets });
        }

        let graph = SymbolGraph { packages };
        let result = critical_path(&graph);

        // The main model should be fitted.
        assert!(result.model.is_some(), "model should be fitted");

        let cost_model = build_cost_model(&result);
        assert!(cost_model.is_some(), "cost model should be built");

        let cm = cost_model.unwrap();

        // Main model coefficients should be close to 3, 2, 1 (perfect fit).
        assert!(
            (cm.coeff_attr - 3.0).abs() < 0.5,
            "coeff_attr should be ~3.0, got {}",
            cm.coeff_attr,
        );
        assert!(
            (cm.coeff_meta - 2.0).abs() < 0.5,
            "coeff_meta should be ~2.0, got {}",
            cm.coeff_meta,
        );

        // R² should be high (perfect main-model fit).
        assert!(
            cm.r_squared > 0.9,
            "r_squared should be > 0.9, got {}",
            cm.r_squared,
        );
    }

    #[test]
    fn test_build_cost_model_none_without_profiling() {
        // Without profiling data (wall_time_ms = 0), build_cost_model
        // should return None because no model can be fitted.
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let mut crates = HashMap::new();
        crates.insert(
            "my-pkg".to_string(),
            make_crate(Module {
                symbols,
                submodules: HashMap::new(),
            }),
        );

        let graph = make_graph(crates);
        let result = critical_path(&graph);

        assert!(
            build_cost_model(&result).is_none(),
            "should return None without profiling data",
        );
    }
}
