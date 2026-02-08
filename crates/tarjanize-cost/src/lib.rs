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

use indexmap::IndexSet;
use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use tarjanize_schemas::{Module, SymbolGraph, TargetTimings, sum_event_times};
use tarjanize_magsac::{fit_two_var_magsac, r_squared_no_intercept_inliers};

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
}

/// Configuration options for cost reporting.
#[derive(Debug, Clone, Copy, Default)]
pub struct CostOptions {
    /// Fit the two-variable model using lib targets only.
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
    /// The report includes summary statistics and timing tables for all targets.
    pub fn write_report_with_options(
        &self,
        mut w: impl Write,
        options: CostOptions,
    ) -> std::io::Result<()> {
        // Summary statistics.
        writeln!(
            w,
            "Critical path:             {}",
            format_duration_ms(self.critical_path_ms)
        )?;
        writeln!(
            w,
            "Total cost (sequential):   {}",
            format_duration_ms(self.total_cost)
        )?;

        let parallelism = if self.critical_path_ms > 0.0 {
            self.total_cost / self.critical_path_ms
        } else {
            1.0
        };
        writeln!(w, "Parallelism ratio:         {parallelism:.2}x")?;
        writeln!(w, "Target count:              {}", self.target_count)?;
        writeln!(w, "Symbol count:              {}", self.symbol_count)?;

        // Table header for timing details.
        let header = format!(
            "{:>10}  {:>10}  {:>10}  {:<40}  Dependencies",
            "Start", "Finish", "Cost", "Target"
        );
        let separator = "-".repeat(header.len() + 20);

        if !self.path_details.is_empty() {
            writeln!(w, "\nCritical path ({} targets):\n", self.path.len())?;
            writeln!(w, "{header}")?;
            writeln!(w, "{separator}")?;

            for target in &self.path_details {
                writeln!(w, "{target}")?;
            }
        }

        if !self.all_targets.is_empty() {
            writeln!(
                w,
                "\nAll targets by finish time ({} targets):\n",
                self.all_targets.len()
            )?;
            writeln!(w, "{header}")?;
            writeln!(w, "{separator}")?;

            for target in &self.all_targets {
                writeln!(w, "{target}")?;
            }
        }

        self.write_validation_table(&mut w, options.fit_libs_only)?;

        Ok(())
    }

    /// Writes a model validation table showing actual vs predicted compilation
    /// times.
    ///
    /// Only emitted when at least one target has non-zero wall-clock data
    /// from profiling. Fits the two-variable model (`wall = a·attr + b·meta`)
    /// on either all targets with profiling data or lib targets only,
    /// depending on `fit_libs_only`, then shows per-target predictions and
    /// percentage errors. Sorted by absolute error descending so the worst
    /// predictions appear first.
    ///
    /// Falls back to showing just Target + Actual when fewer than 3 fit targets
    /// are available (insufficient for model fitting).
    #[expect(
        clippy::too_many_lines,
        reason = "formatting-heavy table output, splitting would scatter layout logic"
    )]
    fn write_validation_table(
        &self,
        mut w: impl Write,
        fit_libs_only: bool,
    ) -> std::io::Result<()> {
        // Collect targets that have wall-clock profiling data.
        let validated: Vec<&TargetOnPath> = self
            .all_targets
            .iter()
            .filter(|t| t.wall_timings.wall_time_ms > 0.0)
            .collect();

        if validated.is_empty() {
            return Ok(());
        }

        // Build regression data per target: (attr, meta, wall).
        let regression_data: Vec<(f64, f64, f64)> = validated
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
                let wall = t.wall_timings.wall_time_ms;
                (attr, meta, wall)
            })
            .collect();

        let mut lib_count = 0;
        let mut other_count = 0;
        let fit_data = if fit_libs_only {
            let lib_regression_data: Vec<(f64, f64, f64)> = validated
                .iter()
                .zip(&regression_data)
                .filter(|(t, _)| t.name.ends_with("/lib"))
                .map(|(_, &data)| data)
                .collect();
            lib_count = lib_regression_data.len();
            other_count = validated.len() - lib_count;
            lib_regression_data
        } else {
            regression_data.clone()
        };

        let fit = two_var_fit(&fit_data);

        if fit_libs_only {
            writeln!(
                w,
                "\nModel validation ({} targets with profiling, \
                 fit on {} lib targets):\n",
                validated.len(),
                lib_count,
            )?;
        } else {
            writeln!(
                w,
                "\nModel validation ({} targets with profiling):\n",
                validated.len(),
            )?;
        }

        if let Some(ref fit) = fit {
            // Build rows with predictions and sort by absolute error.
            // Predictions use the fitted model for all targets.
            let mut rows: Vec<(&str, f64, f64, f64, f64)> = validated
                .iter()
                .zip(&regression_data)
                .map(|(t, &(attr, meta, _))| {
                    let actual = t.wall_timings.wall_time_ms;
                    let predicted = fit.a * attr + fit.b * meta;
                    let delta_ms = predicted - actual;
                    let error_pct = delta_ms / actual * 100.0;
                    (t.name.as_str(), actual, predicted, delta_ms, error_pct)
                })
                .collect();
            rows.sort_by(|a, b| {
                b.4.abs()
                    .partial_cmp(&a.4.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Table header: actual, predicted, delta, error%.
            writeln!(
                w,
                "{:<35} {:>9} {:>9} {:>9}  {:>7}",
                "Target", "Actual", "Predicted", "Delta", "Error%",
            )?;
            writeln!(w, "{}", "-".repeat(74))?;

            for (name, actual, predicted, delta_ms, error_pct) in &rows {
                writeln!(
                    w,
                    "{:<35} {:>9} {:>9} {:>9}  {:>6.1}%",
                    truncate_name(name, 35),
                    format_duration_ms(*actual),
                    format_duration_ms(*predicted),
                    format_duration_ms(*delta_ms),
                    error_pct,
                )?;
            }
        } else {
            // Fewer than 3 targets — can't fit the model. Show actuals.
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
        if fit_libs_only {
            writeln!(
                w,
                "Targets compared:    {} ({} lib, {} other)",
                validated.len(),
                lib_count,
                other_count,
            )?;
        } else {
            writeln!(w, "Targets compared:    {}", validated.len(),)?;
        }

        if let Some(ref fit) = fit {
            writeln!(
                w,
                "Two-var fit:         wall = {:.2} * attr + {:.2} * meta",
                fit.a, fit.b,
            )?;
            writeln!(w, "R² (two-var):        {:.4}", fit.r_squared)?;
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

/// Result of two-variable no-intercept regression.
///
/// Fits `y = a * x1 + b * x2` (no intercept). Used to model
/// `wall_time = a * attr + b * meta` where `attr` is the sum of
/// symbol-attributed event times and `meta` is the sum of
/// `metadata_decode_*` event times.
#[derive(Debug)]
pub struct TwoVarFit {
    /// Coefficient for the first predictor (symbol attribution).
    pub a: f64,
    /// Coefficient for the second predictor (metadata decode).
    pub b: f64,
    /// Coefficient of determination (R²).
    pub r_squared: f64,
}

/// Fits `y = a * x1 + b * x2` (no intercept) using MAGSAC++ for robustness.
///
/// The no-intercept constraint is physically motivated: zero code plus zero
/// dependencies should produce zero compilation time.
///
/// Returns `None` if fewer than 3 data points or if the robust estimator
/// cannot produce a stable model.
///
/// # Arguments
///
/// * `data` - Slice of `(x1, x2, y)` triples: (symbol attribution sum,
///   metadata decode sum, wall time).
pub fn two_var_fit(data: &[(f64, f64, f64)]) -> Option<TwoVarFit> {
    if data.len() < 3 {
        return None;
    }

    let result = fit_two_var_magsac(data)?;
    let r_squared = r_squared_no_intercept_inliers(
        data,
        result.model,
        result.inlier_threshold,
    );

    Some(TwoVarFit {
        a: result.model.a,
        b: result.model.b,
        r_squared,
    })
}

/// Computes the critical path of a symbol graph at the target level.
///
/// Only frontend time matters for scheduling — backend time (LLVM codegen)
/// is parallel via CGUs and doesn't affect the critical path.
///
/// Each compilation target (lib, test, bin, etc.) is a separate node. This
/// naturally resolves dev-dependency "cycles" because test targets depend on
/// lib targets, not vice versa.
#[expect(
    clippy::too_many_lines,
    reason = "core algorithm, splitting would obscure logic"
)]
pub fn critical_path(symbol_graph: &SymbolGraph) -> CriticalPathResult {
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
        };
    }

    let total_cost: f64 = target_timings.iter().map(|t| t.wall_time_ms).sum();

    // Topological sort. With target-level analysis, the graph should always
    // be a DAG (no cycles from dev-dependencies). If there are cycles, it
    // indicates a real circular dependency which is a configuration error.
    let Ok(sorted) = toposort(&graph, None) else {
        eprintln!(
            "WARNING: cycle detected in target dependency graph, \
             skipping critical path computation"
        );
        // Return early with just the target timings (no scheduling computed).
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
        };
    };

    // DP scheduling: process targets in topological order.
    //
    // For each target t:
    //   start[t] = max(finish[dep] for dep in dependencies)
    //   finish[t] = start[t] + wall_time_ms[t]
    //
    // The critical path is the chain of dependencies that determines the
    // latest finish time. We track predecessors based on which dependency's
    // finish time determined each target's start time.
    let mut start: Vec<f64> = vec![0.0; n];
    let mut finish: Vec<f64> = vec![0.0; n];
    let mut predecessor: Vec<Option<NodeIndex>> = vec![None; n];

    for &node in &sorted {
        let idx = node.index();
        let timings = &target_timings[idx];

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
        finish[idx] = max_dep_finish + timings.wall_time_ms;
        predecessor[idx] = max_dep_node;
    }

    // Critical path ends at the target with latest finish time.
    let (max_node, &critical_path_ms) = finish
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // Reconstruct the critical path by following predecessors from the end node.
    let mut path_nodes = Vec::new();
    let mut current = Some(NodeIndex::new(max_node));
    while let Some(node) = current {
        path_nodes.push(node);
        current = predecessor[node.index()];
    }
    path_nodes.reverse();

    // Helper to build TargetOnPath from index.
    let build_target_on_path = |idx: usize| {
        let node = NodeIndex::new(idx);
        let name = target_names.get_index(idx).unwrap().clone();
        let timings = &target_timings[idx];

        let dependencies: Vec<String> = graph
            .neighbors_directed(node, Direction::Incoming)
            .map(|dep| target_names.get_index(dep.index()).unwrap().clone())
            .collect();

        TargetOnPath {
            name,
            frontend_cost: timings.wall_time_ms,
            start_time: start[idx],
            finish_time: finish[idx],
            dependencies,
            wall_timings: wall_timings[idx].clone(),
            symbol_timings: symbol_timings[idx].clone(),
        }
    };

    // Build path names and details.
    let path: Vec<String> = path_nodes
        .iter()
        .map(|node| target_names.get_index(node.index()).unwrap().clone())
        .collect();

    let path_details: Vec<TargetOnPath> = path_nodes
        .iter()
        .map(|&node| build_target_on_path(node.index()))
        .collect();

    // Build all_targets, sorted by finish_time descending.
    let mut all_targets: Vec<TargetOnPath> =
        (0..n).map(build_target_on_path).collect();
    all_targets
        .sort_by(|a, b| b.finish_time.partial_cmp(&a.finish_time).unwrap());

    CriticalPathResult {
        critical_path_ms,
        path,
        path_details,
        all_targets,
        total_cost,
        target_count: n,
        symbol_count,
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

    let result = critical_path(&symbol_graph);
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
    // two_var_fit tests
    // =========================================================================

    #[test]
    fn test_validation_includes_two_var_fit() {
        // Build 4 targets with wall-clock data and metadata decode events.
        // The two-variable fit and per-target predictions should appear.
        let mut packages = HashMap::new();
        for (pkg, sym_cost, meta_cost, wall) in [
            ("a", 100.0, 20.0, 400.0),
            ("b", 50.0, 40.0, 280.0),
            ("c", 200.0, 10.0, 700.0),
            ("d", 30.0, 80.0, 330.0),
        ] {
            let mut symbols = HashMap::new();
            symbols.insert("fn1".to_string(), make_symbol(sym_cost));

            let mut event_times_ms = HashMap::new();
            event_times_ms.insert(
                "metadata_decode_entry_generics_of".to_string(),
                meta_cost,
            );

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
            report.contains("Two-var fit:"),
            "report should contain two-variable fit when profiling data \
             has metadata decode events. Report:\n{report}",
        );
        assert!(
            report.contains("R² (two-var):"),
            "report should contain two-variable R²",
        );
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
            report.contains("Error%"),
            "report should contain Error% column. Report:\n{report}",
        );
    }

    #[test]
    fn test_two_var_fit_perfect() {
        // wall = 3·attr + 2·meta, 5 data points → R²=1.0, a=3.0, b=2.0
        let data: Vec<(f64, f64, f64)> = vec![
            (1.0, 1.0, 5.0),   // 3*1 + 2*1 = 5
            (2.0, 3.0, 12.0),  // 3*2 + 2*3 = 12
            (4.0, 1.0, 14.0),  // 3*4 + 2*1 = 14
            (0.5, 5.0, 11.5),  // 3*0.5 + 2*5 = 11.5
            (10.0, 2.0, 34.0), // 3*10 + 2*2 = 34
        ];
        let fit = two_var_fit(&data).expect("should fit");
        assert!((fit.a - 3.0).abs() < 1e-10, "Expected a=3.0, got {}", fit.a,);
        assert!((fit.b - 2.0).abs() < 1e-10, "Expected b=2.0, got {}", fit.b,);
        assert!(
            (fit.r_squared - 1.0).abs() < 1e-10,
            "Expected R²=1.0, got {}",
            fit.r_squared,
        );
    }

    #[test]
    fn test_two_var_fit_zero_metadata() {
        // All metadata values zero → degenerates to wall = a·attr.
        let data: Vec<(f64, f64, f64)> = vec![
            (1.0, 0.0, 3.0),
            (2.0, 0.0, 6.0),
            (5.0, 0.0, 15.0),
            (10.0, 0.0, 30.0),
        ];
        let fit = two_var_fit(&data).expect("should fit");
        assert!((fit.a - 3.0).abs() < 1e-10, "Expected a=3.0, got {}", fit.a,);
        // b is irrelevant when all x2=0, but coefficient is well-defined (0).
        assert!(
            (fit.r_squared - 1.0).abs() < 1e-10,
            "Expected R²=1.0, got {}",
            fit.r_squared,
        );
    }

    #[test]
    fn test_two_var_fit_zero_attribution() {
        // All attr values zero → degenerates to wall = b·meta.
        let data: Vec<(f64, f64, f64)> = vec![
            (0.0, 1.0, 2.0),
            (0.0, 3.0, 6.0),
            (0.0, 5.0, 10.0),
            (0.0, 10.0, 20.0),
        ];
        let fit = two_var_fit(&data).expect("should fit");
        assert!((fit.b - 2.0).abs() < 1e-10, "Expected b=2.0, got {}", fit.b,);
        assert!(
            (fit.r_squared - 1.0).abs() < 1e-10,
            "Expected R²=1.0, got {}",
            fit.r_squared,
        );
    }

    #[test]
    fn test_two_var_fit_insufficient_data() {
        // Fewer than 3 points → None.
        assert!(two_var_fit(&[]).is_none());
        assert!(two_var_fit(&[(1.0, 2.0, 3.0)]).is_none());
        assert!(two_var_fit(&[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]).is_none());
    }

    #[test]
    fn test_two_var_fit_realistic() {
        // Simulated data with a≈3.4, b≈2.8 and some noise.
        // Points generated as: wall ≈ 3.4·attr + 2.8·meta + noise
        let data: Vec<(f64, f64, f64)> = vec![
            (100.0, 20.0, 398.0),   // 3.4*100 + 2.8*20 = 396
            (50.0, 40.0, 283.0),    // 3.4*50 + 2.8*40 = 282
            (200.0, 10.0, 710.0),   // 3.4*200 + 2.8*10 = 708
            (30.0, 80.0, 325.0),    // 3.4*30 + 2.8*80 = 326
            (150.0, 50.0, 651.0),   // 3.4*150 + 2.8*50 = 650
            (500.0, 100.0, 1982.0), // 3.4*500 + 2.8*100 = 1980
            (10.0, 5.0, 49.0),      // 3.4*10 + 2.8*5 = 48
        ];
        let fit = two_var_fit(&data).expect("should fit");
        assert!(
            fit.r_squared > 0.99,
            "Expected R² > 0.99, got {}",
            fit.r_squared,
        );
        // Coefficients should be close to 3.4 and 2.8.
        assert!((fit.a - 3.4).abs() < 0.1, "Expected a ≈ 3.4, got {}", fit.a,);
        assert!((fit.b - 2.8).abs() < 0.1, "Expected b ≈ 2.8, got {}", fit.b,);
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

        // 4 lib targets — enough for fitting.
        for (pkg, sym_cost, meta_cost, wall) in [
            ("a", 100.0, 20.0, 400.0),
            ("b", 50.0, 40.0, 280.0),
            ("c", 200.0, 10.0, 700.0),
            ("d", 30.0, 80.0, 330.0),
        ] {
            let mut symbols = HashMap::new();
            symbols.insert("fn1".to_string(), make_symbol(sym_cost));

            let mut event_times_ms = HashMap::new();
            event_times_ms.insert(
                "metadata_decode_entry_generics_of".to_string(),
                meta_cost,
            );

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
        let result = critical_path(&graph);

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

        // Header should show lib target count.
        assert!(
            report.contains("fit on 4 lib targets"),
            "header should mention lib target count. Report:\n{report}",
        );

        // Summary should show lib/other breakdown.
        assert!(
            report.contains("4 lib, 1 other"),
            "summary should show lib/other breakdown. Report:\n{report}",
        );

        // Non-lib target should still appear in predictions.
        assert!(
            report.contains("a/test"),
            "non-lib target should appear in table. Report:\n{report}",
        );

        // Delta column should be present.
        assert!(
            report.contains("Delta"),
            "Delta column should appear. Report:\n{report}",
        );
    }
}
