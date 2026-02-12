//! Cost model fitting for symbol graphs.
//!
//! Fits a regression model predicting compilation wall time from three
//! predictors: attributed symbol costs (`attr`), metadata decode time
//! (`meta`), and other unattributed event times (`other`).
//!
//! ```text
//! wall = coeff_attr * attr + coeff_meta * meta + coeff_other * other
//! ```
//!
//! Uses MAGSAC++ for robust regression, automatically detecting and
//! down-weighting outlier targets during fitting.
//!
//! ## Usage
//!
//! The fitted model is consumed by:
//! - `tarjanize viz` for schedule visualization (predicted target costs)
//! - `tarjanize condense` for synthetic crate cost prediction
//!
//! For critical path scheduling and interactive visualization, see
//! `tarjanize-viz`.

use std::io::{Read, Write};

use tarjanize_magsac::{
    LinearModel, fit_magsac, r_squared_no_intercept_inliers,
};
use tarjanize_schemas::{CostModel, Module, SymbolGraph, sum_event_times};

/// Per-target regression predictors and actual wall time.
///
/// Extracted from the `SymbolGraph` for model fitting. The three
/// predictors capture different sources of compilation cost:
/// - `attr`: time attributed to individual symbols (type checking, etc.)
/// - `meta`: time spent decoding external crate metadata
/// - `other`: remaining unattributed event times
#[derive(Debug, Clone)]
struct TargetPredictors {
    /// Target identifier in `{package}/{target}` format.
    name: String,

    /// Sum of all symbol `event_times_ms` values (recursive through
    /// the module tree). Captures frontend work directly attributable
    /// to individual symbols.
    attr: f64,

    /// Sum of `metadata_decode_*` target-level event times.
    /// Captures time spent decoding external crate metadata.
    meta: f64,

    /// Sum of non-`metadata_decode_*` target-level event times.
    /// Captures remaining unattributed frontend work.
    other: f64,

    /// Profiled wall-clock compilation time in milliseconds.
    /// Zero when no profiling data is available.
    wall: f64,

    /// Number of symbols in this target (recursive through module tree).
    symbol_count: usize,
}

/// Result of cost model fitting.
///
/// Contains the per-target predictors, the fitted model (if successful),
/// and summary statistics. Used for report generation and `CostModel`
/// extraction.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Per-target predictor data, one entry per target in the graph.
    targets: Vec<TargetPredictors>,

    /// Fitted regression model, if enough profiled targets were available.
    model: Option<ModelFit>,

    /// Total number of targets in the graph.
    target_count: usize,

    /// Total number of symbols across all targets.
    symbol_count: usize,
}

/// Configuration options for cost model fitting.
#[derive(Debug, Clone, Copy, Default)]
pub struct CostOptions {
    /// Fit the regression model using lib targets only.
    /// Non-lib targets are still shown in the validation table.
    pub fit_libs_only: bool,
}

/// Result of N-variable no-intercept regression via MAGSAC++.
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

/// Extracts a [`CostModel`] from a [`FitResult`].
///
/// Returns `None` if no model was fitted (insufficient profiling data).
/// The coefficients come entirely from the MAGSAC++ regression fitted
/// during [`fit`].
pub fn build_cost_model(result: &FitResult) -> Option<CostModel> {
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

/// Fits a cost model from a `SymbolGraph`.
///
/// Extracts per-target predictors (attr, meta, other) and actual wall
/// times, then fits a MAGSAC++ regression model. When `fit_libs_only`
/// is set, only lib targets are used for fitting (but all targets appear
/// in the result for validation).
pub fn fit(symbol_graph: &SymbolGraph, options: CostOptions) -> FitResult {
    let targets = extract_predictors(symbol_graph);
    let target_count = targets.len();
    let symbol_count: usize = targets.iter().map(|t| t.symbol_count).sum();

    // Build regression data from targets with wall-clock profiling data.
    // Each row: [attr, meta, other, wall].
    let regression_data: Vec<Vec<f64>> = targets
        .iter()
        .filter(|t| t.wall > 0.0)
        .filter(|t| {
            if options.fit_libs_only {
                t.name.ends_with("/lib")
            } else {
                true
            }
        })
        .map(|t| vec![t.attr, t.meta, t.other, t.wall])
        .collect();

    let model = model_fit(&regression_data);

    FitResult {
        targets,
        model,
        target_count,
        symbol_count,
    }
}

/// Reads a symbol graph, fits a cost model, and writes a validation report.
pub fn run(
    mut input: impl Read,
    output: impl Write,
    options: CostOptions,
) -> std::io::Result<()> {
    let mut json = String::new();
    input.read_to_string(&mut json)?;

    let symbol_graph: SymbolGraph = serde_json::from_str(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let result = fit(&symbol_graph, options);
    result.write_report(output)
}

impl FitResult {
    /// Writes a human-readable model validation report.
    ///
    /// Shows summary statistics and a validation table comparing actual
    /// vs predicted compilation times for targets with profiling data.
    pub fn write_report(&self, mut w: impl Write) -> std::io::Result<()> {
        writeln!(w, "Target count:    {}", self.target_count)?;
        writeln!(w, "Symbol count:    {}", self.symbol_count)?;

        self.write_validation_table(&mut w)?;
        Ok(())
    }

    /// Writes a model validation table showing actual vs predicted times.
    ///
    /// Only emitted when at least one target has non-zero wall-clock data
    /// from profiling. Uses MAGSAC's inlier threshold to mark outliers.
    /// Sorted by absolute error descending so the worst predictions
    /// appear first.
    ///
    /// Falls back to showing just Target + Actual when no model was
    /// fitted (insufficient profiled targets).
    fn write_validation_table(&self, mut w: impl Write) -> std::io::Result<()> {
        // Collect targets that have wall-clock profiling data.
        let validated: Vec<&TargetPredictors> =
            self.targets.iter().filter(|t| t.wall > 0.0).collect();

        if validated.is_empty() {
            return Ok(());
        }

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
                .map(|t| {
                    let predicted =
                        fit.model.predict(&[t.attr, t.meta, t.other]);
                    let delta_ms = predicted - t.wall;
                    let is_outlier =
                        (t.wall - predicted).abs() > fit.inlier_threshold;
                    (t.name.as_str(), t.wall, predicted, delta_ms, is_outlier)
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
                    format_duration_ms(target.wall),
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

/// Extracts per-target predictors from a `SymbolGraph`.
///
/// For each target in the graph, computes:
/// - `attr`: sum of all symbol `event_times_ms` (recursive through
///   module tree)
/// - `meta`: sum of `metadata_decode_*` target-level events
/// - `other`: sum of remaining target-level events
/// - `wall`: profiled wall-clock time (zero if not profiled)
/// - `symbol_count`: total symbols in the target
fn extract_predictors(symbol_graph: &SymbolGraph) -> Vec<TargetPredictors> {
    let mut targets = Vec::new();

    for (package_name, package) in &symbol_graph.packages {
        for (target_key, target_data) in &package.targets {
            let name = format!("{package_name}/{target_key}");

            let attr = collect_frontend_cost(&target_data.root);
            let meta: f64 = target_data
                .timings
                .event_times_ms
                .iter()
                .filter(|(k, _)| k.starts_with("metadata_decode_"))
                .map(|(_, v)| v)
                .sum();
            let other: f64 = target_data
                .timings
                .event_times_ms
                .iter()
                .filter(|(k, _)| !k.starts_with("metadata_decode_"))
                .map(|(_, v)| v)
                .sum();
            let wall = target_data.timings.wall_time.as_secs_f64() * 1000.0;
            let symbol_count = count_symbols_in_module(&target_data.root);

            targets.push(TargetPredictors {
                name,
                attr,
                meta,
                other,
                wall,
                symbol_count,
            });
        }
    }

    targets
}

/// Recursively sums all symbol `event_times_ms` in a module tree.
/// Frontend work is serial, so we sum all `event_times_ms` values per
/// symbol.
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

/// Counts symbols in a module tree recursively.
fn count_symbols_in_module(module: &Module) -> usize {
    let mut count = module.symbols.len();
    for submodule in module.submodules.values() {
        count += count_symbols_in_module(submodule);
    }
    count
}

/// Formats a millisecond duration as a human-readable string.
///
/// Uses `jiff::SignedDuration` with `SpanPrinter` configured for
/// sign-prefix style (`Direction::Sign`), producing `"-1m 10s"` instead
/// of `"1m 10s ago"`.
///
/// Values >= 1s are rounded to the nearest second for cleaner output.
/// Sub-second values keep millisecond precision (rounding would lose all
/// information).
fn format_duration_ms(ms: f64) -> String {
    use jiff::fmt::friendly::{Direction, SpanPrinter};

    #[expect(
        clippy::cast_possible_truncation,
        reason = "practical durations are well within i64 range"
    )]
    let mut dur = jiff::SignedDuration::from_millis(ms.round() as i64);

    // Round to seconds when >= 1s; keep milliseconds for sub-second values.
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use tarjanize_schemas::{
        Package, Symbol, SymbolKind, TargetTimings, Visibility,
    };

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

    /// Creates a package with a single "lib" target containing
    /// the given symbols and timings.
    fn make_lib_package(
        symbols: HashMap<String, Symbol>,
        timings: TargetTimings,
    ) -> Package {
        let mut targets = HashMap::new();
        targets.insert(
            "lib".to_string(),
            tarjanize_schemas::Crate {
                timings,
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            },
        );
        Package { targets }
    }

    /// Creates a simple `SymbolGraph` with lib targets having wall-clock
    /// and event time data.
    fn make_profiled_graph(data: &[(&str, f64, f64, f64, f64)]) -> SymbolGraph {
        let mut packages = HashMap::new();

        for &(pkg, sym_cost, meta_cost, other_cost, wall) in data {
            let mut symbols = HashMap::new();
            symbols.insert("fn1".to_string(), make_symbol(sym_cost));

            let mut event_times_ms = HashMap::new();
            event_times_ms
                .insert("metadata_decode_entry_foo".to_string(), meta_cost);
            event_times_ms.insert("typeck".to_string(), other_cost);

            packages.insert(
                pkg.to_string(),
                make_lib_package(
                    symbols,
                    TargetTimings {
                        wall_time: Duration::from_secs_f64(wall / 1000.0),
                        event_times_ms,
                    },
                ),
            );
        }

        SymbolGraph { packages }
    }

    // =====================================================================
    // Predictor extraction tests
    // =====================================================================

    #[test]
    fn test_extract_predictors_basic() {
        // Verify that predictors are correctly extracted from a simple
        // graph with one profiled target.
        let graph = make_profiled_graph(&[("a", 100.0, 20.0, 10.0, 440.0)]);
        let predictors = extract_predictors(&graph);

        assert_eq!(predictors.len(), 1);
        let p = &predictors[0];
        assert_eq!(p.name, "a/lib");
        assert!((p.attr - 100.0).abs() < f64::EPSILON, "attr={}", p.attr);
        assert!((p.meta - 20.0).abs() < f64::EPSILON, "meta={}", p.meta);
        assert!((p.other - 10.0).abs() < f64::EPSILON, "other={}", p.other);
        assert!((p.wall - 440.0).abs() < f64::EPSILON, "wall={}", p.wall);
        assert_eq!(p.symbol_count, 1);
    }

    #[test]
    fn test_extract_predictors_no_profiling() {
        // A target with no wall-clock data should have wall=0.
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(25.0));
        symbols.insert("bar".to_string(), make_symbol(15.0));

        let mut packages = HashMap::new();
        packages.insert(
            "my-pkg".to_string(),
            make_lib_package(symbols, TargetTimings::default()),
        );

        let graph = SymbolGraph { packages };
        let predictors = extract_predictors(&graph);

        assert_eq!(predictors.len(), 1);
        let p = &predictors[0];
        assert!((p.attr - 40.0).abs() < f64::EPSILON);
        assert!((p.wall).abs() < f64::EPSILON, "wall should be 0");
        assert_eq!(p.symbol_count, 2);
    }

    #[test]
    fn test_extract_predictors_multiple_targets() {
        let graph = make_profiled_graph(&[
            ("a", 100.0, 20.0, 10.0, 440.0),
            ("b", 50.0, 40.0, 20.0, 340.0),
        ]);
        let predictors = extract_predictors(&graph);
        assert_eq!(predictors.len(), 2);
    }

    #[test]
    fn test_extract_predictors_empty_graph() {
        let graph = SymbolGraph {
            packages: HashMap::new(),
        };
        let predictors = extract_predictors(&graph);
        assert!(predictors.is_empty());
    }

    // =====================================================================
    // fit() tests
    // =====================================================================

    #[test]
    fn test_fit_basic() {
        // 5 targets with profiling data → model should be fitted.
        let graph = make_profiled_graph(&[
            ("a", 100.0, 20.0, 10.0, 440.0),
            ("b", 50.0, 40.0, 20.0, 340.0),
            ("c", 200.0, 10.0, 5.0, 725.0),
            ("d", 30.0, 80.0, 15.0, 405.0),
            ("e", 150.0, 30.0, 25.0, 625.0),
        ]);
        let result = fit(&graph, CostOptions::default());

        assert_eq!(result.target_count, 5);
        assert_eq!(result.symbol_count, 5); // 1 symbol per target
        assert!(result.model.is_some(), "model should be fitted");
    }

    #[test]
    fn test_fit_no_profiling() {
        // Without profiling data, no model should be fitted.
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let mut packages = HashMap::new();
        packages.insert(
            "my-pkg".to_string(),
            make_lib_package(symbols, TargetTimings::default()),
        );

        let graph = SymbolGraph { packages };
        let result = fit(&graph, CostOptions::default());

        assert_eq!(result.target_count, 1);
        assert!(result.model.is_none());
    }

    #[test]
    fn test_fit_empty_graph() {
        let graph = SymbolGraph {
            packages: HashMap::new(),
        };
        let result = fit(&graph, CostOptions::default());

        assert_eq!(result.target_count, 0);
        assert_eq!(result.symbol_count, 0);
        assert!(result.model.is_none());
    }

    #[test]
    fn test_fit_libs_only() {
        // Mix of lib and non-lib targets. Model should fit on lib targets
        // only, but all targets appear in the result.
        let mut packages = HashMap::new();

        // 5 lib targets — enough for fitting.
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

            packages.insert(
                pkg.to_string(),
                make_lib_package(
                    symbols,
                    TargetTimings {
                        wall_time: Duration::from_secs_f64(wall / 1000.0),
                        event_times_ms,
                    },
                ),
            );
        }

        // Add 1 test target with profiling data to package "a".
        let mut test_symbols = HashMap::new();
        test_symbols.insert("test_fn".to_string(), make_symbol(60.0));

        let mut test_event_times = HashMap::new();
        test_event_times
            .insert("metadata_decode_entry_generics_of".to_string(), 15.0);
        test_event_times.insert("typeck".to_string(), 5.0);

        packages.get_mut("a").unwrap().targets.insert(
            "test".to_string(),
            tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time: Duration::from_secs_f64(250.0 / 1000.0),
                    event_times_ms: test_event_times,
                },
                root: Module {
                    symbols: test_symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            },
        );

        let graph = SymbolGraph { packages };
        let result = fit(
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
        // All targets should appear (6: 5 lib + 1 test).
        assert_eq!(result.target_count, 6);
    }

    // =====================================================================
    // build_cost_model tests
    // =====================================================================

    #[test]
    fn test_build_cost_model_basic() {
        // Build a SymbolGraph with 5 packages, fit model, then extract
        // CostModel and verify the coefficients match.
        //
        // Data (wall = 3*attr + 2*meta + 1*other):
        let graph = make_profiled_graph(&[
            ("a", 100.0, 20.0, 10.0, 350.0),
            ("b", 50.0, 40.0, 20.0, 250.0),
            ("c", 200.0, 10.0, 5.0, 625.0),
            ("d", 30.0, 80.0, 15.0, 265.0),
            ("e", 150.0, 30.0, 25.0, 535.0),
        ]);
        let result = fit(&graph, CostOptions::default());

        assert!(result.model.is_some(), "model should be fitted");

        let cost_model = build_cost_model(&result);
        assert!(cost_model.is_some(), "cost model should be built");

        let cm = cost_model.unwrap();

        // Coefficients should be close to 3, 2, 1 (perfect fit).
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
        // Without profiling data, build_cost_model should return None.
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0));

        let mut packages = HashMap::new();
        packages.insert(
            "my-pkg".to_string(),
            make_lib_package(symbols, TargetTimings::default()),
        );

        let graph = SymbolGraph { packages };
        let result = fit(&graph, CostOptions::default());

        assert!(
            build_cost_model(&result).is_none(),
            "should return None without profiling data",
        );
    }

    // =====================================================================
    // Validation table / report tests
    // =====================================================================

    #[test]
    fn test_validation_table_appears_with_wall_data() {
        // Validation table is emitted when wall-clock data is present.
        let graph = make_profiled_graph(&[("my-pkg", 10.0, 5.0, 3.0, 50.0)]);
        let result = fit(&graph, CostOptions::default());

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

        let mut packages = HashMap::new();
        packages.insert(
            "my-pkg".to_string(),
            make_lib_package(symbols, TargetTimings::default()),
        );

        let graph = SymbolGraph { packages };
        let result = fit(&graph, CostOptions::default());

        let mut output = Vec::new();
        result.write_report(&mut output).unwrap();
        let report = String::from_utf8(output).unwrap();

        assert!(
            !report.contains("Model validation"),
            "report should NOT contain validation table without wall data"
        );
    }

    #[test]
    fn test_validation_includes_model_fit() {
        // 5 targets with wall-clock data. The model fit and per-target
        // predictions should appear in the report.
        let graph = make_profiled_graph(&[
            ("a", 100.0, 20.0, 10.0, 440.0),
            ("b", 50.0, 40.0, 20.0, 340.0),
            ("c", 200.0, 10.0, 5.0, 725.0),
            ("d", 30.0, 80.0, 15.0, 405.0),
            ("e", 150.0, 30.0, 25.0, 625.0),
        ]);
        let result = fit(&graph, CostOptions::default());

        let mut output = Vec::new();
        result.write_report(&mut output).unwrap();
        let report = String::from_utf8(output).unwrap();

        assert!(
            report.contains("Model fit:"),
            "report should contain model fit. Report:\n{report}",
        );
        assert!(report.contains("R²:"), "report should contain R²");
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
    fn test_validation_table_marks_outliers() {
        // 7 normal targets plus one egregious outlier. The model should
        // detect it via MAGSAC's inlier threshold and mark it with `*`.
        let graph = make_profiled_graph(&[
            ("a", 100.0, 20.0, 10.0, 350.0),
            ("b", 50.0, 40.0, 20.0, 250.0),
            ("c", 200.0, 10.0, 5.0, 625.0),
            ("d", 30.0, 80.0, 15.0, 265.0),
            ("e", 150.0, 30.0, 25.0, 535.0),
            ("f", 80.0, 50.0, 30.0, 370.0),
            ("g", 120.0, 15.0, 10.0, 400.0),
            ("outlier", 10.0, 5.0, 2.0, 5000.0), // model predicts ~42
        ]);
        let result = fit(&graph, CostOptions::default());

        let mut output = Vec::new();
        result.write_report(&mut output).unwrap();
        let report = String::from_utf8(output).unwrap();

        // The outlier target should have an asterisk in its row.
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

    #[test]
    fn test_validation_fit_lib_only_report() {
        // Mix of lib and test targets, fitting on libs only.
        // Both should appear in the report.
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

            packages.insert(
                pkg.to_string(),
                make_lib_package(
                    symbols,
                    TargetTimings {
                        wall_time: Duration::from_secs_f64(wall / 1000.0),
                        event_times_ms,
                    },
                ),
            );
        }

        // Add test target with profiling data.
        let mut test_symbols = HashMap::new();
        test_symbols.insert("test_fn".to_string(), make_symbol(60.0));

        let mut test_event_times = HashMap::new();
        test_event_times
            .insert("metadata_decode_entry_generics_of".to_string(), 15.0);
        test_event_times.insert("typeck".to_string(), 5.0);

        packages.get_mut("a").unwrap().targets.insert(
            "test".to_string(),
            tarjanize_schemas::Crate {
                timings: TargetTimings {
                    wall_time: Duration::from_secs_f64(250.0 / 1000.0),
                    event_times_ms: test_event_times,
                },
                root: Module {
                    symbols: test_symbols,
                    submodules: HashMap::new(),
                },
                ..Default::default()
            },
        );

        let graph = SymbolGraph { packages };
        let options = CostOptions {
            fit_libs_only: true,
        };
        let result = fit(&graph, options);

        let mut output = Vec::new();
        result.write_report(&mut output).unwrap();
        let report = String::from_utf8(output).unwrap();

        // Non-lib target should still appear in the table.
        assert!(
            report.contains("a/test"),
            "non-lib target should appear in table. Report:\n{report}",
        );

        // Delta column should be present (model was fitted).
        assert!(
            report.contains("Delta"),
            "Delta column should appear. Report:\n{report}",
        );
    }

    // =====================================================================
    // model_fit tests
    // =====================================================================

    #[test]
    fn test_model_fit_perfect_3var() {
        // wall = 3*attr + 2*meta + 4*other, 5 data points
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
        // wall = 3*attr + 2*meta, 5 data points (2-var)
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
        // All metadata values zero → degenerates to wall = a*attr.
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
        // All attr values zero → degenerates to wall = b*meta.
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
        // Simulated data with a ~ 3.4, b ~ 2.8 and some noise.
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
        assert!((c[0] - 3.4).abs() < 0.1, "Expected a ~ 3.4, got {}", c[0],);
        assert!((c[1] - 2.8).abs() < 0.1, "Expected b ~ 2.8, got {}", c[1],);
    }

    // =====================================================================
    // CostModel tests
    // =====================================================================

    #[test]
    fn test_cost_model_json_roundtrip() {
        // Verify CostModel serializes to JSON and deserializes back.
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

    // =====================================================================
    // format_duration_ms tests
    // =====================================================================

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
}
