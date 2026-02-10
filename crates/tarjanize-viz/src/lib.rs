//! Interactive HTML build schedule visualization.
//!
//! Generates a self-contained HTML file with a canvas-based Gantt chart
//! showing the build schedule for a Rust workspace. Supports both original
//! extracted graphs (real build times) and condensed graphs (hypothetical
//! post-split times).
//!
//! ## Pipeline
//!
//! ```text
//! SymbolGraph + CostModel → TargetGraph → ScheduleData → HTML
//! ```
//!
//! The `SymbolGraph` provides the dependency structure and per-symbol costs.
//! The `CostModel` predicts compilation wall time from three predictors
//! (attributed, metadata, other). When no `CostModel` is available, falls
//! back to per-symbol cost sums.

pub mod data;
mod error;
mod html;
mod schedule;

use std::io::{Read, Write};

pub use error::VizError;
use tarjanize_schemas::{CostModel, Module, SymbolGraph, sum_event_times};

use crate::schedule::{TargetGraph, compute_schedule};

/// Generates an interactive HTML build schedule visualization.
///
/// Reads a `SymbolGraph` from `input`, computes the schedule using the
/// `cost_model` for wall time predictions, and writes a self-contained
/// HTML file to `output`.
///
/// When `cost_model` is `None`, falls back to per-symbol cost sums
/// (the "effective" timing from the symbol graph).
pub fn run(
    mut input: impl Read,
    cost_model: Option<&CostModel>,
    output: impl Write,
) -> Result<(), VizError> {
    let mut json = String::new();
    input.read_to_string(&mut json)?;

    let symbol_graph: SymbolGraph =
        serde_json::from_str(&json).map_err(VizError::deserialize)?;

    let target_graph = build_target_graph(&symbol_graph, cost_model);
    let schedule_data = compute_schedule(&target_graph);

    html::generate(&schedule_data, output)
}

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
fn build_target_graph(
    symbol_graph: &SymbolGraph,
    cost_model: Option<&CostModel>,
) -> TargetGraph {
    use std::collections::HashMap;

    use indexmap::IndexSet;
    use petgraph::graph::DiGraph;

    let mut names: IndexSet<String> = IndexSet::new();
    let mut costs: Vec<f64> = Vec::new();
    let mut symbol_counts: Vec<usize> = Vec::new();

    // First pass: register targets, compute costs and symbol counts.
    // We need a temporary map because test target augmentation happens
    // after the initial pass.
    let mut cost_map: HashMap<String, f64> = HashMap::new();
    let mut sym_count_map: HashMap<String, usize> = HashMap::new();

    for (package_name, package) in &symbol_graph.packages {
        for (target_key, target_data) in &package.targets {
            let target_id = format!("{package_name}/{target_key}");
            names.insert(target_id.clone());

            let syms = count_symbols(&target_data.root);
            sym_count_map.insert(target_id.clone(), syms);

            // Compute cost: use CostModel if available, else effective.
            let attr = collect_frontend_cost(&target_data.root);
            let cost = if let Some(model) = cost_model {
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
                model.predict(attr, meta, other)
            } else if target_data.timings.wall_time_ms > 0.0 {
                // Use wall-clock when profiled.
                target_data.timings.wall_time_ms
            } else {
                // Fall back to per-symbol sum.
                attr
            };

            cost_map.insert(target_id, cost);
        }
    }

    // Augment test targets with lib costs when the test has no wall-clock
    // profiling data and no cost model is in use. When a cost model IS
    // provided, predictions already account for the full compilation.
    if cost_model.is_none() {
        for (package_name, package) in &symbol_graph.packages {
            if !package.targets.contains_key("test") {
                continue;
            }
            let test_id = format!("{package_name}/test");
            let lib_id = format!("{package_name}/lib");

            let test_has_wall = package
                .targets
                .get("test")
                .is_some_and(|t| t.timings.wall_time_ms > 0.0);

            // Augment when test lacks wall-clock data.
            if !test_has_wall
                && let Some(&lib_cost) = cost_map.get(&lib_id)
                && let Some(test_cost) = cost_map.get_mut(&test_id)
            {
                *test_cost += lib_cost;
            }
        }
    }

    // Collect vectors in names order.
    for name in &names {
        costs.push(*cost_map.get(name).unwrap_or(&0.0));
        symbol_counts.push(*sym_count_map.get(name).unwrap_or(&0));
    }

    // Build dependency graph.
    let mut graph = DiGraph::<usize, ()>::with_capacity(names.len(), 0);
    for _ in 0..names.len() {
        graph.add_node(0);
    }

    for (package_name, package) in &symbol_graph.packages {
        for (target_key, target_data) in &package.targets {
            let target_id = format!("{package_name}/{target_key}");
            let Some(target_idx) = names.get_index_of(&target_id) else {
                continue;
            };

            for dep_target in &target_data.dependencies {
                if dep_target == &target_id {
                    continue;
                }
                if let Some(dep_idx) = names.get_index_of(dep_target) {
                    graph.add_edge(
                        petgraph::graph::NodeIndex::new(dep_idx),
                        petgraph::graph::NodeIndex::new(target_idx),
                        (),
                    );
                }
            }
        }
    }

    TargetGraph {
        names,
        costs,
        symbol_counts,
        graph,
    }
}

/// Recursively sums all symbol `event_times_ms` in a module tree.
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

/// Recursively counts symbols in a module tree.
fn count_symbols(module: &Module) -> usize {
    let mut count = module.symbols.len();
    for submodule in module.submodules.values() {
        count += count_symbols(submodule);
    }
    count
}
