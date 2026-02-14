//! Graph condensation and partitioning for dependency analysis.
//!
//! This crate computes strongly connected components (SCCs) from a symbol
//! graph, merges them into optimal crate groupings using union-find, and
//! produces an optimized `SymbolGraph`. This is Phase 2 of the tarjanize
//! pipeline.
//!
//! ## Algorithm
//!
//! 1. Build a directed graph with symbols as nodes, dependencies as edges
//! 2. Run petgraph's `condensation` to find SCCs
//! 3. Use union-find to merge SCCs into optimal crate groupings
//! 4. Fix anchor constraints (orphan rule) via global hitting set
//! 5. Build output `SymbolGraph` with new crate structure
//!
//! ## Orphan Rule
//!
//! Rust's orphan rule requires trait impls to be in the same crate as either
//! the trait or the self type. After union-find merging, some impl blocks may
//! end up in crates without any of their anchors. The hitting set algorithm
//! finds minimal merges to satisfy all anchor constraints.
//!
//! ## Usage
//!
//! ```no_run
//! use tarjanize_condense::run;
//!
//! let input = std::io::stdin().lock();
//! let mut output = Vec::new();
//! run(input, &mut output, None).unwrap();
//! ```

mod error;
mod rewrite;
mod scc;

use std::io::{Read, Write};

use tarjanize_schemas::{CostModel, SymbolGraph};
use tracing::debug_span;

#[doc(inline)]
pub use crate::error::CondenseError;
use crate::error::CondenseErrorKind;
use crate::scc::condense_and_partition;

/// Run the condense and partition operation.
///
/// Reads a `SymbolGraph` from the input, computes SCCs, merges them into
/// optimal crate groupings, and writes the optimized `SymbolGraph` to output.
///
/// When `cost_model` is provided, synthetic crate wall times are predicted
/// using dep-count-based sub-models instead of the internal max-constituent
/// heuristic. The model should be produced by `tarjanize cost --output-model`.
///
/// # Errors
///
/// Returns [`CondenseError`] if:
/// - Reading from input fails ([`CondenseError::is_io`])
/// - The input is not valid JSON ([`CondenseError::is_deserialization`])
/// - Writing to output fails ([`CondenseError::is_io`])
/// - JSON serialization fails ([`CondenseError::is_serialization`])
///
/// # Example
///
/// ```no_run
/// use std::io::stdout;
/// use tarjanize_condense::run;
///
/// let input = std::fs::File::open("symbol_graph.json").unwrap();
/// let mut out = stdout().lock();
/// run(input, &mut out, None).unwrap();
/// ```
///
/// Why: provides a single, stream-oriented entry point for the CLI pipeline.
pub fn run(
    mut input: impl Read,
    mut output: impl Write,
    cost_model: Option<&CostModel>,
) -> Result<(), CondenseError> {
    let _span = debug_span!("run").entered();

    // Step 1: Read and parse input JSON.
    // Why: the condense pipeline operates on a fully materialized SymbolGraph.
    let mut json = String::new();
    input.read_to_string(&mut json)?;
    let symbol_graph: SymbolGraph =
        serde_json::from_str(&json).map_err(|e| {
            CondenseError::new(CondenseErrorKind::Deserialization(e))
        })?;

    // Step 2: Condense and partition the graph.
    // Why: SCCs and anchor constraints must be resolved before rewriting paths.
    let optimized = condense_and_partition(&symbol_graph, cost_model);

    // Step 3: Write output JSON.
    // Why: downstream stages expect a stable, pretty-printed schema format.
    serde_json::to_writer_pretty(&mut output, &optimized)
        .map_err(|e| CondenseError::new(CondenseErrorKind::Serialization(e)))?;
    writeln!(output)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tarjanize_schemas::{Module, Package, Symbol, SymbolKind, Visibility};

    use super::*;

    /// Helper to create a simple symbol for testing.
    ///
    /// Why: keeps fixture construction concise across multiple tests.
    fn make_symbol(deps: &[&str]) -> Symbol {
        Symbol {
            file: "test.rs".to_string(),
            event_times_ms: HashMap::new(),
            dependencies: deps.iter().map(|&s| s.to_string()).collect(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        }
    }

    /// Helper to create a crate with default overhead for testing.
    ///
    /// Why: avoids repeating boilerplate module setup in each test.
    fn make_crate(
        symbols: HashMap<String, Symbol>,
    ) -> tarjanize_schemas::Target {
        tarjanize_schemas::Target {
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        }
    }

    /// Helper to wrap a Target into a Package with a "lib" target.
    ///
    /// Why: tests rely on the standard "pkg/lib" target naming.
    fn make_package(crate_data: tarjanize_schemas::Target) -> Package {
        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), crate_data);
        Package { targets }
    }

    /// Helper to create a `SymbolGraph` from a map of crate names to Crates.
    ///
    /// Why: consolidates test graph assembly for readability.
    fn make_graph(
        crates: HashMap<String, tarjanize_schemas::Target>,
    ) -> SymbolGraph {
        let packages = crates
            .into_iter()
            .map(|(name, c)| (name, make_package(c)))
            .collect();
        SymbolGraph { packages }
    }

    /// Roundtrip smoke test: `run` produces valid `SymbolGraph` JSON.
    ///
    /// Why: verifies the CLI path can parse, condense, and serialize.
    #[test]
    fn test_run_roundtrip() {
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let input_json = serde_json::to_string(&symbol_graph).unwrap();

        let mut output = Vec::new();
        run(input_json.as_bytes(), &mut output, None).unwrap();

        // Verify output is valid SymbolGraph JSON.
        let optimized: SymbolGraph = serde_json::from_slice(&output)
            .expect("output should be valid SymbolGraph JSON");

        // For a single symbol with no dependencies, it stays in its own crate.
        assert_eq!(optimized.packages.len(), 1);
    }

    /// Invalid JSON should yield a deserialization error classification.
    ///
    /// Why: callers rely on error classification for user-facing messaging.
    #[test]
    fn test_run_invalid_json() {
        let mut output = Vec::new();
        let result = run("not valid json".as_bytes(), &mut output, None);

        assert!(result.is_err());
        assert!(result.unwrap_err().is_deserialization());
    }
}
