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
//! run(input, &mut output).unwrap();
//! ```

mod error;
mod scc;

use std::io::{Read, Write};

use tarjanize_schemas::SymbolGraph;
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
/// run(input, &mut out).unwrap();
/// ```
pub fn run(
    mut input: impl Read,
    mut output: impl Write,
) -> Result<(), CondenseError> {
    let _span = debug_span!("run").entered();

    // Step 1: Read and parse input JSON.
    let mut json = String::new();
    input.read_to_string(&mut json)?;
    let symbol_graph: SymbolGraph =
        serde_json::from_str(&json).map_err(|e| {
            CondenseError::new(CondenseErrorKind::Deserialization(e))
        })?;

    // Step 2: Condense and partition the graph.
    let optimized = condense_and_partition(&symbol_graph);

    // Step 3: Write output JSON.
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
    fn make_crate(
        symbols: HashMap<String, Symbol>,
    ) -> tarjanize_schemas::Crate {
        tarjanize_schemas::Crate {
            root: Module {
                symbols,
                submodules: HashMap::new(),
            },
            ..Default::default()
        }
    }

    /// Helper to wrap a Crate into a Package with a "lib" target.
    fn make_package(crate_data: tarjanize_schemas::Crate) -> Package {
        let mut targets = HashMap::new();
        targets.insert("lib".to_string(), crate_data);
        Package { targets }
    }

    /// Helper to create a `SymbolGraph` from a map of crate names to Crates.
    fn make_graph(
        crates: HashMap<String, tarjanize_schemas::Crate>,
    ) -> SymbolGraph {
        let packages = crates
            .into_iter()
            .map(|(name, c)| (name, make_package(c)))
            .collect();
        SymbolGraph { packages }
    }

    #[test]
    fn test_run_roundtrip() {
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert("my_crate".to_string(), make_crate(symbols));

        let symbol_graph = make_graph(crates);
        let input_json = serde_json::to_string(&symbol_graph).unwrap();

        let mut output = Vec::new();
        run(input_json.as_bytes(), &mut output).unwrap();

        // Verify output is valid SymbolGraph JSON.
        let optimized: SymbolGraph = serde_json::from_slice(&output)
            .expect("output should be valid SymbolGraph JSON");

        // For a single symbol with no dependencies, it stays in its own crate.
        assert_eq!(optimized.packages.len(), 1);
    }

    #[test]
    fn test_run_invalid_json() {
        let mut output = Vec::new();
        let result = run("not valid json".as_bytes(), &mut output);

        assert!(result.is_err());
        assert!(result.unwrap_err().is_deserialization());
    }
}
