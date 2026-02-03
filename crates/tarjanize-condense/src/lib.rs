//! Graph condensation for dependency analysis.
//!
//! This crate computes strongly connected components (SCCs) from a symbol
//! graph and produces a condensed DAG. This is Phase 2 of the tarjanize
//! pipeline.
//!
//! ## Algorithm
//!
//! 1. Build a directed graph with symbols as nodes, dependencies as edges
//! 2. Run petgraph's `condensation` to find SCCs and build the condensed DAG
//! 3. Collect impl anchors for orphan rule enforcement in Phase 3
//!
//! ## Orphan Rule
//!
//! Rust's orphan rule requires trait impls to be in the same crate as either
//! the trait or the self type. Rather than forcing impls and their anchors
//! into the same SCC (which would reduce Phase 3's flexibility), we track
//! impl anchors separately. Each SCC's `anchor_sets` field lists the SCCs containing
//! its impl blocks' types and traits. Phase 3 uses this to ensure valid crate
//! groupings.
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
//!
//! ## Re-exports
//!
//! This crate re-exports types from `tarjanize_schemas` for convenience.

mod error;
mod scc;

use std::io::{Read, Write};

use tarjanize_schemas::SymbolGraph;
// Re-export schema types for convenience.
#[doc(inline)]
pub use tarjanize_schemas::{AnchorSet, CondensedGraph, Scc};
use tracing::debug_span;

#[doc(inline)]
pub use crate::error::CondenseError;
use crate::error::CondenseErrorKind;
use crate::scc::compute_condensed_graph;

/// Run the condense operation.
///
/// Reads a `SymbolGraph` from the input, condenses it, and writes the
/// `CondensedGraph` to the output as JSON.
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
    output: &mut dyn Write,
) -> Result<(), CondenseError> {
    let _span = debug_span!("run").entered();

    // Step 1: Read and parse input JSON.
    let mut json = String::new();
    input.read_to_string(&mut json)?;
    let symbol_graph: SymbolGraph =
        serde_json::from_str(&json).map_err(|e| {
            CondenseError::new(CondenseErrorKind::Deserialization(e))
        })?;

    // Step 2: Condense the graph.
    let condensed = compute_condensed_graph(&symbol_graph);

    // Step 3: Write output JSON.
    serde_json::to_writer_pretty(&mut *output, &condensed)
        .map_err(|e| CondenseError::new(CondenseErrorKind::Serialization(e)))?;
    writeln!(output)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tarjanize_schemas::{Module, Symbol, SymbolKind, Visibility};

    use super::*;

    /// Helper to create a simple symbol for testing.
    fn make_symbol(deps: &[&str]) -> Symbol {
        Symbol {
            file: "test.rs".to_string(),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: deps.iter().map(|&s| s.to_string()).collect(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        }
    }

    #[test]
    fn test_run_roundtrip() {
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(&[]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            tarjanize_schemas::Crate {
                linking_ms: 0.0,
                metadata_ms: 0.0,
                root: Module {
                    symbols,
                    submodules: HashMap::new(),
                },
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let input_json = serde_json::to_string(&symbol_graph).unwrap();

        let mut output = Vec::new();
        run(input_json.as_bytes(), &mut output).unwrap();

        // Verify output is valid JSON.
        let condensed: CondensedGraph = serde_json::from_slice(&output)
            .expect("output should be valid JSON");

        assert_eq!(condensed.sccs.len(), 1);
        assert_eq!(condensed.sccs[0].symbols.len(), 1);
        assert!(condensed.sccs[0].symbols.contains("my_crate::foo"));
    }

    #[test]
    fn test_run_invalid_json() {
        let mut output = Vec::new();
        let result = run("not valid json".as_bytes(), &mut output);

        assert!(result.is_err());
        assert!(result.unwrap_err().is_deserialization());
    }
}
