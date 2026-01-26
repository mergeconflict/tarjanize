//! Symbol graph extraction from Rust workspaces.
//!
//! This crate extracts symbol dependency graphs from Rust workspaces using
//! rust-analyzer's semantic analysis. The extracted graph captures symbols
//! (functions, structs, traits, impl blocks, etc.) and their dependency
//! relationships.
//!
//! ## Usage
//!
//! ```no_run
//! use tarjanize_extract::run;
//!
//! let mut output = Vec::new();
//! run("path/to/workspace", &mut output).unwrap();
//! ```
//!
//! ## Re-exports
//!
//! This crate re-exports types from `tarjanize_schemas` for convenience.
//! See [`tarjanize_schemas`] for schema documentation.

mod crates;
mod dependencies;
mod error;
mod modules;
mod workspaces;

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use crates::extract_crate;
use ra_ap_base_db::{FileId, SourceDatabase};
use ra_ap_hir::{Crate, Semantics, attach_db};
use ra_ap_ide_db::RootDatabase;
use rayon::prelude::*;
// Re-export schema types for convenience.
#[doc(inline)]
pub use tarjanize_schemas::{Edge, Module, Symbol, SymbolGraph, SymbolKind};
use tracing::{instrument, warn};

#[doc(inline)]
pub use crate::error::ExtractError;
pub(crate) use crate::workspaces::load_workspace;

/// Look up the filesystem path for a file ID.
///
/// Uses the database's source roots, which are populated by both
/// `load_workspace` (production) and test fixtures.
pub(crate) fn file_path(
    db: &RootDatabase,
    file_id: FileId,
) -> Result<ra_ap_base_db::VfsPath, ExtractError> {
    let source_root_id = db.file_source_root(file_id).source_root_id(db);
    let source_root = db.source_root(source_root_id).source_root(db);
    source_root
        .path_for_file(&file_id)
        .cloned()
        .ok_or_else(|| ExtractError::file_path_not_found(file_id))
}

/// Extracts a complete symbol graph from a rust-analyzer database.
///
/// Walks all workspace crates and extracts their symbols and dependency
/// relationships.
#[instrument(skip(db))]
pub(crate) fn extract_symbol_graph(db: RootDatabase) -> SymbolGraph {
    // Filter to workspace members. Crate::all() returns ALL crates
    // (workspace + all transitive dependencies), but we only analyze
    // workspace members since those are what we might split.
    let workspace_crates: Vec<_> = Crate::all(&db)
        .into_iter()
        .filter(|krate| krate.origin(&db).is_local())
        .collect();

    let (crates, edges) = workspace_crates
        // Iterate over all workspace crates in parallel...
        .par_iter()
        .fold_with(
            // Start each thread with a fresh (db, crates, edges) accumulator.
            // Note that cloning the db is cheap (it's effectively a handle).
            (db, HashMap::new(), HashSet::new()),
            // For each crate ...
            |(db, mut crates, mut edges), &krate| {
                // Attach the database for this thread (required for type inference).
                attach_db(&db, || {
                    let sema = Semantics::new(&db);
                    // Extract the crate and accumulate the results.
                    match extract_crate(&sema, krate) {
                        Ok((name, module, crate_edges)) => {
                            crates.insert(name, module);
                            edges.extend(crate_edges);
                        }
                        Err(e) => {
                            match krate.display_name(&db) {
                                Some(n) => warn!(crate_name = %n, error = %e, "skipping crate"),
                                None => warn!(error = %e, "skipping unnamed crate"),
                            }
                        }
                    }
                });
                (db, crates, edges)
            },
        )
        // Discard the db from the accumulator tuples.
        .map(|(_, crates, edges)| (crates, edges))
        // Reduce all thread-local accumulators into a single (crates, edges) tuple.
        .reduce(
            || (HashMap::new(), HashSet::new()),
            |(mut crates, mut edges), (more_crates, more_edges)| {
                crates.extend(more_crates);
                edges.extend(more_edges);
                (crates, edges)
            },
        );

    SymbolGraph { crates, edges }
}

/// Run the extract operation.
///
/// Loads a workspace, extracts the symbol graph, and writes it to the
/// provided output writer as JSON.
///
/// # Errors
///
/// Returns [`ExtractError`] if:
/// - The workspace cannot be loaded ([`ExtractError::is_workspace_load`])
/// - Writing to the output fails ([`ExtractError::is_io`])
/// - JSON serialization fails ([`ExtractError::is_serialization`])
///
/// # Example
///
/// ```no_run
/// use std::io::stdout;
/// use tarjanize_extract::run;
///
/// let mut out = stdout().lock();
/// run("path/to/workspace", &mut out).unwrap();
/// ```
pub fn run(
    workspace_path: impl AsRef<Path> + std::fmt::Debug,
    output: &mut dyn Write,
) -> Result<(), ExtractError> {
    // Step 1: load the workspace into a rust-analyzer database.
    let db = load_workspace(workspace_path)?;

    // Step 2: extract the symbol graph from the db.
    let symbol_graph = extract_symbol_graph(db);

    // Step 3: stream to output.
    serde_json::to_writer_pretty(&mut *output, &symbol_graph)?;
    writeln!(output)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// run() should read a real Cargo project and produce reasonable output.
    #[test]
    fn test_run() {
        let fixture = PathBuf::from("tests/fixtures/minimal_crate");
        let mut output = Vec::new();

        run(&fixture, &mut output).expect("run() should succeed");

        let graph: SymbolGraph = serde_json::from_slice(&output)
            .expect("run() should output valid JSON");
        assert!(
            graph.crates.contains_key("minimal_crate"),
            "JSON output should contain minimal_crate"
        );
    }
}
