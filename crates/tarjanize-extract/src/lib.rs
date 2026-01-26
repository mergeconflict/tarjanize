//! Symbol graph extraction from Rust workspaces.
//!
//! This crate provides functionality to extract symbol dependency graphs from
//! Rust workspaces using rust-analyzer's semantic analysis. The extracted graph
//! captures symbols (functions, structs, traits, impl blocks, etc.) and their
//! dependency relationships.
//!
//! ## Usage
//!
//! ```ignore
//! use tarjanize_extract::{load_workspace, extract_symbol_graph, run};
//!
//! // Option 1: Full pipeline - load workspace, extract graph, write JSON
//! let mut output = Vec::new();
//! run("path/to/workspace", &mut output)?;
//!
//! // Option 2: Individual steps for more control
//! let (db, _vfs) = load_workspace("path/to/workspace")?;
//! let graph = extract_symbol_graph(&db, "my_workspace");
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

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use ra_ap_base_db::{FileId, SourceDatabase};
use ra_ap_hir::{Crate, Semantics, attach_db};
use ra_ap_ide_db::RootDatabase;
use tracing::{info, warn};

pub use crate::error::ExtractError;
pub use crate::workspaces::load_workspace;

// Re-export schema types for convenience.
#[doc(inline)]
pub use tarjanize_schemas::{Edge, Module, Symbol, SymbolGraph, SymbolKind};

use crates::extract_crate;

/// Look up the filesystem path for a file ID.
///
/// Uses the database's source roots, which are populated by both
/// `load_workspace` (production) and test fixtures.
///
/// # Errors
///
/// Returns an error if the file ID is not found in any source root.
/// This is unexpected for valid workspace files and may indicate a bug.
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
///
/// The workspace_name parameter is provided by the caller (typically derived
/// from the root directory or Cargo.toml) since rust-analyzer doesn't have
/// a single "workspace name" concept.
///
/// # Example
///
/// ```ignore
/// let (db, _vfs) = load_workspace(".")?;
/// let graph = extract_symbol_graph(&db, "my_project");
///
/// println!("Found {} crates", graph.crates.len());
/// println!("Found {} dependency edges", graph.edges.len());
///
/// // Serialize to JSON for downstream processing
/// let json = serde_json::to_string_pretty(&graph)?;
/// ```
pub fn extract_symbol_graph(
    db: &RootDatabase,
    workspace_name: &str,
) -> SymbolGraph {
    // The new ra_ap_hir_ty solver requires the database to be attached to
    // a thread-local for type inference operations. We wrap all semantic
    // analysis in attach_db to ensure the database is available.
    attach_db(db, || {
        let sema = Semantics::new(db);

        // Filter to workspace members. Crate::all() returns ALL crates
        // (workspace + all transitive dependencies), but we only analyze
        // workspace members since those are what we might split.
        let workspace_crates: Vec<_> = Crate::all(db)
            .into_iter()
            .filter(|krate| krate.origin(db).is_local())
            .collect();

        let mut edges = HashSet::new();
        let mut crates = Vec::new();

        for krate in workspace_crates {
            match extract_crate(&sema, krate, &mut edges) {
                Ok(crate_module) => crates.push(crate_module),
                Err(e) => {
                    let name = krate
                        .display_name(db)
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "(unnamed)".to_string());
                    tracing::warn!(crate_name = %name, error = %e, "skipping crate");
                }
            }
        }

        SymbolGraph {
            workspace_name: workspace_name.to_string(),
            crates,
            edges,
        }
    })
}

/// Run the extract operation.
///
/// Loads a workspace, extracts the symbol graph, and writes it to the
/// provided output writer as JSON.
///
/// # Errors
///
/// Returns [`ExtractError`] if:
/// - The workspace cannot be loaded
/// - Writing to the output fails
/// - JSON serialization fails
///
/// # Example
///
/// ```ignore
/// use std::io::stdout;
/// use tarjanize_extract::run;
///
/// let mut out = stdout().lock();
/// run("path/to/workspace", &mut out)?;
/// ```
pub fn run(
    workspace_path: &str,
    output: &mut dyn Write,
) -> Result<(), ExtractError> {
    info!(path = %workspace_path, "workspace.loading");
    let (db, _vfs) = load_workspace(workspace_path)?;

    // Derive workspace name from Cargo.toml, falling back to directory name.
    let workspace_name = read_workspace_name(workspace_path).unwrap_or_else(|| {
        let name = Path::new(workspace_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("workspace")
            .to_string();
        warn!(
            fallback = %name,
            "Could not read workspace name from Cargo.toml, using directory name"
        );
        name
    });

    // Extract the symbol graph from the workspace.
    let symbol_graph = extract_symbol_graph(&db, &workspace_name);

    info!(
        crates = symbol_graph.crates.len(),
        edges = symbol_graph.edges.len(),
        "extraction.complete"
    );

    // Serialize and write to output.
    let json = serde_json::to_string_pretty(&symbol_graph)?;
    writeln!(output, "{}", json)?;

    Ok(())
}

/// Read the workspace/package name from Cargo.toml.
///
/// Tries to read [package].name first, then falls back to parsing the
/// [workspace] section if present. Returns None if Cargo.toml doesn't
/// exist or doesn't contain a usable name.
fn read_workspace_name(workspace_path: &str) -> Option<String> {
    let cargo_toml_path = Path::new(workspace_path).join("Cargo.toml");
    let contents = std::fs::read_to_string(&cargo_toml_path).ok()?;
    let toml: toml::Table = contents.parse().ok()?;

    // Try [package].name first (most common case: single-crate workspace)
    if let Some(package) = toml.get("package").and_then(|p| p.as_table())
        && let Some(name) = package.get("name").and_then(|n| n.as_str())
    {
        return Some(name.to_string());
    }

    // For multi-crate workspaces, there's no standard "name" field.
    // We could parse [workspace].members but that gives crate names, not
    // a workspace name. Just return None and let caller use directory name.
    None
}
