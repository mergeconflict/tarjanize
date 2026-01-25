//! Extraction of symbol graphs from rust-analyzer's semantic model.
//!
//! This module is the entry point for Phase 1 of tarjanize. It coordinates
//! the extraction process by iterating workspace crates and assembling the
//! final SymbolGraph.
//!
//! The actual extraction logic lives in:
//! - `crates.rs`: Crate-level extraction
//! - `modules.rs`: Module and symbol extraction
//! - `dependencies.rs`: Dependency analysis

use std::collections::HashSet;

#[cfg(test)]
use ra_ap_base_db::SourceDatabase;
use ra_ap_base_db::{FileId, VfsPath};
use ra_ap_hir::{Crate, Semantics, attach_db};
use ra_ap_ide_db::RootDatabase;
use ra_ap_vfs::Vfs;

use crate::crates::extract_crate;
use crate::schemas::SymbolGraph;

/// Trait for resolving file IDs to file paths.
///
/// This abstraction allows the extraction code to work with both:
/// - Production code using `Vfs` from `load_workspace`
/// - Test code using fixtures where paths are stored in the database
pub trait FilePathResolver {
    /// Look up the path for a given file ID.
    fn file_path(&self, file_id: FileId) -> VfsPath;
}

/// Implementation for the production VFS.
impl FilePathResolver for Vfs {
    fn file_path(&self, file_id: FileId) -> VfsPath {
        Vfs::file_path(self, file_id).clone()
    }
}

/// File path resolver that queries the database's source roots.
///
/// Used for tests with `ra_ap_test_fixture` where file paths are stored
/// in the database rather than a separate VFS.
#[cfg(test)]
pub struct DatabaseFileResolver<'db> {
    db: &'db RootDatabase,
}

#[cfg(test)]
impl<'db> DatabaseFileResolver<'db> {
    /// Create a new resolver from a database.
    pub fn new(db: &'db RootDatabase) -> Self {
        Self { db }
    }
}

#[cfg(test)]
impl FilePathResolver for DatabaseFileResolver<'_> {
    fn file_path(&self, file_id: FileId) -> VfsPath {
        // Get the source root for this file, then look up the path in its FileSet.
        let source_root_id =
            self.db.file_source_root(file_id).source_root_id(self.db);
        let source_root =
            self.db.source_root(source_root_id).source_root(self.db);
        source_root
            .path_for_file(&file_id)
            .cloned()
            .unwrap_or_else(|| VfsPath::new_virtual_path(String::new()))
    }
}

/// Extracts a complete symbol graph from a rust-analyzer database.
///
/// This is the main entry point for Phase 1 of tarjanize. It walks all
/// workspace crates and extracts their symbols and dependency relationships.
///
/// The workspace_name parameter is provided by the caller (typically derived
/// from the root directory or Cargo.toml) since rust-analyzer doesn't have
/// a single "workspace name" concept.
///
/// The `file_resolver` is used to convert FileIds to file paths. Pass `&vfs`
/// for production code or `DatabaseFileResolver::new(&db)` for tests.
///
/// # Example
///
/// ```ignore
/// let (db, vfs) = load_workspace(".")?;
/// let graph = extract_symbol_graph(&db, &vfs, "my_project");
///
/// println!("Found {} crates", graph.crates.len());
/// println!("Found {} dependency edges", graph.edges.len());
///
/// // Serialize to JSON for downstream processing
/// let json = serde_json::to_string_pretty(&graph)?;
/// ```
pub fn extract_symbol_graph<F: FilePathResolver>(
    db: &RootDatabase,
    file_resolver: &F,
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
            let crate_module =
                extract_crate(&sema, file_resolver, krate, &mut edges);
            crates.push(crate_module);
        }

        SymbolGraph {
            workspace_name: workspace_name.to_string(),
            crates,
            edges,
        }
    })
}
