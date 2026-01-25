//! Crate-level extraction for tarjanize.
//!
//! This module provides the bridge between workspace iteration and module
//! extraction. It handles crate-specific concerns like extracting the crate
//! name from Cargo.toml and determining the crate root directory.
//!
//! The key insight is that a crate is conceptually just its root module with
//! some additional metadata. This module extracts that metadata and delegates
//! the actual symbol extraction to the `modules` module.

use std::collections::HashSet;

use ra_ap_hir::{Crate, Semantics};
use ra_ap_ide_db::RootDatabase;
use ra_ap_paths::AbsPathBuf;

use crate::extract::FilePathResolver;
use crate::modules::extract_module;
use crate::schemas::{Edge, Module as SchemaModule};

/// Extract a crate as a SchemaModule (the crate root module).
///
/// A crate is represented as its root module, which contains all symbols
/// and submodules. This unifies crates and modules in the schema - a crate
/// is just a module with a name from Cargo.toml.
pub fn extract_crate<F: FilePathResolver>(
    sema: &Semantics<'_, RootDatabase>,
    file_resolver: &F,
    krate: Crate,
    edges: &mut HashSet<Edge>,
) -> SchemaModule {
    let db = sema.db;

    let crate_name = krate
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    // Get the crate root directory for computing relative file paths.
    // The crate root file is lib.rs or main.rs; its parent is the crate root dir.
    let file_id = krate.root_file(db);
    let crate_root: Option<AbsPathBuf> = file_resolver
        .file_path(file_id)
        .as_path()
        .and_then(|p| p.parent())
        .map(|p| p.to_owned());

    // Get the crate's root module and extract it recursively.
    let root_module = krate.root_module(db);
    extract_module(
        sema,
        file_resolver,
        crate_root.as_deref(),
        &root_module,
        &crate_name,
        edges,
    )
}
