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

use tarjanize_schemas::{Edge, Module as SchemaModule};

use crate::error::ExtractError;
use crate::file_path;
use crate::modules::extract_module;

/// Extract a crate as a SchemaModule (the crate root module).
///
/// A crate is represented as its root module, which contains all symbols
/// and submodules. This unifies crates and modules in the schema - a crate
/// is just a module with a name from Cargo.toml.
///
/// # Errors
///
/// Returns an error if:
/// - The crate root file path cannot be resolved
/// - The crate root file has no parent directory
pub fn extract_crate(
    sema: &Semantics<'_, RootDatabase>,
    krate: Crate,
    edges: &mut HashSet<Edge>,
) -> Result<SchemaModule, ExtractError> {
    let db = sema.db;

    let crate_name = krate
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    // Get the crate root directory for computing relative file paths.
    // The crate root file is lib.rs or main.rs; its parent is the crate root dir.
    let root_file_id = krate.root_file(db);
    let crate_root_path = file_path(db, root_file_id)?;
    let crate_root = crate_root_path
        .parent()
        .ok_or_else(|| ExtractError::crate_root_no_parent(&crate_name))?;

    // Get the crate's root module and extract it recursively.
    let root_module = krate.root_module(db);
    Ok(extract_module(
        sema,
        &crate_root,
        &root_module,
        &crate_name,
        edges,
    ))
}
