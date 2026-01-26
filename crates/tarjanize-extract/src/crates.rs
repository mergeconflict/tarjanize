//! Crate-level extraction for tarjanize.
//!
//! This module provides the bridge between workspace iteration and module
//! extraction. It handles crate-specific concerns like extracting the crate
//! name from Cargo.toml and determining the crate root directory.
//!
//! A crate is conceptually just its root module with some additional metadata.
//! This module extracts that metadata and delegates the actual symbol
//! extraction to the `modules` module.

use std::collections::HashSet;

use ra_ap_hir::{Crate, Semantics};
use ra_ap_ide_db::RootDatabase;
use tarjanize_schemas::{Edge, Module as SchemaModule};

use crate::error::ExtractError;
use crate::file_path;
use crate::modules::extract_module;

/// Extract a crate as a (name, SchemaModule) pair.
///
/// A crate is represented as its root module, which contains all symbols
/// and submodules. This unifies crates and modules in the schema - a crate
/// is just a module with a name from Cargo.toml.
///
/// Returns the crate name separately because the schema uses HashMaps keyed
/// by name rather than storing the name inside the Module struct.
///
/// # Errors
///
/// Returns [`ExtractError`] if:
/// - The crate has no display name ([`ExtractError::is_crate_name_missing`])
/// - The crate root file path cannot be resolved ([`ExtractError::is_file_path_not_found`])
/// - The crate root file has no parent directory ([`ExtractError::is_crate_root_no_parent`])
pub(crate) fn extract_crate(
    sema: &Semantics<'_, RootDatabase>,
    krate: Crate,
    edges: &mut HashSet<Edge>,
) -> Result<(String, SchemaModule), ExtractError> {
    let db = sema.db;

    // All Cargo workspace crates must have names in Cargo.toml. A missing
    // display name indicates a non-Cargo build system or synthetic crate.
    let crate_name = krate
        .display_name(db)
        .ok_or_else(ExtractError::crate_name_missing)?
        .to_string();

    // Get the crate root directory for computing relative file paths.
    // The crate root file is lib.rs or main.rs; its parent is the crate root dir.
    let root_file_id = krate.root_file(db);
    let crate_root_path = file_path(db, root_file_id)?;
    let crate_root = crate_root_path
        .parent()
        .ok_or_else(|| ExtractError::crate_root_no_parent(&crate_name))?;

    // Get the crate's root module and extract it recursively.
    let root_module = krate.root_module(db);
    let (_, module) =
        extract_module(sema, &crate_root, &root_module, &crate_name, edges);

    // Return the crate name from Cargo.toml (not the module name, which may
    // be empty for root modules).
    Ok((crate_name, module))
}
