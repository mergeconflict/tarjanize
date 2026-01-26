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
use tarjanize_schemas::{Edge, Module};

use crate::error::ExtractError;
use crate::file_path;
use crate::modules::extract_module;

/// Extract a crate as a (name, module, edges) tuple.
///
/// A crate is represented as its root module, which contains all symbols
/// and submodules. This unifies crates and modules in the schema - a crate
/// is just a module with a name from Cargo.toml.
///
/// Returns edges separately to support parallel extraction - each crate's
/// edges are collected independently and merged by the caller.
pub(crate) fn extract_crate(
    sema: &Semantics<'_, RootDatabase>,
    krate: Crate,
) -> Result<(String, Module, HashSet<Edge>), ExtractError> {
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

    // Collect edges for this crate into a local set.
    let mut edges = HashSet::new();

    // Get the crate's root module and extract it recursively.
    let root_module = krate.root_module(db);
    let (_, module) = extract_module(
        sema,
        &crate_root,
        &root_module,
        &crate_name,
        &mut edges,
    );

    Ok((crate_name, module, edges))
}
