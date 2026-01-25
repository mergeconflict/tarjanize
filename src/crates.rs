use std::collections::HashSet;

use ra_ap_hir::{Crate, Semantics};
use ra_ap_ide_db::RootDatabase;

use crate::modules::extract_module;
use crate::schemas::{Edge, Module as SchemaModule};

/// Extract a crate as a SchemaModule (the crate root module).
///
/// A crate is represented as its root module, which contains all symbols
/// and submodules. This unifies crates and modules in the schema - a crate
/// is just a module with a name from Cargo.toml.
pub fn extract_crate(
    sema: &Semantics<'_, RootDatabase>,
    krate: Crate,
    edges: &mut HashSet<Edge>,
) -> SchemaModule {
    let db = sema.db;

    let crate_name = krate
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    // Get the crate's root module and extract it recursively.
    let root_module = krate.root_module();
    extract_module(sema, &root_module, &crate_name, edges)
}
