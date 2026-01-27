//! Crate-level extraction for tarjanize.
//!
//! This module provides the bridge between workspace iteration and module
//! extraction. It handles crate-specific concerns like extracting the crate
//! name from Cargo.toml and determining the crate root directory.
//!
//! A crate is conceptually just its root module with some additional metadata.
//! This module extracts that metadata and delegates the actual symbol
//! extraction to the `modules` module.

use ra_ap_hir::{Crate, Semantics};
use ra_ap_ide_db::RootDatabase;
use tarjanize_schemas::Module;
use tracing::debug_span;

use crate::error::ExtractError;
use crate::file_path;
use crate::modules::extract_module;

/// Extract a crate as a (name, module) tuple.
///
/// A crate is represented as its root module, which contains all symbols
/// and submodules (each symbol includes its own dependencies). This unifies
/// crates and modules in the schema - a crate is just a module with a name
/// from Cargo.toml.
pub(crate) fn extract_crate(
    sema: &Semantics<'_, RootDatabase>,
    krate: Crate,
) -> Result<(String, Module), ExtractError> {
    let db = sema.db;

    // All Cargo workspace crates must have names in Cargo.toml. A missing
    // display name indicates a non-Cargo build system or synthetic crate.
    let crate_name = krate
        .display_name(db)
        .ok_or_else(ExtractError::crate_name_missing)?
        .to_string();

    let _span = debug_span!("extract_crate", %crate_name).entered();

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
        extract_module(sema, &crate_root, &root_module, &crate_name);

    Ok((crate_name, module))
}

#[cfg(test)]
mod tests {
    use ra_ap_hir::{Crate, Semantics};
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;

    use super::*;

    #[test]
    fn test_extract_crate_returns_name_and_symbols() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
"#,
        );

        let krate = Crate::all(&db)
            .into_iter()
            .find(|k| k.origin(&db).is_local())
            .expect("should have a local crate");

        let sema = Semantics::new(&db);
        let (name, _) =
            extract_crate(&sema, krate).expect("extraction should succeed");

        assert_eq!(name, "test_crate");
    }
}
