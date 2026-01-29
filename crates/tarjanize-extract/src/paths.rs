//! Path building utilities for tarjanize.
//!
//! This module provides functions for constructing fully-qualified paths
//! for symbols in the dependency graph. These paths serve as unique identifiers
//! for symbols across the workspace.
//!
//! Path formats:
//! - ModuleDefs: `crate::module::item` (e.g., `mycrate::foo::MyStruct`)
//! - Impl blocks: `crate::module::impl Trait for Type` or `crate::module::impl Type`

use ra_ap_base_db::VfsPath;
use ra_ap_hir::{Impl, ModuleDef};
use ra_ap_ide_db::{FileId, RootDatabase};
use tracing::warn;

use crate::file_path;
use crate::impls::impl_name;

/// Returns the fully-qualified path for a ModuleDef.
///
/// Uses rust-analyzer's `canonical_path` method and prepends the crate name.
///
/// Returns None for items that don't have meaningful paths:
/// - Unnamed consts (`const _: () = ...`) have no name
/// - Built-in types (i32, bool, str) have no containing module
/// - Crate root modules have no containing module
/// - Items in crates without display names (shouldn't happen in practice)
pub(crate) fn module_def_path(
    db: &RootDatabase,
    def: &ModuleDef,
) -> Option<String> {
    let containing_module = def.module(db)?;
    let krate = containing_module.krate(db);
    let crate_name = krate.display_name(db)?;
    let edition = krate.edition(db);

    let canonical = def.canonical_path(db, edition)?;
    Some(format!("{}::{}", crate_name, canonical))
}

/// Returns the fully-qualified path for an Impl.
///
/// Impl paths use the format "crate::module::impl Trait for Type" or
/// "crate::module::impl Type" for inherent impls.
///
/// Returns None if the crate has no display name (shouldn't happen in practice).
pub(crate) fn impl_path(db: &RootDatabase, impl_: &Impl) -> Option<String> {
    let module = impl_.module(db);
    let krate = module.krate(db);
    let edition = krate.edition(db);

    // Build path: crate::mod1::mod2::impl Type
    // Unlike ModuleDef, Impl doesn't have canonical_path, so we build it manually.
    let crate_name = krate.display_name(db)?.to_string();
    let path = module
        .path_to_root(db)
        .into_iter()
        .rev()
        .map(|m| {
            // Use the module name if available; fallback to crate name for the
            // unnamed root module.
            m.name(db)
                .map(|n| n.display(db, edition).to_string())
                .unwrap_or_else(|| crate_name.clone())
        })
        .chain([impl_name(db, impl_)])
        .collect::<Vec<_>>()
        .join("::");

    Some(path)
}

/// Compute the file path relative to the crate root.
///
/// # Path handling
///
/// rust-analyzer uses `VfsPath` which can be either a real filesystem path
/// or a virtual path (for test fixtures). There's no built-in "relative to
/// crate root" API, so we compute it manually:
///
/// 1. Real paths: `/a/b/c/lib.rs` with root `/a/b/c` →
///    `strip_prefix()` returns `lib.rs`
/// 2. Virtual paths: `/lib.rs` with root `""` (parent of root-level file) →
///    `strip_prefix("")` returns `/lib.rs` unchanged, so we strip the `/`
pub(crate) fn compute_relative_file_path(
    db: &RootDatabase,
    crate_root: &VfsPath,
    file_id: FileId,
) -> String {
    let vfs_path = match file_path(db, file_id) {
        Ok(path) => path,
        Err(e) => {
            warn!(file_id = ?file_id, crate_root = ?crate_root, error = %e, "couldn't compute file path relative to crate root");
            return String::new();
        }
    };

    // Try to make the path relative to the crate root. If strip_prefix fails
    // (e.g., file outside crate root, or virtual path quirks), fall back to
    // the absolute path rather than panicking.
    let path = vfs_path
        .strip_prefix(crate_root)
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| vfs_path.to_string());

    // Normalize: virtual paths keep leading "/" after strip_prefix("").
    path.strip_prefix('/').unwrap_or(&path).to_owned()
}
