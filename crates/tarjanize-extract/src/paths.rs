//! Path building utilities for tarjanize.
//!
//! This module provides functions for constructing fully-qualified paths
//! for symbols in the dependency graph. These paths serve as unique identifiers
//! for symbols across the workspace.
//!
//! Path formats:
//! - ModuleDefs: `crate::module::item` (e.g., `mycrate::foo::MyStruct`)
//! - Impl blocks: `crate::module::impl Trait for Type` or `crate::module::impl Type`

use ra_ap_hir::{
    DisplayTarget, HirDisplay, Impl, Module as HirModule, ModuleDef,
};
use ra_ap_ide_db::RootDatabase;
use ra_ap_ide_db::defs::Definition;
use tracing::warn;

use crate::file_path;

/// Build the fully-qualified path to a module (e.g., "mycrate::foo::bar").
///
/// We build paths manually rather than using rust-analyzer's display methods
/// because we need stable, consistent identifiers for the dependency graph.
pub(crate) fn build_module_path(
    db: &RootDatabase,
    module: &HirModule,
    crate_name: &str,
) -> String {
    let parts: Vec<_> = std::iter::once(crate_name.to_owned())
        .chain(
            module
                .path_to_root(db)
                .into_iter()
                .rev()
                .filter_map(|m| m.name(db))
                // We use as_str() rather than display() because we want a stable
                // name that doesn't depend on edition.
                .map(|n| n.as_str().to_owned()),
        )
        .collect();

    parts.join("::")
}

/// Returns the fully-qualified path for a ModuleDef.
///
/// Returns None for items without clear paths (e.g., built-in types).
pub(crate) fn module_def_path(
    db: &RootDatabase,
    def: &ModuleDef,
) -> Option<String> {
    let name = def.name(db)?;
    let containing_module = def.module(db)?;

    let crate_name = containing_module
        .krate(db)
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    let module_path = build_module_path(db, &containing_module, &crate_name);

    Some(format!("{}::{}", module_path, name.as_str()))
}

/// Returns the fully-qualified path for a Definition dependency target.
///
/// This handles the normalized Definition variants that represent valid
/// dependency targets. Other Definition variants (Module, BuiltinType, Local,
/// etc.) should have been filtered out by `normalize_definition`.
pub(crate) fn definition_path(
    db: &RootDatabase,
    def: &Definition,
) -> Option<String> {
    match def {
        // Impl blocks use special formatting.
        Definition::SelfType(impl_) => impl_path(db, impl_),

        // All other valid dependency targets are module-level definitions.
        // Convert to ModuleDef for path computation.
        Definition::Function(f) => {
            module_def_path(db, &ModuleDef::Function(*f))
        }
        Definition::Adt(a) => module_def_path(db, &ModuleDef::Adt(*a)),
        Definition::Const(c) => module_def_path(db, &ModuleDef::Const(*c)),
        Definition::Static(s) => module_def_path(db, &ModuleDef::Static(*s)),
        Definition::Trait(t) => module_def_path(db, &ModuleDef::Trait(*t)),
        Definition::TypeAlias(ta) => {
            module_def_path(db, &ModuleDef::TypeAlias(*ta))
        }
        Definition::Macro(m) => module_def_path(db, &ModuleDef::Macro(*m)),

        // These should have been filtered out by normalize_definition.
        // Return None rather than panicking to be defensive.
        Definition::Module(_)
        | Definition::Crate(_)
        | Definition::Variant(_)
        | Definition::Field(_)
        | Definition::TupleField(_)
        | Definition::BuiltinType(_)
        | Definition::BuiltinLifetime(_)
        | Definition::Local(_)
        | Definition::GenericParam(_)
        | Definition::Label(_)
        | Definition::DeriveHelper(_)
        | Definition::BuiltinAttr(_)
        | Definition::ToolModule(_)
        | Definition::ExternCrateDecl(_)
        | Definition::InlineAsmRegOrRegClass(_)
        | Definition::InlineAsmOperand(_) => None,
    }
}

/// Returns the fully-qualified path for an Impl.
///
/// Impl paths use the format "crate::module::impl Trait for Type" or
/// "crate::module::impl Type" for inherent impls.
pub(crate) fn impl_path(db: &RootDatabase, impl_: &Impl) -> Option<String> {
    let module = impl_.module(db);
    let krate = module.krate(db);
    let crate_name = krate
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    let module_path = build_module_path(db, &module, &crate_name);
    let name = impl_name(db, impl_);

    Some(format!("{}::{}", module_path, name))
}

/// Build the display name for an impl block.
///
/// Uses HirDisplay to get proper type names including generics, references,
/// slices, trait objects, etc. This ensures unique names for impls like
/// `impl<T> Foo for &T` vs `impl<T> Foo for Box<T>`.
///
/// Returns names like "impl Trait for Type" or "impl Type" for inherent impls.
pub(crate) fn impl_name(db: &RootDatabase, impl_: &Impl) -> String {
    let self_ty = impl_.self_ty(db);
    let display_target =
        DisplayTarget::from_crate(db, impl_.module(db).krate(db).into());
    let self_ty_name = self_ty.display(db, display_target).to_string();

    if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    }
}

/// Compute the file path relative to the crate root.
///
/// # Path handling
///
/// rust-analyzer uses `VfsPath` which can be either a real filesystem path
/// or a virtual path (for test fixtures). There's no built-in "relative to
/// crate root" API, so we compute it manually:
///
/// 1. Real paths: `/Users/.../src/lib.rs` with root `/Users/.../src` →
///    `strip_prefix()` returns `lib.rs`
/// 2. Virtual paths: `/lib.rs` with root `""` (parent of root-level file) →
///    `strip_prefix("")` returns `/lib.rs` unchanged, so we strip the `/`
pub(crate) fn compute_relative_file_path(
    db: &RootDatabase,
    crate_root: &ra_ap_base_db::VfsPath,
    file_id: ra_ap_ide_db::FileId,
) -> String {
    let vfs_path = match file_path(db, file_id) {
        Ok(path) => path,
        Err(e) => {
            warn!(%e, "could not resolve file path");
            return String::new();
        }
    };

    let path = vfs_path
        .strip_prefix(crate_root)
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| vfs_path.to_string());

    // Normalize: virtual paths keep leading "/" after strip_prefix("").
    path.strip_prefix('/').unwrap_or(&path).to_owned()
}
