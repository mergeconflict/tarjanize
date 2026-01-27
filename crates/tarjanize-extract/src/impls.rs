//! Impl block extraction for tarjanize.
//!
//! This module handles extraction of impl blocks into the schema format.
//! Impl blocks are anonymous (you can't write a path like `mycrate::SomeImpl`),
//! so they require special handling compared to named ModuleDefs.
//!
//! Impl blocks are important compilation units because:
//! - They define method implementations that create dependencies
//! - The orphan rule constrains where they can live (same crate as trait or type)
//! - Multiple impl blocks with the same signature are merged in the schema

use std::collections::HashSet;

use ra_ap_base_db::VfsPath;
use ra_ap_hir::{Impl, ModuleDef, Semantics};
use ra_ap_ide::TryToNav;
use ra_ap_ide_db::RootDatabase;
use tarjanize_schemas::{Symbol, SymbolKind};

use crate::dependencies::{find_dependencies, is_local};
use crate::paths::{
    compute_relative_file_path, definition_path, impl_name, module_def_path,
};

/// Extract an impl block as a (name, Symbol) pair.
pub(crate) fn extract_impl(
    sema: &Semantics<'_, RootDatabase>,
    crate_root: &VfsPath,
    impl_: Impl,
) -> Option<(String, Symbol)> {
    let db = sema.db;

    let name = impl_name(db, &impl_);

    // Get source location for file path and cost calculation via TryToNav.
    let nav = impl_.try_to_nav(sema)?.call_site;
    let file = compute_relative_file_path(db, crate_root, nav.file_id);
    let cost: f64 = u32::from(nav.full_range.len()).into();

    // Extract self_type and trait paths for the schema.
    // These are used to enforce the orphan rule during analysis.
    let self_type_path = impl_
        .self_ty(db)
        .as_adt()
        .and_then(|adt| module_def_path(db, &ModuleDef::Adt(adt)));

    let trait_path = impl_
        .trait_(db)
        .and_then(|t| module_def_path(db, &t.into()));

    // Collect dependencies from the impl declaration (self type, trait).
    let impl_deps = find_impl_dependencies(db, impl_);
    let mut dependencies: HashSet<String> = impl_deps
        .deps
        .into_iter()
        .filter(|dep| is_local(db, dep))
        .filter_map(|dep| module_def_path(db, &dep))
        .collect();

    // Collect dependencies from associated items (methods, consts, type aliases).
    // Per PLAN.md, we collapse these to the impl block - the impl depends on
    // whatever its methods depend on, but the methods aren't separate symbols.
    for item in impl_deps.items {
        dependencies.extend(
            find_dependencies(sema, item.into())
                .into_iter()
                .filter_map(|dep| definition_path(db, &dep)),
        );
    }

    Some((
        name,
        Symbol {
            file,
            cost,
            dependencies,
            kind: SymbolKind::Impl {
                self_type: self_type_path,
                trait_: trait_path,
            },
        },
    ))
}

/// Analyze dependencies for an impl block.
///
/// Impl blocks are NOT part of ModuleDef (they're anonymous - you can't write
/// a path like `my_crate::SomeImpl`). This function handles the impl-specific
/// dependencies that `find_dependencies` cannot capture.
///
/// ## Dependencies captured
///
/// 1. **Self type**: `impl Foo { }` depends on Foo
///    - `impl Vec<Bar> { }` depends on both Vec and Bar
///    - We get the ADT (struct/enum/union) from the self type
///
/// 2. **Trait** (for trait impls): `impl Trait for Type { }` depends on Trait
///
/// 3. **Impl body**: The methods and associated items inside the impl
///    - These are analyzed separately as Functions via `find_dependencies`
///    - We return them for the caller to process
///
/// ## Why impl dependencies matter for tarjanize
///
/// If we split `struct Foo` into crate A and `impl Foo { }` stays in crate B,
/// we create a dependency from B→A. Worse, for `impl Trait for Type`, the impl
/// MUST live in the same crate as either Trait or Type (orphan rules). We need
/// to track these dependencies to respect those constraints.
pub(crate) fn find_impl_dependencies(
    db: &RootDatabase,
    impl_: Impl,
) -> ImplDependencies {
    let mut deps = Vec::new();

    // Get the self type (the type being implemented).
    // For `impl Foo { }` this is Foo.
    // For `impl Trait for Type { }` this is Type.
    let self_ty = impl_.self_ty(db);

    // Extract the ADT (struct/enum/union) from the self type.
    // For `impl Foo { }` or `impl Foo<Bar> { }`, we get Foo.
    // Generic parameters like Bar are typically captured when the impl body
    // references them in type positions (which path resolution handles).
    if let Some(adt) = self_ty.as_adt() {
        deps.push(ModuleDef::Adt(adt));
    }

    // Get the trait (if this is a trait impl).
    // For `impl Clone for Foo { }` this is Some(Clone).
    // For `impl Foo { }` (inherent impl) this is None.
    if let Some(trait_) = impl_.trait_(db) {
        deps.push(ModuleDef::Trait(trait_));
    }

    // Get the associated items (methods, consts, types) in the impl.
    // These need their own dependency analysis via find_dependencies().
    let items = impl_.items(db);

    ImplDependencies { deps, items }
}

/// The result of analyzing an impl block's dependencies.
#[derive(Debug)]
pub(crate) struct ImplDependencies {
    /// Direct dependencies from the impl declaration itself (self type, trait).
    pub deps: Vec<ModuleDef>,

    /// Associated items in the impl that need their own dependency analysis.
    /// The caller should process these with `find_dependencies()`.
    pub items: Vec<ra_ap_hir::AssocItem>,
}
