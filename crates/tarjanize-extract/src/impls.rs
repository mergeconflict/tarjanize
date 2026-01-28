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
use ra_ap_ide_db::defs::Definition;
use tarjanize_schemas::{Symbol, SymbolKind};

use crate::dependencies::is_local_def;
use crate::module_defs::find_dependencies as find_module_def_dependencies;
use crate::paths::{compute_relative_file_path, impl_name, module_def_path};

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

    // Extract self type ADT and trait from the impl declaration.
    // Used for both the schema (orphan rule analysis) and dependency collection.
    let self_type = impl_.self_ty(db).as_adt();
    let trait_ = impl_.trait_(db);

    let self_type_path =
        self_type.and_then(|adt| module_def_path(db, &ModuleDef::Adt(adt)));
    let trait_path =
        trait_.and_then(|t| module_def_path(db, &ModuleDef::Trait(t)));

    // Collect all dependencies: impl declaration (self type, trait) and
    // associated items (methods, consts, type aliases).
    let dependencies = find_dependencies(sema, impl_, self_type, trait_);

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

/// Find all items that an impl block depends on.
///
/// Impl blocks are NOT part of ModuleDef (they're anonymous - you can't write
/// a path like `my_crate::SomeImpl`). This function handles the impl-specific
/// dependencies.
///
/// ## Dependencies captured
///
/// 1. **Self type**: `impl Foo { }` depends on Foo
///    - `impl Vec<Bar> { }` depends on both Vec and Bar
///    - The ADT (struct/enum/union) is passed in from the caller
///
/// 2. **Trait** (for trait impls): `impl Trait for Type { }` depends on Trait
///
/// 3. **Impl body**: The methods and associated items inside the impl
///    - Per PLAN.md, we collapse these to the impl block - the impl depends on
///      whatever its methods depend on, but the methods aren't separate symbols.
///
/// ## Why impl dependencies matter for tarjanize
///
/// If we split `struct Foo` into crate A and `impl Foo { }` stays in crate B,
/// we create a dependency from B→A. Worse, for `impl Trait for Type`, the impl
/// MUST live in the same crate as either Trait or Type (orphan rules). We need
/// to track these dependencies to respect those constraints.
fn find_dependencies(
    sema: &Semantics<'_, RootDatabase>,
    impl_: Impl,
    self_adt: Option<ra_ap_hir::Adt>,
    trait_: Option<ra_ap_hir::Trait>,
) -> HashSet<String> {
    let db = sema.db;

    // Collect all dependencies: declaration (self type, trait) and associated items.
    [self_adt.map(ModuleDef::Adt), trait_.map(ModuleDef::Trait)]
        .into_iter()
        .flatten()
        .filter(|dep| is_local_def(db, &Definition::from(*dep)))
        .filter_map(|dep| module_def_path(db, &dep))
        .chain(
            impl_.items(db).into_iter().flat_map(|item| {
                find_module_def_dependencies(sema, item.into())
            }),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;
    use tarjanize_schemas::SymbolKind;

    use crate::extract_symbol_graph;

    /// Inherent impl (`impl Foo { }`) should have self_type but no trait.
    #[test]
    fn test_inherent_impl() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Foo;

impl Foo {
    pub fn method(&self) {}
}
"#,
        );
        let graph = extract_symbol_graph(db);
        let root = &graph.crates["test_crate"];

        let symbol = root
            .symbols
            .get("impl Foo")
            .expect("Should have 'impl Foo' symbol");

        // Verify it's an Impl with correct self_type and trait_
        if let SymbolKind::Impl { self_type, trait_ } = &symbol.kind {
            assert_eq!(
                self_type.as_deref(),
                Some("test_crate::Foo"),
                "self_type should be test_crate::Foo"
            );
            assert_eq!(trait_.as_deref(), None, "trait_ should be None");
        } else {
            panic!("Expected SymbolKind::Impl, got {:?}", symbol.kind);
        }

        // Verify file path and cost are populated
        assert_eq!(symbol.file, "lib.rs");
        assert!(symbol.cost > 0.0, "cost should be non-zero");
    }

    /// Trait impl (`impl Trait for Foo { }`) should have both self_type and trait.
    #[test]
    fn test_trait_impl() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    fn method(&self);
}

pub struct Foo;

impl MyTrait for Foo {
    fn method(&self) {}
}
"#,
        );
        let graph = extract_symbol_graph(db);
        let root = &graph.crates["test_crate"];

        let symbol = root
            .symbols
            .get("impl MyTrait for Foo")
            .expect("Should have 'impl MyTrait for Foo' symbol");

        if let SymbolKind::Impl { self_type, trait_ } = &symbol.kind {
            assert_eq!(
                self_type.as_deref(),
                Some("test_crate::Foo"),
                "self_type should be test_crate::Foo"
            );
            assert_eq!(
                trait_.as_deref(),
                Some("test_crate::MyTrait"),
                "trait_ should be test_crate::MyTrait"
            );
        } else {
            panic!("Expected SymbolKind::Impl, got {:?}", symbol.kind);
        }
    }

    /// Impl for reference type (`impl Trait for &Foo { }`) has self_type = None
    /// because &Foo is not an ADT.
    #[test]
    fn test_impl_for_reference() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    fn method(&self);
}

pub struct Foo;

impl MyTrait for &Foo {
    fn method(&self) {}
}
"#,
        );
        let graph = extract_symbol_graph(db);
        let root = &graph.crates["test_crate"];

        let symbol = root
            .symbols
            .get("impl MyTrait for &Foo")
            .expect("Should have 'impl MyTrait for &Foo' symbol");

        if let SymbolKind::Impl { self_type, trait_ } = &symbol.kind {
            // &Foo is not an ADT, so self_type is None
            assert_eq!(
                self_type.as_deref(),
                None,
                "self_type should be None for &Foo"
            );
            assert_eq!(
                trait_.as_deref(),
                Some("test_crate::MyTrait"),
                "trait_ should be test_crate::MyTrait"
            );
        } else {
            panic!("Expected SymbolKind::Impl, got {:?}", symbol.kind);
        }
    }

    /// Blanket impl (`impl<T> Trait for T { }`) has self_type = None
    /// because T is a generic param, not an ADT.
    #[test]
    fn test_blanket_impl() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    fn method(&self);
}

impl<T> MyTrait for T {
    fn method(&self) {}
}
"#,
        );
        let graph = extract_symbol_graph(db);
        let root = &graph.crates["test_crate"];

        let symbol = root
            .symbols
            .get("impl MyTrait for T")
            .expect("Should have 'impl MyTrait for T' symbol");

        if let SymbolKind::Impl { self_type, trait_ } = &symbol.kind {
            // T is a generic param, not an ADT
            assert_eq!(
                self_type.as_deref(),
                None,
                "self_type should be None for T"
            );
            assert_eq!(
                trait_.as_deref(),
                Some("test_crate::MyTrait"),
                "trait_ should be test_crate::MyTrait"
            );
        } else {
            panic!("Expected SymbolKind::Impl, got {:?}", symbol.kind);
        }
    }

    /// Multiple impl blocks with the same signature should be merged,
    /// combining their costs and dependencies.
    #[test]
    fn test_impl_merging() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Foo;
pub struct DepA;
pub struct DepB;

impl Foo {
    pub fn method_a(&self) -> DepA { DepA }
}

impl Foo {
    pub fn method_b(&self) -> DepB { DepB }
}
"#,
        );
        let graph = extract_symbol_graph(db);
        let root = &graph.crates["test_crate"];

        // Should have exactly one "impl Foo" symbol (merged)
        let symbol = root
            .symbols
            .get("impl Foo")
            .expect("Should have merged 'impl Foo' symbol");

        // Dependencies should include both DepA and DepB
        assert!(
            symbol.dependencies.contains("test_crate::DepA"),
            "Should depend on DepA"
        );
        assert!(
            symbol.dependencies.contains("test_crate::DepB"),
            "Should depend on DepB"
        );
    }
}
