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
use ra_ap_hir::{HasSource, Impl, ModuleDef, Semantics};
use ra_ap_ide::TryToNav;
use ra_ap_ide_db::RootDatabase;
use ra_ap_syntax::AstNode;
use ra_ap_syntax::ast::HasGenericParams;
use tarjanize_schemas::{Symbol, SymbolKind};
use tracing::debug_span;

use crate::dependencies::is_local_def;
use crate::paths::{compute_relative_file_path, module_def_path};

/// Build the display name for an impl block from its AST.
///
/// Extracts the signature directly from source, giving names like:
/// - `impl<T> MyTrait for Wrapper<T>` (trait impl with generics)
/// - `impl Foo` (inherent impl)
/// - `impl<T: Clone> MyTrait for &T` (with bounds)
/// - `impl<T> MyTrait for Foo where T: Clone` (with where clause)
///
/// # Panics
///
/// Panics if AST is unavailable. This should never happen for local impls
/// that pass `try_to_nav()` - if it does, our assumptions are wrong.
pub(crate) fn impl_name(db: &RootDatabase, impl_: &Impl) -> String {
    use std::fmt::Write;

    let source = impl_.source(db).expect("local impl should have source");
    let ast = source.value;

    // Build the name incrementally to avoid intermediate allocations.
    // SyntaxText implements Display, so write! writes directly to the buffer.
    let mut name = String::new();

    // Unsafe keyword: `unsafe impl`
    if ast.unsafe_token().is_some() {
        name.push_str("unsafe ");
    }

    name.push_str("impl");

    // Generic params: `<T: Clone>`
    if let Some(g) = ast.generic_param_list() {
        _ = write!(name, "{}", g.syntax().text());
    }

    name.push(' ');

    // Trait part: `MyTrait for ` or `!MyTrait for ` (negative impl)
    if let Some(t) = ast.trait_() {
        if ast.excl_token().is_some() {
            name.push('!');
        }
        _ = write!(name, "{} for ", t.syntax().text());
    }

    // Self type: `Wrapper<T>`
    let self_ty = ast.self_ty().expect("impl should have self type");
    _ = write!(name, "{}", self_ty.syntax().text());

    // Where clause: ` where T: Clone`
    if let Some(w) = ast.where_clause() {
        _ = write!(name, " {}", w.syntax().text());
    }

    name
}

/// Extract an impl block as a (name, Symbol) pair.
pub(crate) fn extract_impl(
    sema: &Semantics<'_, RootDatabase>,
    crate_root: &VfsPath,
    impl_: Impl,
) -> Option<(String, Symbol)> {
    let db = sema.db;

    let name = impl_name(db, &impl_);
    let _span = debug_span!("extract_impl", %name).entered();

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

    // Collect all dependencies: impl signature (self type, trait, generic
    // bounds, where clauses) and associated items (methods, consts, type aliases).
    let dependencies = find_dependencies(sema, impl_);

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
/// Walks the entire impl syntax (signature + body) to collect dependencies.
/// This captures self type, trait, generic bounds, where clauses, and all
/// references in method/const/type bodies.
fn find_dependencies(
    sema: &Semantics<'_, RootDatabase>,
    impl_: Impl,
) -> HashSet<String> {
    let db = sema.db;

    // Walk the entire impl syntax (signature + body) to collect dependencies.
    // This captures: self type, trait, generic bounds, where clauses, and
    // all references in method/const/type bodies.
    sema.source(impl_)
        .map(|src| {
            crate::dependencies::collect_path_deps(sema, src.value.syntax())
        })
        .unwrap_or_default()
        .into_iter()
        .filter(|dep| is_local_def(db, dep))
        .filter_map(|dep| crate::module_defs::definition_path(db, &dep))
        .collect()
}

#[cfg(test)]
mod tests {
    use ra_ap_hir::{Crate, Semantics, attach_db};
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;
    use tarjanize_schemas::SymbolKind;

    use super::extract_impl;
    use crate::file_path;

    /// Helper to check impl extraction results.
    fn check_impl(
        fixture: &str,
        expected_name: &str,
        expected_self_type: Option<&str>,
        expected_trait: Option<&str>,
    ) {
        let db = RootDatabase::with_files(fixture);
        attach_db(&db, || {
            let krate = Crate::all(&db)
                .into_iter()
                .find(|k| {
                    k.display_name(&db)
                        .is_some_and(|n| n.to_string() == "test_crate")
                })
                .expect("test_crate not found");

            let crate_root = file_path(&db, krate.root_file(&db))
                .expect("crate root file")
                .parent()
                .expect("crate root parent");

            let sema = Semantics::new(&db);
            let impl_ = krate
                .root_module(&db)
                .impl_defs(&db)
                .into_iter()
                .next()
                .expect("no impl found");

            let (name, symbol) = extract_impl(&sema, &crate_root, impl_)
                .expect("extract_impl failed");

            assert_eq!(name, expected_name);

            let SymbolKind::Impl { self_type, trait_ } = &symbol.kind else {
                panic!("Expected SymbolKind::Impl, got {:?}", symbol.kind);
            };
            assert_eq!(self_type.as_deref(), expected_self_type);
            assert_eq!(trait_.as_deref(), expected_trait);
        });
    }

    // =========================================================================
    // IMPL EXTRACTION TESTS
    //
    // Tests for extract_impl covering name generation and symbol field extraction.
    // =========================================================================

    #[test]
    fn test_inherent_impl() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub struct Foo;
impl Foo {
    pub fn method(&self) {}
}
"#,
            "impl Foo",
            Some("test_crate::Foo"),
            None,
        );
    }

    #[test]
    fn test_trait_impl() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait { fn method(&self); }
pub struct Foo;
impl MyTrait for Foo { fn method(&self) {} }
"#,
            "impl MyTrait for Foo",
            Some("test_crate::Foo"),
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_impl_for_reference() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait { fn method(&self); }
pub struct Foo;
impl MyTrait for &Foo { fn method(&self) {} }
"#,
            "impl MyTrait for &Foo",
            None, // &Foo is not an ADT
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_blanket_impl() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait { fn method(&self); }
impl<T> MyTrait for T { fn method(&self) {} }
"#,
            "impl<T> MyTrait for T",
            None, // T is a generic param
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_inherent_impl_generic() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub struct Wrapper<T>(T);
impl<T> Wrapper<T> {
    pub fn new(t: T) -> Self { Self(t) }
}
"#,
            "impl<T> Wrapper<T>",
            Some("test_crate::Wrapper"),
            None,
        );
    }

    #[test]
    fn test_generic_bounds() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
impl<T: Clone> MyTrait for Foo {}
"#,
            "impl<T: Clone> MyTrait for Foo",
            Some("test_crate::Foo"),
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_lifetime_params() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Ref<'a>(&'a str);
impl<'a> MyTrait for Ref<'a> {}
"#,
            "impl<'a> MyTrait for Ref<'a>",
            Some("test_crate::Ref"),
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_multiple_type_params() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Pair<T, U>(T, U);
impl<T, U> MyTrait for Pair<T, U> {}
"#,
            "impl<T, U> MyTrait for Pair<T, U>",
            Some("test_crate::Pair"),
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_generic_trait_impl() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Wrapper<T>(T);
impl<T> MyTrait for Wrapper<T> {}
"#,
            "impl<T> MyTrait for Wrapper<T>",
            Some("test_crate::Wrapper"),
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_where_clause() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
impl<T> MyTrait for Foo where T: Clone {}
"#,
            "impl<T> MyTrait for Foo where T: Clone",
            Some("test_crate::Foo"),
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_unsafe_impl() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub unsafe trait UnsafeTrait {}
pub struct Foo;
unsafe impl UnsafeTrait for Foo {}
"#,
            "unsafe impl UnsafeTrait for Foo",
            Some("test_crate::Foo"),
            Some("test_crate::UnsafeTrait"),
        );
    }

    #[test]
    fn test_tuple_type() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct A;
pub struct B;
impl MyTrait for (A, B) {}
"#,
            "impl MyTrait for (A, B)",
            None, // tuple is not an ADT
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_array_type() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
impl MyTrait for [Foo; 3] {}
"#,
            "impl MyTrait for [Foo; 3]",
            None, // array is not an ADT
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_dyn_trait() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub trait OtherTrait {}
impl MyTrait for dyn OtherTrait {}
"#,
            "impl MyTrait for dyn OtherTrait",
            None, // dyn trait is not an ADT
            Some("test_crate::MyTrait"),
        );
    }

    #[test]
    fn test_external_generic_with_local_type() {
        check_impl(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
impl MyTrait for Box<Foo> {}
"#,
            "impl MyTrait for Box<Foo>",
            None, // Box<Foo> is not a local ADT
            Some("test_crate::MyTrait"),
        );
    }
}
