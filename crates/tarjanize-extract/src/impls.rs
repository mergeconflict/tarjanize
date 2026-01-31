//! Impl block extraction for tarjanize.
//!
//! This module handles extraction of impl blocks into the schema format.
//! Impl blocks are anonymous (you can't write a path like `mycrate::SomeImpl`),
//! so they require special handling compared to named `ModuleDefs`.
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
use ra_ap_syntax::ast::{HasAttrs as _, HasGenericParams};
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

    // Collect all dependencies: impl signature (self type, trait, generic
    // bounds, where clauses) and associated items (methods, consts, type aliases).
    let dependencies = find_dependencies(sema, impl_);

    // Collect orphan rule anchors: workspace-local types/traits that can
    // satisfy the orphan rule. For `impl<P1..=Pn> Trait<T1..=Tn> for T0`:
    // - The trait (if local)
    // - T0..=Tn (the self type and trait type parameters, if local)
    let anchors = collect_anchors(db, impl_);

    Some((
        name,
        Symbol {
            file,
            cost,
            dependencies,
            kind: SymbolKind::Impl { anchors },
        },
    ))
}

/// Collect orphan rule anchors for an impl block.
///
/// Returns the fully qualified paths of workspace-local types and traits
/// that can satisfy the orphan rule for this impl.
///
/// For `impl<P1..=Pn> Trait<T1..=Tn> for T0`, the orphan rule allows the impl
/// if either:
/// - The trait is local, OR
/// - At least one of T0..=Tn is local
///
/// Fundamental type constructors (`&`, `&mut`, `Box`, `Pin`) are special:
/// `Wrapper<LocalType>` is considered a local type, so we recursively
/// extract anchors from inside them.
fn collect_anchors(db: &RootDatabase, impl_: Impl) -> HashSet<String> {
    let mut anchors = HashSet::new();

    // Collect anchors from self type T0, handling fundamental wrappers.
    collect_anchors_from_type(db, &impl_.self_ty(db), &mut anchors);

    // Add trait if it's local.
    if let Some(trait_) = impl_.trait_(db) {
        if let Some(path) = module_def_path(db, &ModuleDef::Trait(trait_)) {
            anchors.insert(path);
        }

        // Collect anchors from trait type parameters T1..=Tn.
        // Index 0 is Self, so we skip it and iterate from 1 onward.
        if let Some(trait_ref) = impl_.trait_ref(db) {
            for ty_ns in (1..).map_while(|i| trait_ref.get_type_argument(i)) {
                collect_anchors_from_type(db, &ty_ns.to_type(db), &mut anchors);
            }
        }
    }

    anchors
}

/// Recursively collect anchors from a type, unwrapping fundamental types.
///
/// Fundamental types (`&T`, `&mut T`, `Box<T>`, `Pin<T>`) don't "cover" their
/// inner type for orphan rule purposes, so `Box<LocalType>` counts as local.
/// We recursively unwrap these to find the actual anchors.
fn collect_anchors_from_type(
    db: &RootDatabase,
    ty: &ra_ap_hir::Type,
    anchors: &mut HashSet<String>,
) {
    // Strip all reference layers (&T, &mut T are fundamental).
    let ty = ty.strip_references();

    // Only ADTs (struct/enum/union) can be anchors.
    let Some(adt) = ty.as_adt() else {
        return;
    };

    // Fundamental types don't cover inner types, so recurse to find anchors.
    if is_fundamental_adt(db, &adt) {
        for ty_arg in ty.type_arguments() {
            collect_anchors_from_type(db, &ty_arg, anchors);
        }
        return;
    }

    // Add as anchor if it's a workspace-local ADT (module_def_path returns
    // None for external crates like std).
    if let Some(path) = module_def_path(db, &ModuleDef::Adt(adt)) {
        anchors.insert(path);
    }

    // For tuples, arrays, etc. - these are NOT fundamental, so inner types
    // don't count as anchors. We don't recurse into them.
}

/// Check if an ADT has the `#[fundamental]` attribute.
///
/// [Fundamental types] cannot "[cover]" other types for the orphan rule, so
/// `Box<LocalType>` allows `LocalType` to satisfy the rule. References (`&`,
/// `&mut`) are also fundamental but handled separately via `strip_references()`.
///
/// [Fundamental types]: https://doc.rust-lang.org/reference/glossary.html#fundamental-type-constructors
/// [cover]: https://doc.rust-lang.org/reference/glossary.html#uncovered-type
fn is_fundamental_adt(db: &RootDatabase, adt: &ra_ap_hir::Adt) -> bool {
    adt.source(db)
        .is_some_and(|src| src.value.has_atom_attr("fundamental"))
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
    use std::collections::HashSet;

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
        expected_anchors: &[&str],
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

            let SymbolKind::Impl { anchors } = &symbol.kind else {
                panic!("Expected SymbolKind::Impl, got {:?}", symbol.kind);
            };
            let expected: HashSet<String> =
                expected_anchors.iter().map(|&s| s.to_string()).collect();
            assert_eq!(anchors, &expected);
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
            r"
//- /lib.rs crate:test_crate
pub struct Foo;
impl Foo {
    pub fn method(&self) {}
}
",
            "impl Foo",
            &["test_crate::Foo"],
        );
    }

    #[test]
    fn test_trait_impl() {
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait { fn method(&self); }
pub struct Foo;
impl MyTrait for Foo { fn method(&self) {} }
",
            "impl MyTrait for Foo",
            &["test_crate::Foo", "test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_impl_for_reference() {
        // &Foo - reference is fundamental, so Foo inside counts as an anchor.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait { fn method(&self); }
pub struct Foo;
impl MyTrait for &Foo { fn method(&self) {} }
",
            "impl MyTrait for &Foo",
            &["test_crate::MyTrait", "test_crate::Foo"],
        );
    }

    #[test]
    fn test_blanket_impl() {
        // T is a generic param, so only the trait is an anchor.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait { fn method(&self); }
impl<T> MyTrait for T { fn method(&self) {} }
",
            "impl<T> MyTrait for T",
            &["test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_inherent_impl_generic() {
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub struct Wrapper<T>(T);
impl<T> Wrapper<T> {
    pub fn new(t: T) -> Self { Self(t) }
}
",
            "impl<T> Wrapper<T>",
            &["test_crate::Wrapper"],
        );
    }

    #[test]
    fn test_generic_bounds() {
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
impl<T: Clone> MyTrait for Foo {}
",
            "impl<T: Clone> MyTrait for Foo",
            &["test_crate::Foo", "test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_lifetime_params() {
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Ref<'a>(&'a str);
impl<'a> MyTrait for Ref<'a> {}
",
            "impl<'a> MyTrait for Ref<'a>",
            &["test_crate::Ref", "test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_multiple_type_params() {
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Pair<T, U>(T, U);
impl<T, U> MyTrait for Pair<T, U> {}
",
            "impl<T, U> MyTrait for Pair<T, U>",
            &["test_crate::Pair", "test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_generic_trait_impl() {
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Wrapper<T>(T);
impl<T> MyTrait for Wrapper<T> {}
",
            "impl<T> MyTrait for Wrapper<T>",
            &["test_crate::Wrapper", "test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_where_clause() {
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
impl<T> MyTrait for Foo where T: Clone {}
",
            "impl<T> MyTrait for Foo where T: Clone",
            &["test_crate::Foo", "test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_unsafe_impl() {
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub unsafe trait UnsafeTrait {}
pub struct Foo;
unsafe impl UnsafeTrait for Foo {}
",
            "unsafe impl UnsafeTrait for Foo",
            &["test_crate::Foo", "test_crate::UnsafeTrait"],
        );
    }

    #[test]
    fn test_tuple_type() {
        // Tuple is not an ADT, so only the trait is an anchor.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct A;
pub struct B;
impl MyTrait for (A, B) {}
",
            "impl MyTrait for (A, B)",
            &["test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_array_type() {
        // Array is not an ADT, so only the trait is an anchor.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
impl MyTrait for [Foo; 3] {}
",
            "impl MyTrait for [Foo; 3]",
            &["test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_dyn_trait() {
        // dyn Trait is not an ADT, so only the impl'd trait is an anchor.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub trait OtherTrait {}
impl MyTrait for dyn OtherTrait {}
",
            "impl MyTrait for dyn OtherTrait",
            &["test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_fundamental_box() {
        // Box<T> where Box has #[fundamental] - we recurse into T.
        // Since T is a type parameter (not a concrete type), only the trait
        // is an anchor.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
#[fundamental]
pub struct Box<T>(T);
impl<T> MyTrait for Box<T> {}
",
            "impl<T> MyTrait for Box<T>",
            &["test_crate::MyTrait"],
        );
    }

    #[test]
    fn test_fundamental_box_with_concrete_type() {
        // Box<Foo> where Box has #[fundamental] - Foo inside is the anchor.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
#[fundamental]
pub struct Box<T>(T);
impl MyTrait for Box<Foo> {}
",
            "impl MyTrait for Box<Foo>",
            &["test_crate::MyTrait", "test_crate::Foo"],
        );
    }

    #[test]
    fn test_fundamental_mut_ref() {
        // &mut Foo - mutable reference is fundamental, so Foo counts as anchor.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
impl MyTrait for &mut Foo {}
",
            "impl MyTrait for &mut Foo",
            &["test_crate::MyTrait", "test_crate::Foo"],
        );
    }

    #[test]
    fn test_non_fundamental_wrapper() {
        // Wrapper<Foo> - custom wrapper is NOT fundamental, so Foo doesn't count.
        // Only the wrapper itself and the trait are anchors.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct Foo;
pub struct Wrapper<T>(T);
impl<T> MyTrait for Wrapper<T> {}
",
            "impl<T> MyTrait for Wrapper<T>",
            &["test_crate::MyTrait", "test_crate::Wrapper"],
        );
    }

    #[test]
    fn test_trait_type_param() {
        // impl MyTrait<Foo> for i32 - Foo is a trait type parameter.
        // Note: We define our own trait since std::From isn't available.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub struct Foo;
pub trait Convert<T> {
    fn convert(self) -> T;
}
impl Convert<Foo> for i32 {
    fn convert(self) -> Foo { Foo }
}
",
            "impl Convert<Foo> for i32",
            // Both the trait and its type parameter are local anchors.
            &["test_crate::Convert", "test_crate::Foo"],
        );
    }

    #[test]
    fn test_trait_type_param_with_local_self() {
        // impl Convert<A> for B - both A (trait type param) and B (self) are anchors.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub struct A;
pub struct B;
pub trait Convert<T> {
    fn convert(self) -> T;
}
impl Convert<A> for B {
    fn convert(self) -> A { A }
}
",
            "impl Convert<A> for B",
            &["test_crate::A", "test_crate::B", "test_crate::Convert"],
        );
    }

    #[test]
    fn test_trait_multiple_type_params() {
        // impl MyTrait<A, B> for C - all type params and self are anchors.
        check_impl(
            r"
//- /lib.rs crate:test_crate
pub struct A;
pub struct B;
pub struct C;
pub trait MyTrait<T, U> {}
impl MyTrait<A, B> for C {}
",
            "impl MyTrait<A, B> for C",
            &[
                "test_crate::A",
                "test_crate::B",
                "test_crate::C",
                "test_crate::MyTrait",
            ],
        );
    }
}
