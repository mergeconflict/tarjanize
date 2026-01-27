//! ModuleDef extraction for tarjanize.
//!
//! This module handles extraction of `ModuleDef` items (functions, structs,
//! enums, traits, consts, statics, type aliases, macros) into the schema format.
//!
//! ModuleDefs are the named items declared at module scope. They contrast with
//! impl blocks, which are anonymous and handled separately in `impls.rs`.

use std::collections::HashSet;

use ra_ap_base_db::VfsPath;
use ra_ap_hir::{HasVisibility, ModuleDef, Semantics};
use ra_ap_ide::TryToNav;
use ra_ap_ide_db::{RootDatabase, SymbolKind as RaSymbolKind};
use tarjanize_schemas::{Symbol, SymbolKind, Visibility};
use tracing::debug_span;

use crate::dependencies::find_dependencies;
use crate::paths::{compute_relative_file_path, definition_path};

/// Extract a single ModuleDef as a (name, Symbol) pair.
///
/// Returns None for items without names (e.g., unnamed consts) or without
/// source locations (e.g., built-in types).
pub(crate) fn extract_module_def(
    sema: &Semantics<'_, RootDatabase>,
    crate_root: &VfsPath,
    def: ModuleDef,
) -> Option<(String, Symbol)> {
    // Skip modules - they're represented in the output as the nested Module
    // structure, not as symbols within their parent module.
    if let ModuleDef::Module(_) = def {
        return None;
    }

    let db = sema.db;

    // Get name. Unnamed constants (`const _: T = ...`) return None here.
    let name = def.name(db)?.as_str().to_owned();

    let _span = debug_span!("extract_module_def", %name).entered();

    // Get source location for file path and cost calculation via TryToNav.
    let nav = def.try_to_nav(sema)?.call_site;
    let file = compute_relative_file_path(db, crate_root, nav.file_id);

    // Compute cost as the byte size of the symbol's syntax node.
    // This is a rough proxy for compile-time complexity.
    let cost = u32::from(nav.full_range.len()) as f64;

    // Collect dependencies in this workspace for this symbol.
    let dependencies: HashSet<String> = find_dependencies(sema, def)
        .into_iter()
        .filter_map(|dep| definition_path(db, &dep))
        .collect();

    // Determine the kind string using rust-analyzer's SymbolKind.
    let kind = format!("{:?}", RaSymbolKind::from_module_def(db, def));

    // Extract visibility from the ModuleDef.
    let visibility = match def.visibility(db) {
        ra_ap_hir::Visibility::Public => Visibility::Public,
        _ => Visibility::NonPublic,
    };

    Some((
        name,
        Symbol {
            file,
            cost,
            dependencies,
            kind: SymbolKind::ModuleDef { kind, visibility },
        },
    ))
}

#[cfg(test)]
mod tests {
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;

    use crate::extract_symbol_graph;

    /// Verify macro-generated symbols are extracted with correct file paths.
    ///
    /// When a macro_rules! macro generates a function, the symbol's HirFileId
    /// points to the macro expansion. Our `compute_relative_file_path` uses
    /// `original_file()` to trace back to the source file containing the
    /// macro invocation.
    #[test]
    fn test_macro_generated_symbol_has_file_path() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
macro_rules! make_fn {
    ($name:ident) => {
        pub fn $name() {}
    };
}

make_fn!(generated_function);
"#,
        );
        let graph = extract_symbol_graph(db);

        // Find the generated function
        let root = &graph.crates["test_crate"];

        assert!(
            root.symbols.contains_key("generated_function"),
            "Should find macro-generated function. Found symbols: {:?}",
            root.symbols.keys().collect::<Vec<_>>()
        );

        // File path should trace back to the source file containing the macro
        // invocation, not be empty.
        let symbol = &root.symbols["generated_function"];
        assert_eq!(
            symbol.file, "lib.rs",
            "Macro-generated symbol should have file path relative to crate root"
        );
    }

    /// Verify that submodules appear in the nested Module structure, not as symbols.
    ///
    /// Modules are represented hierarchically via Module::submodules, not as
    /// Symbol entries. This prevents double-representation and matches how
    /// we actually process modules (recursively via extract_module).
    #[test]
    fn test_modules_not_in_symbols() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub mod inner {
    pub fn inner_fn() {}
}

pub fn outer_fn() {}
"#,
        );
        let graph = extract_symbol_graph(db);
        let root = &graph.crates["test_crate"];

        // The submodule should appear in the nested structure.
        assert!(
            root.submodules.contains_key("inner"),
            "Submodule 'inner' should be in submodules"
        );

        // The submodule should NOT appear as a symbol.
        assert!(
            !root.symbols.contains_key("inner"),
            "Submodule 'inner' should not be in symbols. Found: {:?}",
            root.symbols.keys().collect::<Vec<_>>()
        );

        // Regular functions should still be symbols.
        assert!(
            root.symbols.contains_key("outer_fn"),
            "Function 'outer_fn' should be in symbols"
        );

        // Inner module's function should be in the submodule's symbols.
        let inner = &root.submodules["inner"];
        assert!(
            inner.symbols.contains_key("inner_fn"),
            "Function 'inner_fn' should be in inner module's symbols"
        );
    }

    /// Test visibility extraction: only `pub` is Public, everything else is NonPublic.
    #[test]
    fn test_visibility() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub fn public_fn() {}

pub(crate) fn pub_crate_fn() {}

fn private_fn() {}

pub mod inner {
    pub(super) fn pub_super_fn() {}
}
"#,
        );
        let graph = extract_symbol_graph(db);

        let root = &graph.crates["test_crate"];

        // pub -> Public
        if let tarjanize_schemas::SymbolKind::ModuleDef { visibility, .. } =
            &root.symbols["public_fn"].kind
        {
            assert_eq!(*visibility, tarjanize_schemas::Visibility::Public);
        }

        // pub(crate) -> NonPublic
        if let tarjanize_schemas::SymbolKind::ModuleDef { visibility, .. } =
            &root.symbols["pub_crate_fn"].kind
        {
            assert_eq!(*visibility, tarjanize_schemas::Visibility::NonPublic);
        }

        // private -> NonPublic
        if let tarjanize_schemas::SymbolKind::ModuleDef { visibility, .. } =
            &root.symbols["private_fn"].kind
        {
            assert_eq!(*visibility, tarjanize_schemas::Visibility::NonPublic);
        }

        // pub(super) -> NonPublic
        let inner = &root.submodules["inner"];
        if let tarjanize_schemas::SymbolKind::ModuleDef { visibility, .. } =
            &inner.symbols["pub_super_fn"].kind
        {
            assert_eq!(*visibility, tarjanize_schemas::Visibility::NonPublic);
        }
    }

    /// Unnamed constants (`const _: T = ...`) should be skipped.
    #[test]
    fn test_unnamed_const_skipped() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
const _: () = {
    // compile-time assertion or side effect
};

const NAMED: i32 = 42;
"#,
        );
        let graph = extract_symbol_graph(db);
        let root = &graph.crates["test_crate"];

        // Named const should be present
        assert!(
            root.symbols.contains_key("NAMED"),
            "Named const should be in symbols"
        );

        // Unnamed const should not appear (no "_" key)
        assert!(
            !root.symbols.contains_key("_"),
            "Unnamed const should not be in symbols"
        );

        // Should only have one symbol
        assert_eq!(
            root.symbols.len(),
            1,
            "Should only have NAMED, not unnamed const. Found: {:?}",
            root.symbols.keys().collect::<Vec<_>>()
        );
    }

    /// Kind strings should match rust-analyzer's SymbolKind enum names.
    #[test]
    fn test_kind_strings() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub fn my_function() {}
pub struct MyStruct;
pub enum MyEnum { A }
pub union MyUnion { a: i32 }
pub trait MyTrait {}
pub type MyTypeAlias = i32;
pub const MY_CONST: i32 = 0;
pub static MY_STATIC: i32 = 0;
macro_rules! my_macro { () => {} }
"#,
        );
        let graph = extract_symbol_graph(db);
        let root = &graph.crates["test_crate"];

        let get_kind = |name: &str| -> String {
            let symbol = root.symbols.get(name).unwrap_or_else(|| {
                panic!(
                    "Symbol '{}' not found. Available: {:?}",
                    name,
                    root.symbols.keys().collect::<Vec<_>>()
                )
            });
            if let tarjanize_schemas::SymbolKind::ModuleDef { kind, .. } =
                &symbol.kind
            {
                kind.clone()
            } else {
                panic!("{} is not a ModuleDef", name);
            }
        };

        assert_eq!(get_kind("my_function"), "Function");
        assert_eq!(get_kind("MyStruct"), "Struct");
        assert_eq!(get_kind("MyEnum"), "Enum");
        assert_eq!(get_kind("MyUnion"), "Union");
        assert_eq!(get_kind("MyTrait"), "Trait");
        assert_eq!(get_kind("MyTypeAlias"), "TypeAlias");
        assert_eq!(get_kind("MY_CONST"), "Const");
        assert_eq!(get_kind("MY_STATIC"), "Static");
        assert_eq!(get_kind("my_macro"), "Macro");
    }
}
