//! `ModuleDef` extraction for tarjanize.
//!
//! This module handles extraction of `ModuleDef` items (functions, structs,
//! enums, traits, consts, statics, type aliases, macros) into the schema format.
//!
//! `ModuleDefs` are the named items declared at module scope. They contrast with
//! impl blocks, which are anonymous and handled separately in `impls.rs`.

use std::collections::HashSet;

use ra_ap_base_db::VfsPath;
use ra_ap_hir::{
    Adt, HasSource, HasVisibility, ModuleDef, Semantics,
    Visibility as HirVisibility,
};
use ra_ap_ide::TryToNav;
use ra_ap_ide_db::defs::Definition;
use ra_ap_ide_db::{RootDatabase, SymbolKind as RaSymbolKind};
use ra_ap_syntax::AstNode;
use tarjanize_schemas::{Symbol, SymbolKind, Visibility};
use tracing::debug_span;

use crate::dependencies::{collect_path_deps, is_local_def};
use crate::paths::{compute_relative_file_path, impl_path, module_def_path};

/// Returns the fully-qualified path for a Definition dependency target.
///
/// Handles the normalized Definition variants that represent valid dependency
/// targets. The key distinction from `module_def_path` is that this also handles
/// `Definition::SelfType(Impl)` for impl block dependencies - impl blocks are not
/// `ModuleDef`s but are valid dependency targets when a symbol references methods
/// or associated items from an impl.
///
/// Other Definition variants (Module, `BuiltinType`, Local, `GenericParam`, etc.)
/// should have been filtered out by `normalize_definition` in the dependency
/// collection phase, but we return None defensively if they slip through.
pub(crate) fn definition_path(
    db: &RootDatabase,
    def: &Definition,
) -> Option<String> {
    match def {
        // Impl blocks use special formatting since they're not ModuleDefs.
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

        // Everything else should have been filtered out by normalize_definition.
        // Return None rather than panicking to be defensive.
        _ => None,
    }
}

/// Extract a single `ModuleDef` as a (name, Symbol) pair.
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
    let cost = f64::from(u32::from(nav.full_range.len()));

    // Collect dependencies in this workspace for this symbol.
    let dependencies = find_dependencies(sema, def);

    // Determine the kind string using rust-analyzer's SymbolKind.
    let kind = format!("{:?}", RaSymbolKind::from_module_def(db, def));

    // Extract visibility from the ModuleDef.
    let visibility = match def.visibility(db) {
        HirVisibility::Public => Visibility::Public,
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

/// Find all items that a given `ModuleDef` depends on.
///
/// This is the core of dependency analysis for module-level definitions.
/// Given an item (function, struct, etc.), we find every other item it
/// references in its definition.
///
/// Returns fully-qualified paths to the dependencies. These have already been
/// collapsed (e.g., associated items → their impl/trait, enum variants → their
/// enum) and filtered to local (workspace) items only.
///
/// ## What we capture
/// - Path references: `Foo`, `bar()`, `mod::Type`
/// - Method calls: `x.method()` - requires special handling (no visible path)
///
/// ## Collapsing to containers
///
/// Associated items collapse to their containers:
/// - Impl methods/consts/types → the impl block
/// - Trait methods/consts/types → the trait
/// - Enum variants → the enum
pub(crate) fn find_dependencies(
    sema: &Semantics<'_, RootDatabase>,
    def: ModuleDef,
) -> HashSet<String> {
    /// Collect dependencies from any item that implements `HasSource`.
    ///
    /// We use `sema.source()` instead of `item.source(db)` because Semantics'
    /// version registers the syntax tree with its internal cache. This allows
    /// subsequent calls like `resolve_path()` to work on nodes from that tree.
    fn go<T: HasSource>(
        sema: &Semantics<'_, RootDatabase>,
        item: T,
    ) -> HashSet<Definition> {
        sema.source(item)
            .map(|src| collect_path_deps(sema, src.value.syntax()))
            .unwrap_or_default()
    }

    let db = sema.db;

    // For each item type, we get its source and collect dependencies.
    let deps = match def {
        ModuleDef::Function(f) => go(sema, f),
        ModuleDef::Adt(adt) => match adt {
            Adt::Struct(s) => go(sema, s),
            Adt::Enum(e) => go(sema, e),
            Adt::Union(u) => go(sema, u),
        },
        ModuleDef::Const(c) => go(sema, c),
        ModuleDef::Static(s) => go(sema, s),
        ModuleDef::Trait(t) => go(sema, t),
        ModuleDef::TypeAlias(t) => go(sema, t),

        // These variants have no analyzable source - return empty set.
        ModuleDef::Module(_)
        | ModuleDef::Variant(_)
        | ModuleDef::Macro(_)
        | ModuleDef::BuiltinType(_) => HashSet::new(),
    };

    // Filter to local (workspace) dependencies and convert to paths.
    deps.into_iter()
        .filter(|dep| is_local_def(db, dep))
        .filter_map(|dep| definition_path(db, &dep))
        .collect()
}

#[cfg(test)]
mod tests {
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;

    use crate::extract_symbol_graph;

    /// Verify macro-generated symbols are extracted with correct file paths.
    ///
    /// When a `macro_rules`! macro generates a function, the symbol's `HirFileId`
    /// points to the macro expansion. Our `compute_relative_file_path` uses
    /// `original_file()` to trace back to the source file containing the
    /// macro invocation.
    #[test]
    fn test_macro_generated_symbol_has_file_path() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
macro_rules! make_fn {
    ($name:ident) => {
        pub fn $name() {}
    };
}

make_fn!(generated_function);
",
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
    /// Modules are represented hierarchically via `Module::submodules`, not as
    /// Symbol entries. This prevents double-representation and matches how
    /// we actually process modules (recursively via `extract_module`).
    #[test]
    fn test_modules_not_in_symbols() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub mod inner {
    pub fn inner_fn() {}
}

pub fn outer_fn() {}
",
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

    /// Test visibility extraction: only `pub` is Public, everything else is `NonPublic`.
    #[test]
    fn test_visibility() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub fn public_fn() {}

pub(crate) fn pub_crate_fn() {}

fn private_fn() {}

pub mod inner {
    pub(super) fn pub_super_fn() {}
}
",
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
            r"
//- /lib.rs crate:test_crate
const _: () = {
    // compile-time assertion or side effect
};

const NAMED: i32 = 42;
",
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

    /// Kind strings should match rust-analyzer's `SymbolKind` enum names.
    #[test]
    fn test_kind_strings() {
        let db = RootDatabase::with_files(
            r"
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
",
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
                panic!("{name} is not a ModuleDef");
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

    // =========================================================================
    // DEFINITION PATH TESTS
    //
    // Direct unit tests for `definition_path` which converts Definition
    // variants to fully-qualified path strings.
    // =========================================================================

    use ra_ap_hir::{Crate, Module, ModuleDef, attach_db};
    use ra_ap_ide_db::defs::Definition;

    use super::definition_path;

    /// Helper to get the root module of the `test_crate` from a fixture.
    fn get_test_crate_root(db: &RootDatabase) -> Module {
        Crate::all(db)
            .into_iter()
            .find(|k| {
                k.display_name(db)
                    .is_some_and(|n| n.to_string() == "test_crate")
            })
            .expect("test_crate not found")
            .root_module(db)
    }

    /// Helper to find a declaration by name and convert to Definition.
    fn find_def(db: &RootDatabase, name: &str) -> Definition {
        let root = get_test_crate_root(db);
        for decl in root.declarations(db) {
            if decl.name(db).is_some_and(|n| n.as_str() == name) {
                return match decl {
                    ModuleDef::Function(f) => Definition::Function(f),
                    ModuleDef::Adt(a) => Definition::Adt(a),
                    ModuleDef::Const(c) => Definition::Const(c),
                    ModuleDef::Static(s) => Definition::Static(s),
                    ModuleDef::Trait(t) => Definition::Trait(t),
                    ModuleDef::TypeAlias(t) => Definition::TypeAlias(t),
                    ModuleDef::Macro(m) => Definition::Macro(m),
                    _ => panic!("Unexpected ModuleDef variant"),
                };
            }
        }
        // Also check legacy macros (macro_rules!)
        for mac in root.legacy_macros(db) {
            if mac.name(db).as_str() == name {
                return Definition::Macro(mac);
            }
        }
        panic!("Definition '{name}' not found");
    }

    /// Helper to find a nested declaration by module path and name.
    fn find_nested_def(
        db: &RootDatabase,
        module_path: &[&str],
        name: &str,
    ) -> Definition {
        let mut module = get_test_crate_root(db);
        for &segment in module_path {
            module = module
                .children(db)
                .find(|m| m.name(db).is_some_and(|n| n.as_str() == segment))
                .unwrap_or_else(|| panic!("Module '{segment}' not found"));
        }
        for decl in module.declarations(db) {
            if decl.name(db).is_some_and(|n| n.as_str() == name) {
                return match decl {
                    ModuleDef::Function(f) => Definition::Function(f),
                    _ => panic!("Expected function"),
                };
            }
        }
        panic!("Definition '{name}' not found in module");
    }

    /// Function produces correct path.
    #[test]
    fn test_definition_path_function() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub fn my_function() {}
",
        );
        let def = find_def(&db, "my_function");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::my_function".to_string())
        );
    }

    /// Struct produces correct path.
    #[test]
    fn test_definition_path_struct() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub struct MyStruct;
",
        );
        let def = find_def(&db, "MyStruct");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::MyStruct".to_string())
        );
    }

    /// Enum produces correct path.
    #[test]
    fn test_definition_path_enum() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub enum MyEnum { A }
",
        );
        let def = find_def(&db, "MyEnum");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::MyEnum".to_string())
        );
    }

    /// Trait produces correct path.
    #[test]
    fn test_definition_path_trait() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
",
        );
        let def = find_def(&db, "MyTrait");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::MyTrait".to_string())
        );
    }

    /// Type alias produces correct path.
    #[test]
    fn test_definition_path_type_alias() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub type MyAlias = i32;
",
        );
        let def = find_def(&db, "MyAlias");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::MyAlias".to_string())
        );
    }

    /// Const produces correct path.
    #[test]
    fn test_definition_path_const() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub const MY_CONST: i32 = 42;
",
        );
        let def = find_def(&db, "MY_CONST");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::MY_CONST".to_string())
        );
    }

    /// Static produces correct path.
    #[test]
    fn test_definition_path_static() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub static MY_STATIC: i32 = 42;
",
        );
        let def = find_def(&db, "MY_STATIC");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::MY_STATIC".to_string())
        );
    }

    /// Macro produces correct path.
    #[test]
    fn test_definition_path_macro() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
macro_rules! my_macro { () => {} }
",
        );
        let def = find_def(&db, "my_macro");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::my_macro".to_string())
        );
    }

    /// Inherent impl produces correct impl path format.
    ///
    /// Uses `attach_db` because `impl_path` calls `impl_.self_ty()` which
    /// requires type inference context.
    #[test]
    fn test_definition_path_inherent_impl() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub struct Target;
impl Target {
    pub fn method() {}
}
",
        );
        attach_db(&db, || {
            let root = get_test_crate_root(&db);
            let impl_ = root.impl_defs(&db).into_iter().next().expect("impl");
            let def = Definition::SelfType(impl_);

            assert_eq!(
                definition_path(&db, &def),
                Some("test_crate::impl Target".to_string())
            );
        });
    }

    /// Trait impl produces correct impl path format.
    ///
    /// Uses `attach_db` because `impl_path` calls `impl_.self_ty()` which
    /// requires type inference context.
    #[test]
    fn test_definition_path_trait_impl() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub struct Target;
pub trait MyTrait {}
impl MyTrait for Target {}
",
        );
        attach_db(&db, || {
            let root = get_test_crate_root(&db);
            let impl_ = root
                .impl_defs(&db)
                .into_iter()
                .find(|i| i.trait_(&db).is_some())
                .expect("trait impl");
            let def = Definition::SelfType(impl_);

            assert_eq!(
                definition_path(&db, &def),
                Some("test_crate::impl MyTrait for Target".to_string())
            );
        });
    }

    /// Nested module path is fully-qualified.
    #[test]
    fn test_definition_path_nested_module() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub mod outer {
    pub mod inner {
        pub fn target() {}
    }
}
",
        );
        let def = find_nested_def(&db, &["outer", "inner"], "target");
        assert_eq!(
            definition_path(&db, &def),
            Some("test_crate::outer::inner::target".to_string())
        );
    }

    /// Non-dependency Definition variants return None.
    #[test]
    fn test_definition_path_non_dependency_returns_none() {
        let db = RootDatabase::with_files(
            r"
//- /lib.rs crate:test_crate
pub fn foo() {
    let local_var = 42;
}
",
        );
        // Definition::Module should return None (modules aren't dependency targets)
        let root = get_test_crate_root(&db);
        let def = Definition::Module(root);
        assert_eq!(definition_path(&db, &def), None);
    }
}
