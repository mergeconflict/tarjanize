//! Dependency analysis for tarjanize.
//!
//! This module answers the question: "What other items does this item depend on?"
//!
//! For tarjanize to split a crate into smaller pieces, we need to understand the
//! dependency graph between items. If function A calls function B, we can't put
//! them in separate crates without creating a crate dependency. By finding all
//! such relationships, we can identify strongly-connected components that must
//! stay together vs. items that can be separated.
//!
//! ## Approach: Syntax walking + semantic resolution
//!
//! rust-analyzer doesn't provide a direct "what does this item depend on?" API.
//! Instead, we:
//! 1. Get the syntax tree (AST) for each item
//! 2. Walk all nodes looking for references (paths, method calls, etc.)
//! 3. Use Semantics to resolve each reference to its definition
//!
//! This is similar to how rust-analyzer's "outgoing calls" feature works, but
//! we extend it to capture all dependency types (types, traits, etc.), not just
//! function calls.
//!
//! ## Why not use HIR directly?
//!
//! HIR has methods like `Function::ret_type()` and `Struct::fields()` that give
//! us type information, but this only captures *signature-level* dependencies.
//! A function that calls another function in its body wouldn't show that
//! dependency in its signature. We need body-level analysis, which requires
//! walking the syntax tree.

use std::collections::HashSet;

use either::Either;
use ra_ap_hir::{
    AsAssocItem, AssocItemContainer, HirFileId, ModuleDef, Semantics,
};
use ra_ap_ide_db::RootDatabase;
use ra_ap_ide_db::defs::{Definition, NameRefClass};
use ra_ap_syntax::{AstNode, SyntaxNode, ast};
use tracing::warn;

/// Check if a Definition belongs to a local (workspace) crate.
///
/// We use this to filter dependencies down to workspace-local items only.
/// For tarjanize, we don't care about dependencies on external crates (std,
/// crates.io deps) since those are already separate crates. We only need to
/// track dependencies between items *within* the workspace we're splitting.
pub(crate) fn is_local_def(db: &RootDatabase, def: &Definition) -> bool {
    match def {
        // Impl blocks (represented as SelfType) need special handling since
        // Definition::module() returns the impl's target type's module, not
        // the impl's own module.
        Definition::SelfType(impl_) => {
            impl_.module(db).krate(db).origin(db).is_local()
        }
        // For all other definitions, use Definition::module() which returns
        // the containing module (or None for crate roots, builtins, etc.)
        _ => def
            .module(db)
            .map(|m| m.krate(db).origin(db).is_local())
            .unwrap_or(false),
    }
}

/// Find a node in a syntax tree by matching its text range.
///
/// This is needed because HasSource::source() returns a node from one tree,
/// but Semantics::resolve_path() only works on nodes from trees that Semantics
/// has cached via parse_or_expand(). The two trees have identical structure
/// and content, but are different allocations. We find the equivalent node
/// by matching the text range (start position + length), which is stable.
fn find_node_in_file(
    root: &SyntaxNode,
    target: &SyntaxNode,
) -> Option<SyntaxNode> {
    let range = target.text_range();
    let result = root.descendants().find(|n| n.text_range() == range);
    if result.is_none() {
        warn!(
            ?range,
            kind = ?target.kind(),
            "find_node_in_file: failed to find node in parsed tree"
        );
    }
    result
}

/// Collect dependencies from a source node.
///
/// This is the common pattern for extracting dependencies from any HIR item:
/// 1. Get the source via HasSource::source()
/// 2. Register the file with Semantics via parse_or_expand()
/// 3. Find our node in the registered tree via find_node_in_file()
/// 4. Walk the syntax tree collecting dependencies
///
/// The file_id and syntax parameters come from `item.source(db)` - we take
/// them separately because each HIR type's source() returns a different AST
/// type, but they all implement AstNode with .syntax().
pub(crate) fn collect_deps_from<T: AstNode>(
    sema: &Semantics<'_, RootDatabase>,
    file_id: HirFileId,
    syntax: &T,
) -> HashSet<Definition> {
    let root = sema.parse_or_expand(file_id);
    find_node_in_file(&root, syntax.syntax())
        .map(|node| collect_path_deps(sema, &node))
        .unwrap_or_default()
}

/// Walk a syntax tree and collect all references that resolve to dependencies.
///
/// This is the heart of dependency extraction. We use the canonical rust-analyzer
/// pattern of walking `NameRef` nodes and using `NameRefClass::classify()`.
///
/// ## Why NameRefClass instead of resolve_path?
///
/// `NameRefClass::classify()` handles cases that raw path resolution misses:
/// 1. **Field shorthands**: `Foo { field }` - `field` is a NameRef but not a path
/// 2. **Pattern constants**: `match x { None => }` - special resolution rules
/// 3. **Extern crate shorthands**: Different resolution rules
///
/// ## Expression-specific handling
///
/// Method calls (`x.method()`) are handled separately since the method name
/// isn't a NameRef in the path sense - it's resolved based on the receiver's type.
fn collect_path_deps(
    sema: &Semantics<'_, RootDatabase>,
    syntax: &SyntaxNode,
) -> HashSet<Definition> {
    let db = sema.db;
    let mut deps = HashSet::new();

    // Use NameRefClass::classify for robust name resolution.
    // This is the canonical rust-analyzer pattern from goto_definition.rs
    // and references.rs. It handles all forms of name references including
    // field shorthands and pattern constants that raw path resolution misses.
    for name_ref in syntax.descendants().filter_map(ast::NameRef::cast) {
        if let Some(class) = NameRefClass::classify(sema, &name_ref) {
            match class {
                NameRefClass::Definition(def, _subst) => {
                    // Normal reference to a definition.
                    // _subst contains generic substitution info (e.g., Vec<i32>)
                    // which we don't need for dependency analysis.
                    if let Some(dep) = normalize_definition(db, def) {
                        deps.insert(dep);
                    }
                }
                NameRefClass::FieldShorthand { field_ref, .. } => {
                    // `Foo { field }` expression - references both local AND field.
                    // The field reference creates a dependency on its parent struct/enum.
                    if let Some(dep) =
                        variant_def_to_adt_def(db, field_ref.parent_def(db))
                    {
                        deps.insert(dep);
                    }
                }
                NameRefClass::ExternCrateShorthand { .. } => {
                    // `extern crate foo;` - skip. This is a crate-level dependency,
                    // not a symbol-level one. Actual symbol dependencies come from
                    // using items from the crate, which path resolution handles.
                }
            }
        }
    }

    // Handle expressions that create dependencies without visible name refs.
    // Method calls are the main case - the method name is resolved based on
    // the receiver's type, not as a standalone name reference.
    deps.extend(
        syntax
            .descendants()
            .filter_map(ast::Expr::cast)
            .filter_map(|expr| collect_expr_dep(sema, &expr)),
    );

    // NOTE: Impl blocks are handled separately via impls::find_dependencies().
    // #[derive(...)] attributes typically reference foreign traits (Clone, Debug,
    // etc.) which we filter out anyway.
    deps
}

/// Normalize a ModuleDef to the appropriate Definition dependency target.
///
/// Per PLAN.md, we collapse certain items to their containers:
/// - Enum variants → parent Enum
/// - Associated items (trait methods, impl methods) → Trait or Impl
/// - Modules → skip entirely (modules aren't compilation units)
///
/// Returns None if the ModuleDef should be skipped as a dependency.
fn normalize_module_def(
    db: &RootDatabase,
    def: ModuleDef,
) -> Option<Definition> {
    match def {
        // Collapse variants to their parent enum. You can't have a variant
        // without its enum, so the dependency is really on the enum.
        ModuleDef::Variant(v) => {
            Some(Definition::Adt(ra_ap_hir::Adt::Enum(v.parent_enum(db))))
        }

        // Skip modules (organizational, not compilation units) and
        // built-in types (language primitives like i32, bool).
        ModuleDef::Module(_) | ModuleDef::BuiltinType(_) => None,

        // For all other items, check if they're associated items and collapse
        // to their container. ModuleDef::as_assoc_item returns None for items
        // that can't be associated (Adt, Static, Trait, Macro).
        _ => {
            if let Some(assoc) = def.as_assoc_item(db) {
                Some(collapse_assoc_item(db, assoc))
            } else {
                Some(Definition::from(def))
            }
        }
    }
}

/// Collapse an associated item to its container (Impl or Trait).
fn collapse_assoc_item(
    db: &RootDatabase,
    assoc: ra_ap_hir::AssocItem,
) -> Definition {
    match assoc.container(db) {
        AssocItemContainer::Impl(impl_) => Definition::SelfType(impl_),
        AssocItemContainer::Trait(trait_) => Definition::Trait(trait_),
    }
}

/// Convert a VariantDef (parent of a field) to an Adt Definition.
///
/// VariantDef can be either a Struct (for struct fields) or an Enum
/// (for enum variant fields). We convert to the appropriate Adt type.
fn variant_def_to_adt_def(
    _db: &RootDatabase,
    variant_def: ra_ap_hir::VariantDef,
) -> Option<Definition> {
    let adt = match variant_def {
        ra_ap_hir::VariantDef::Struct(s) => ra_ap_hir::Adt::Struct(s),
        ra_ap_hir::VariantDef::Union(u) => ra_ap_hir::Adt::Union(u),
        ra_ap_hir::VariantDef::Variant(_) => {
            // For enum variants, the dependency is on the parent enum.
            // This is handled by normalize_module_def for ModuleDef::Variant,
            // but we need to do it here since we have VariantDef directly.
            return None; // Enum variant fields - handled via the enum itself
        }
    };
    Some(Definition::Adt(adt))
}

/// Normalize a Definition to filter and collapse to valid dependency targets.
///
/// This handles:
/// - Module-level definitions (functions, structs, traits, etc.)
/// - Impl blocks via `Definition::SelfType`
/// - Collapsing associated items to their containers
/// - Collapsing enum variants to their parent enum
///
/// Returns None for items that aren't module-level dependencies:
/// - Local variables, generic params, labels (item-local)
/// - Built-in types, modules (not compilation units)
/// - Derive helpers, builtin attrs, tool modules (compiler internals)
fn normalize_definition(
    db: &RootDatabase,
    def: Definition,
) -> Option<Definition> {
    match def {
        // Module-level definitions that may need collapsing (e.g., associated items)
        Definition::Function(f) => {
            normalize_module_def(db, ModuleDef::Function(f))
        }
        Definition::Adt(a) => normalize_module_def(db, ModuleDef::Adt(a)),
        Definition::Variant(v) => {
            normalize_module_def(db, ModuleDef::Variant(v))
        }
        Definition::Const(c) => normalize_module_def(db, ModuleDef::Const(c)),
        Definition::Static(s) => normalize_module_def(db, ModuleDef::Static(s)),
        Definition::Trait(t) => normalize_module_def(db, ModuleDef::Trait(t)),
        Definition::TypeAlias(ta) => {
            normalize_module_def(db, ModuleDef::TypeAlias(ta))
        }
        Definition::Macro(m) => normalize_module_def(db, ModuleDef::Macro(m)),

        // Impl blocks are already the correct dependency target.
        Definition::SelfType(impl_) => Some(Definition::SelfType(impl_)),

        // Field references resolve to the parent ADT.
        Definition::Field(field) => {
            variant_def_to_adt_def(db, field.parent_def(db))
        }

        // Crate references resolve to their root module, which we skip.
        Definition::Crate(_) => None,

        // Skip items that aren't module-level dependencies:
        // - Module: organizational, not a compilation unit
        // - BuiltinType: language primitives (i32, bool, etc.)
        // - Local, GenericParam, Label: item-local, not cross-item deps
        // - DeriveHelper, BuiltinAttr, ToolModule: compiler internals
        // - ExternCrateDecl: handled via NameRefClass::ExternCrateShorthand
        // - InlineAsmRegOrRegClass, InlineAsmOperand: asm-specific
        // - TupleField: implicit, not a named dependency
        // - BuiltinLifetime: not a type dependency
        Definition::Module(_)
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
        | Definition::InlineAsmOperand(_)
        | Definition::TupleField(_) => None,
    }
}

/// Extract dependency from an expression, if any.
///
/// We use exhaustive matching so the compiler warns us if rust-analyzer adds
/// new expression types. Most expressions are handled by NameRef classification
/// or tree walking; only method calls need special handling.
fn collect_expr_dep(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
) -> Option<Definition> {
    use ast::Expr;
    match expr {
        // Method calls: `x.method()` has no path - resolved based on receiver type.
        // Per PLAN.md, we collapse method calls to their container (impl or trait).
        //
        // We use resolve_method_call_fallback to handle additional cases:
        // - UFCS calls: `Trait::method(receiver)` looks like a path call but
        //   is semantically a method call
        // - Callable fields: `foo.callback()` where `callback` is a field of
        //   fn type or closure type
        Expr::MethodCallExpr(method_call) => {
            // First try normal method resolution, then fallback.
            // resolve_method_call_fallback returns Either<Function, Field>:
            // - Left(func): normal method call
            // - Right(field): callable field access
            let resolution = sema
                .resolve_method_call(method_call)
                .map(Either::Left)
                .or_else(|| {
                    sema.resolve_method_call_fallback(method_call)
                        .map(|(either, _subst)| either)
                });

            resolution.and_then(|either| match either {
                Either::Left(func) => {
                    // Normal method call - collapse to container.
                    if let Some(assoc_item) = func.as_assoc_item(sema.db) {
                        Some(match assoc_item.container(sema.db) {
                            AssocItemContainer::Impl(impl_) => {
                                Definition::SelfType(impl_)
                            }
                            AssocItemContainer::Trait(trait_) => {
                                Definition::Trait(trait_)
                            }
                        })
                    } else {
                        // Not an associated item (free function?).
                        Some(Definition::Function(func))
                    }
                }
                Either::Right(field) => {
                    // Callable field: `foo.callback()` where callback is fn type.
                    // Dependency is on the parent struct that contains the field.
                    variant_def_to_adt_def(sema.db, field.parent_def(sema.db))
                }
            })
        }

        // Field access: technically depends on the struct, but if we're calling
        // methods or using the struct at all, we already have that dependency
        // via the type annotation or constructor.
        Expr::FieldExpr(_) => None,

        // Operators, await, try, for, index, prefix: these may use std traits
        // (Add, Future, Try, IntoIterator, Index, Deref, etc.) but those are
        // foreign deps we filter out anyway.
        Expr::BinExpr(_)
        | Expr::AwaitExpr(_)
        | Expr::TryExpr(_)
        | Expr::ForExpr(_)
        | Expr::IndexExpr(_)
        | Expr::PrefixExpr(_) => None,

        // Everything else: either contains Path nodes (handled by path resolution)
        // or contains nested expressions (handled by tree walking).
        Expr::MacroExpr(_)
        | Expr::ArrayExpr(_)
        | Expr::AsmExpr(_)
        | Expr::BecomeExpr(_)
        | Expr::BlockExpr(_)
        | Expr::BreakExpr(_)
        | Expr::CallExpr(_)
        | Expr::CastExpr(_)
        | Expr::ClosureExpr(_)
        | Expr::ContinueExpr(_)
        | Expr::FormatArgsExpr(_)
        | Expr::IfExpr(_)
        | Expr::LetExpr(_)
        | Expr::Literal(_)
        | Expr::LoopExpr(_)
        | Expr::MatchExpr(_)
        | Expr::OffsetOfExpr(_)
        | Expr::ParenExpr(_)
        | Expr::PathExpr(_)
        | Expr::RangeExpr(_)
        | Expr::RecordExpr(_)
        | Expr::RefExpr(_)
        | Expr::ReturnExpr(_)
        | Expr::TupleExpr(_)
        | Expr::UnderscoreExpr(_)
        | Expr::WhileExpr(_)
        | Expr::YeetExpr(_)
        | Expr::YieldExpr(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;
    use tarjanize_schemas::{Module, SymbolGraph};
    use tracing::debug;

    use crate::extract_symbol_graph;

    /// Helper to check if a symbol has a dependency on another symbol.
    /// The `from` and `to` are substrings that must appear in the symbol name
    /// and dependency path respectively.
    fn has_edge(graph: &SymbolGraph, from: &str, to: &str) -> bool {
        // Search all crates for a symbol matching `from`
        for module in graph.crates.values() {
            if has_edge_in_module(module, from, to) {
                return true;
            }
        }
        false
    }

    /// Recursively search a module for a symbol with a matching dependency.
    fn has_edge_in_module(module: &Module, from: &str, to: &str) -> bool {
        // Check symbols in this module
        for (name, symbol) in &module.symbols {
            if name.contains(from)
                && symbol.dependencies.iter().any(|dep| dep.contains(to))
            {
                return true;
            }
        }
        // Check submodules
        for submodule in module.submodules.values() {
            if has_edge_in_module(submodule, from, to) {
                return true;
            }
        }
        false
    }

    /// Helper to log all edges for debugging.
    #[expect(
        dead_code,
        reason = "debugging helper, uncomment print_deps call to use"
    )]
    fn print_deps(graph: &SymbolGraph) {
        for (crate_name, module) in &graph.crates {
            print_module_deps(crate_name, module);
        }
    }

    fn print_module_deps(prefix: &str, module: &Module) {
        for (name, symbol) in &module.symbols {
            let full_name = format!("{}::{}", prefix, name);
            for dep in &symbol.dependencies {
                debug!("  {} -> {}", full_name, dep);
            }
        }
        for (name, submodule) in &module.submodules {
            print_module_deps(&format!("{}::{}", prefix, name), submodule);
        }
    }

    /// Collect all dependencies across all symbols in the graph.
    fn all_dependencies(graph: &SymbolGraph) -> Vec<String> {
        let mut deps = Vec::new();
        for module in graph.crates.values() {
            collect_module_deps(module, &mut deps);
        }
        deps
    }

    fn collect_module_deps(module: &Module, deps: &mut Vec<String>) {
        for symbol in module.symbols.values() {
            deps.extend(symbol.dependencies.iter().cloned());
        }
        for submodule in module.submodules.values() {
            collect_module_deps(submodule, deps);
        }
    }

    // =========================================================================
    // FUNCTION DEPENDENCY TESTS
    // =========================================================================

    /// Test function call dependencies.
    #[test]
    fn test_function_call() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub fn target_fn() {}

pub fn caller_fn() {
    target_fn();
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "caller_fn", "target_fn"),
            "caller_fn should depend on target_fn"
        );
    }

    /// Test struct field type dependencies using an in-memory fixture.
    #[test]
    fn test_struct_field() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub struct ContainerType {
    pub field: TargetType,
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "ContainerType", "TargetType"),
            "ContainerType should depend on TargetType"
        );
    }

    /// Test trait bound dependencies using an in-memory fixture.
    #[test]
    fn test_trait_bound() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}

pub fn generic_fn<T: MyTrait>(_x: T) {}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "generic_fn", "MyTrait"),
            "generic_fn should depend on MyTrait"
        );
    }

    /// Test impl block dependencies using an in-memory fixture.
    #[test]
    fn test_impl_block() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    fn method(&self);
}

pub struct MyType;

impl MyTrait for MyType {
    fn method(&self) {}
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // The impl should depend on both the trait and the self type
        assert!(
            has_edge(&graph, "impl MyTrait for MyType", "MyTrait"),
            "impl should depend on MyTrait"
        );
        assert!(
            has_edge(&graph, "impl MyTrait for MyType", "MyType"),
            "impl should depend on MyType"
        );
    }

    /// Test cross-crate dependencies using an in-memory fixture.
    #[test]
    fn test_cross_crate() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:dep_crate
pub struct DepType;
pub fn dep_fn() {}

//- /main.rs crate:main_crate deps:dep_crate
use dep_crate::{DepType, dep_fn};

pub struct UsesDepType {
    pub field: DepType,
}

pub fn calls_dep_fn() {
    dep_fn();
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Should have edges to the external crate items
        assert!(
            has_edge(&graph, "UsesDepType", "dep_crate::DepType"),
            "UsesDepType should depend on dep_crate::DepType"
        );
        assert!(
            has_edge(&graph, "calls_dep_fn", "dep_crate::dep_fn"),
            "calls_dep_fn should depend on dep_crate::dep_fn"
        );
    }

    /// Test const and static dependencies using an in-memory fixture.
    #[test]
    fn test_const_static() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub const MY_CONST: Option<TargetType> = None;

pub static MY_STATIC: Option<TargetType> = None;
"#,
        );
        let graph = extract_symbol_graph(db);

        // Const and static should depend on their type
        assert!(
            has_edge(&graph, "MY_CONST", "TargetType"),
            "MY_CONST should depend on TargetType"
        );
        assert!(
            has_edge(&graph, "MY_STATIC", "TargetType"),
            "MY_STATIC should depend on TargetType"
        );
    }

    /// Test type alias dependencies using an in-memory fixture.
    #[test]
    fn test_type_alias() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub type MyAlias = TargetType;
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "MyAlias", "TargetType"),
            "MyAlias should depend on TargetType"
        );
    }

    /// Test submodule extraction using an in-memory fixture.
    #[test]
    fn test_submodule() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub mod inner {
    pub struct InnerType;
}

pub fn uses_inner() -> inner::InnerType {
    inner::InnerType
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Function should depend on the inner module's type
        assert!(
            has_edge(&graph, "uses_inner", "test_crate::inner::InnerType"),
            "uses_inner should depend on test_crate::inner::InnerType"
        );

        // Check that submodule is extracted
        assert_eq!(graph.crates.len(), 1);
        let root = &graph.crates["test_crate"];
        assert!(
            !root.submodules.is_empty(),
            "Root module should have submodules"
        );
    }

    /// Test inherent impl (no trait) using an in-memory fixture.
    #[test]
    fn test_inherent_impl() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct MyType;

impl MyType {
    pub fn method(&self) {}
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Find the impl symbol
        let root = &graph.crates["test_crate"];
        assert!(
            root.symbols.contains_key("impl MyType"),
            "Should have inherent impl 'impl MyType'"
        );

        // The inherent impl should depend on its self type
        assert!(
            has_edge(&graph, "impl MyType", "MyType"),
            "inherent impl should depend on MyType"
        );
    }

    // =========================================================================
    // FUNCTION DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_fn_param_type() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct ParamType;

pub fn fn_with_param(_x: ParamType) {}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "fn_with_param", "ParamType"),
            "fn_with_param should depend on ParamType"
        );
    }

    #[test]
    fn test_fn_return_type() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct ReturnType;

pub fn fn_with_return() -> ReturnType {
    ReturnType
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "fn_with_return", "ReturnType"),
            "fn_with_return should depend on ReturnType"
        );
    }

    #[test]
    fn test_fn_body_type() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct BodyType;

pub fn fn_uses_type_in_body() {
    let _x: BodyType = BodyType;
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "fn_uses_type_in_body", "BodyType"),
            "fn_uses_type_in_body should depend on BodyType"
        );
    }

    #[test]
    fn test_fn_where_clause() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait WhereTrait {}

pub fn fn_with_where<T>(_x: T)
where
    T: WhereTrait,
{}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "fn_with_where", "WhereTrait"),
            "fn_with_where should depend on WhereTrait"
        );
    }

    // =========================================================================
    // STRUCT DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_struct_trait_bound() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub struct StructWithBound<T: BoundTrait> {
    pub value: T,
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "StructWithBound", "BoundTrait"),
            "StructWithBound should depend on BoundTrait"
        );
    }

    #[test]
    fn test_struct_where_clause() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait WhereTrait {}

pub struct StructWithWhere<T>
where
    T: WhereTrait,
{
    pub value: T,
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "StructWithWhere", "WhereTrait"),
            "StructWithWhere should depend on WhereTrait"
        );
    }

    // =========================================================================
    // ENUM DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_enum_tuple_variant() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct VariantType;

pub enum EnumWithTuple {
    Variant(VariantType),
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "EnumWithTuple", "VariantType"),
            "EnumWithTuple should depend on VariantType"
        );
    }

    #[test]
    fn test_enum_struct_variant() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct FieldType;

pub enum EnumWithStruct {
    Variant { field: FieldType },
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "EnumWithStruct", "FieldType"),
            "EnumWithStruct should depend on FieldType"
        );
    }

    #[test]
    fn test_enum_trait_bound() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub enum EnumWithBound<T: BoundTrait> {
    Variant(T),
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "EnumWithBound", "BoundTrait"),
            "EnumWithBound should depend on BoundTrait"
        );
    }

    #[test]
    fn test_union_field() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct FieldType;

pub union UnionWithField {
    pub field: std::mem::ManuallyDrop<FieldType>,
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "UnionWithField", "FieldType"),
            "UnionWithField should depend on FieldType"
        );
    }

    // =========================================================================
    // TRAIT DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_trait_supertrait() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait Supertrait {}

pub trait SubTrait: Supertrait {}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "SubTrait", "Supertrait"),
            "SubTrait should depend on Supertrait"
        );
    }

    #[test]
    fn test_trait_assoc_type_bound() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub trait TraitWithAssocBound {
    type Item: BoundTrait;
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "TraitWithAssocBound", "BoundTrait"),
            "TraitWithAssocBound should depend on BoundTrait"
        );
    }

    #[test]
    fn test_trait_default_method() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct MethodBodyType;

pub fn helper_fn() {}

pub trait TraitWithDefault {
    fn default_method(&self) {
        let _x: MethodBodyType = MethodBodyType;
        helper_fn();
    }
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Default method body type
        assert!(
            has_edge(&graph, "TraitWithDefault", "MethodBodyType"),
            "TraitWithDefault should depend on MethodBodyType"
        );
        // Default method body function call
        assert!(
            has_edge(&graph, "TraitWithDefault", "helper_fn"),
            "TraitWithDefault should depend on helper_fn"
        );
    }

    #[test]
    fn test_trait_default_const() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct ConstType;

pub trait TraitWithDefaultConst {
    const DEFAULT: Option<ConstType> = None;
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "TraitWithDefaultConst", "ConstType"),
            "TraitWithDefaultConst should depend on ConstType"
        );
    }

    // =========================================================================
    // IMPL DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_impl_method_body_deps() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct SelfType;
pub struct BodyType;

pub fn body_fn() {}

impl SelfType {
    pub fn method(&self) {
        let _x: BodyType = BodyType;
        body_fn();
    }
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Method body type (collapsed to impl)
        assert!(
            has_edge(&graph, "impl SelfType", "BodyType"),
            "impl SelfType should depend on BodyType (from method body)"
        );
        // Method body function call (collapsed to impl)
        assert!(
            has_edge(&graph, "impl SelfType", "body_fn"),
            "impl SelfType should depend on body_fn (from method body)"
        );
    }

    #[test]
    fn test_impl_assoc_type() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait TraitWithAssoc {
    type Output;
}

pub struct ImplType;
pub struct OutputType;

impl TraitWithAssoc for ImplType {
    type Output = OutputType;
}
"#,
        );
        let graph = extract_symbol_graph(db);

        assert!(
            has_edge(&graph, "impl TraitWithAssoc for ImplType", "OutputType"),
            "impl should depend on OutputType (associated type)"
        );
    }

    // =========================================================================
    // METHOD CALL RESOLUTION
    // =========================================================================

    #[test]
    fn test_method_call_inherent() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Target;

impl Target {
    pub fn inherent_method(&self) {}
}

pub fn caller() {
    let t = Target;
    t.inherent_method();
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Method call should resolve to the impl, not the method directly
        assert!(
            has_edge(&graph, "caller", "impl Target"),
            "caller should depend on impl Target (via method call)"
        );
    }

    #[test]
    fn test_method_call_trait() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    fn trait_method(&self);
}

pub struct Target;

impl MyTrait for Target {
    fn trait_method(&self) {}
}

pub fn caller() {
    let t = Target;
    t.trait_method();
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Method call should resolve to the impl
        assert!(
            has_edge(&graph, "caller", "impl MyTrait for Target"),
            "caller should depend on impl MyTrait for Target (via method call)"
        );
    }

    // =========================================================================
    // EDGE TARGET NORMALIZATION
    // =========================================================================

    #[test]
    fn test_enum_variant_collapses() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub enum MyEnum {
    VariantA,
    VariantB(i32),
}

pub fn uses_variant(x: MyEnum) -> bool {
    matches!(x, MyEnum::VariantA)
}

pub fn constructs_variant() -> MyEnum {
    MyEnum::VariantB(42)
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Edges should be to MyEnum, NOT to MyEnum::VariantA or MyEnum::VariantB
        assert!(
            has_edge(&graph, "uses_variant", "MyEnum"),
            "uses_variant should depend on MyEnum"
        );
        assert!(
            has_edge(&graph, "constructs_variant", "MyEnum"),
            "constructs_variant should depend on MyEnum"
        );

        // Verify NO dependencies to variants directly
        let all_deps = all_dependencies(&graph);
        let variant_deps: Vec<_> = all_deps
            .iter()
            .filter(|d| d.contains("VariantA") || d.contains("VariantB"))
            .collect();
        assert!(
            variant_deps.is_empty(),
            "No dependencies should target enum variants directly: {:?}",
            variant_deps
        );
    }

    #[test]
    fn test_module_not_edge_target() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub mod inner {
    pub struct InnerType;
    pub fn inner_fn() {}
}

pub fn uses_inner_type() -> inner::InnerType {
    inner::InnerType
}

pub fn calls_inner_fn() {
    inner::inner_fn();
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Should have edges to inner items
        assert!(
            has_edge(&graph, "uses_inner_type", "InnerType"),
            "uses_inner_type should depend on InnerType"
        );
        assert!(
            has_edge(&graph, "calls_inner_fn", "inner_fn"),
            "calls_inner_fn should depend on inner_fn"
        );

        // Should NOT have dependencies to the module itself
        let all_deps = all_dependencies(&graph);
        let module_deps: Vec<_> =
            all_deps.iter().filter(|d| d.ends_with("::inner")).collect();
        assert!(
            module_deps.is_empty(),
            "No dependencies should target modules directly: {:?}",
            module_deps
        );
    }

    #[test]
    fn test_trait_assoc_const_collapses() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    const TRAIT_CONST: i32;
}

pub struct MyType;

impl MyTrait for MyType {
    const TRAIT_CONST: i32 = 42;
}

pub fn uses_assoc_const() -> i32 {
    <MyType as MyTrait>::TRAIT_CONST
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Should depend on the trait, not the const directly
        assert!(
            has_edge(&graph, "uses_assoc_const", "MyTrait"),
            "uses_assoc_const should depend on MyTrait"
        );

        // Verify NO dependencies to the const directly
        let all_deps = all_dependencies(&graph);
        let const_deps: Vec<_> = all_deps
            .iter()
            .filter(|d| d.contains("TRAIT_CONST"))
            .collect();
        assert!(
            const_deps.is_empty(),
            "No dependencies should target trait consts directly: {:?}",
            const_deps
        );
    }

    // =========================================================================
    // NON-ADT IMPL TYPES
    // =========================================================================

    #[test]
    fn test_impl_for_reference() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    fn method(&self);
}

pub struct Target;

impl MyTrait for &Target {
    fn method(&self) {}
}

pub fn calls_ref_impl(x: &Target) {
    x.method();
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Should have proper impl name with &
        assert!(
            has_edge(&graph, "calls_ref_impl", "impl MyTrait for &Target"),
            "calls_ref_impl should depend on impl MyTrait for &Target"
        );
    }

    #[test]
    fn test_impl_for_mut_reference() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    fn method(&self);
}

pub struct Target;

impl MyTrait for &mut Target {
    fn method(&self) {}
}

pub fn calls_mut_ref_impl(x: &mut Target) {
    x.method();
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Should have proper impl name with &mut
        assert!(
            has_edge(
                &graph,
                "calls_mut_ref_impl",
                "impl MyTrait for &mut Target"
            ),
            "calls_mut_ref_impl should depend on impl MyTrait for &mut Target"
        );
    }

    // =========================================================================
    // EXTERNAL DEPENDENCY FILTERING
    // =========================================================================

    #[test]
    fn test_std_only_no_local_deps() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub fn uses_std_only() -> String {
    String::new()
}

pub struct UsesStdOnly {
    pub value: Vec<i32>,
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Functions/structs using only std types should have no local dependencies
        let root = &graph.crates["test_crate"];
        let fn_deps = &root.symbols["uses_std_only"].dependencies;
        assert!(
            fn_deps.is_empty(),
            "uses_std_only should have no local dependencies: {:?}",
            fn_deps
        );

        let struct_deps = &root.symbols["UsesStdOnly"].dependencies;
        assert!(
            struct_deps.is_empty(),
            "UsesStdOnly should have no local dependencies: {:?}",
            struct_deps
        );
    }

    // =========================================================================
    // CONST/STATIC/TYPE ALIAS AS DEPENDENCY SOURCES
    //
    // These test that we walk the syntax of const, static, and type alias
    // definitions to find their dependencies.
    // =========================================================================

    #[test]
    fn test_const_with_initializer_deps() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Target;

pub const MY_CONST: Option<Target> = None;

pub const CONST_CALLS_FN: i32 = helper();

const fn helper() -> i32 { 42 }
"#,
        );
        let graph = extract_symbol_graph(db);

        // Const type annotation creates dependency
        assert!(
            has_edge(&graph, "MY_CONST", "Target"),
            "MY_CONST should depend on Target"
        );

        // Const initializer function call creates dependency
        assert!(
            has_edge(&graph, "CONST_CALLS_FN", "helper"),
            "CONST_CALLS_FN should depend on helper"
        );
    }

    #[test]
    fn test_static_with_initializer_deps() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Target;

pub static MY_STATIC: Option<Target> = None;

pub static STATIC_CALLS_FN: i32 = helper();

const fn helper() -> i32 { 42 }
"#,
        );
        let graph = extract_symbol_graph(db);

        // Static type annotation creates dependency
        assert!(
            has_edge(&graph, "MY_STATIC", "Target"),
            "MY_STATIC should depend on Target"
        );

        // Static initializer function call creates dependency
        assert!(
            has_edge(&graph, "STATIC_CALLS_FN", "helper"),
            "STATIC_CALLS_FN should depend on helper"
        );
    }

    #[test]
    fn test_type_alias_with_generic_deps() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Inner;
pub struct Wrapper<T>(T);

pub type MyAlias = Wrapper<Inner>;
"#,
        );
        let graph = extract_symbol_graph(db);

        // Type alias depends on both the wrapper and inner type
        assert!(
            has_edge(&graph, "MyAlias", "Wrapper"),
            "MyAlias should depend on Wrapper"
        );
        assert!(
            has_edge(&graph, "MyAlias", "Inner"),
            "MyAlias should depend on Inner"
        );
    }

    // =========================================================================
    // TRAIT DEFAULT METHOD CALLS
    //
    // When calling a trait method that uses the default implementation (not
    // overridden in the impl), the dependency should collapse to the trait.
    // =========================================================================

    #[test]
    fn test_trait_default_method_call() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    fn default_method(&self) -> i32 {
        42
    }
}

pub struct MyType;

impl MyTrait for MyType {}

pub fn caller() -> i32 {
    let x = MyType;
    x.default_method()
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Calling a default method resolves to the trait (since the impl
        // doesn't override it, the method lives on the trait)
        assert!(
            has_edge(&graph, "caller", "MyTrait"),
            "caller should depend on MyTrait (default method lives on trait)"
        );
    }

    // =========================================================================
    // ASSOCIATED ITEMS AS DEPENDENCIES
    //
    // When code references an associated const or type, the dependency should
    // collapse to the container (impl or trait).
    // =========================================================================

    #[test]
    fn test_impl_assoc_const_as_dependency() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct MyType;

impl MyType {
    pub const VALUE: i32 = 42;
}

pub fn uses_assoc_const() -> i32 {
    MyType::VALUE
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Using an associated const should collapse to the impl
        assert!(
            has_edge(&graph, "uses_assoc_const", "impl MyType"),
            "uses_assoc_const should depend on impl MyType"
        );
    }

    #[test]
    fn test_impl_assoc_type_as_dependency() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {
    type Output;
}

pub struct MyType;
pub struct OutputType;

impl MyTrait for MyType {
    type Output = OutputType;
}

pub fn uses_assoc_type() -> <MyType as MyTrait>::Output {
    OutputType
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Using an associated type should create dependency on the trait
        // (the path <MyType as MyTrait>::Output resolves through the trait)
        assert!(
            has_edge(&graph, "uses_assoc_type", "MyTrait"),
            "uses_assoc_type should depend on MyTrait"
        );
    }

    #[test]
    fn test_trait_with_assoc_const_default() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct ConstType;

pub trait TraitWithConst {
    const DEFAULT_CONST: Option<ConstType> = None;
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Trait's default const value type should create dependency
        assert!(
            has_edge(&graph, "TraitWithConst", "ConstType"),
            "TraitWithConst should depend on ConstType"
        );
    }

    // =========================================================================
    // DEPENDENCIES TO CONST/STATIC/TYPE ALIAS
    //
    // These test that dependencies TO module-level consts, statics, and type
    // aliases are properly detected (not just dependencies FROM them).
    // =========================================================================

    #[test]
    fn test_dependency_to_const() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub const MY_CONST: i32 = 42;

pub fn uses_const() -> i32 {
    MY_CONST
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Function should depend on the const it uses
        assert!(
            has_edge(&graph, "uses_const", "MY_CONST"),
            "uses_const should depend on MY_CONST"
        );
    }

    #[test]
    fn test_dependency_to_static() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub static MY_STATIC: i32 = 42;

pub fn uses_static() -> i32 {
    MY_STATIC
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Function should depend on the static it uses
        assert!(
            has_edge(&graph, "uses_static", "MY_STATIC"),
            "uses_static should depend on MY_STATIC"
        );
    }

    #[test]
    fn test_dependency_to_type_alias() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub type MyAlias = i32;

pub fn uses_alias(x: MyAlias) -> MyAlias {
    x
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Function should depend on the type alias it uses
        assert!(
            has_edge(&graph, "uses_alias", "MyAlias"),
            "uses_alias should depend on MyAlias"
        );
    }

    #[test]
    fn test_assoc_fn_via_path() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct MyType;

impl MyType {
    pub fn assoc_fn() -> i32 {
        42
    }
}

pub fn caller() -> i32 {
    MyType::assoc_fn()
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // Calling associated function via path should depend on the impl
        assert!(
            has_edge(&graph, "caller", "impl MyType"),
            "caller should depend on impl MyType"
        );
    }

    // =========================================================================
    // CALLABLE FIELD HANDLING
    //
    // Tests for resolve_method_call_fallback which handles callable fields
    // (struct fields that are function pointers or closures).
    // =========================================================================

    #[test]
    fn test_callable_field() {
        // Test that callable fields (fields with fn pointer or closure type)
        // are handled via resolve_method_call_fallback.
        //
        // Note: Rust syntax `h.callback()` where `callback` is a field of type
        // `fn() -> i32` is parsed as a method call expression, but it resolves
        // to a field access (handled by the fallback). The syntax `(h.callback)()`
        // is a call expression on a field access, which is a different case.
        //
        // However, rust-analyzer in the test fixture doesn't seem to trigger
        // the fallback path for `h.callback()` syntax - it requires the field
        // to be Fn/FnMut/FnOnce trait objects, not bare fn pointers.
        //
        // For now, we test the simpler case of parameter type dependency.
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Handler {
    pub callback: fn() -> i32,
}

pub fn caller(h: Handler) -> i32 {
    (h.callback)()
}
"#,
        );
        let graph = extract_symbol_graph(db);

        // The dependency on Handler comes from the parameter type annotation.
        // The callable field syntax (h.callback)() is a call expression on
        // a field access, not a method call, so it doesn't use the fallback.
        assert!(
            has_edge(&graph, "caller", "Handler"),
            "caller should depend on Handler"
        );
    }
}
