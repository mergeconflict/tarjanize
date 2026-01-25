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

use ra_ap_hir::{
    AsAssocItem, AssocItemContainer, HasSource, Impl, ModuleDef, PathResolution,
    Semantics, db::HirDatabase,
};
use ra_ap_ide_db::RootDatabase;
use ra_ap_syntax::{AstNode, SyntaxNode, ast};

/// A dependency target - either a module-level definition or an impl block.
///
/// We need this because method calls resolve to impl blocks, which are not
/// ModuleDefs. By using this enum, we can properly track dependencies on
/// impl blocks from method calls (per PLAN.md's "collapsing to containers").
#[derive(Debug, Clone)]
pub enum Dependency {
    ModuleDef(ModuleDef),
    Impl(Impl),
}

impl From<ModuleDef> for Dependency {
    fn from(def: ModuleDef) -> Self {
        Dependency::ModuleDef(def)
    }
}

impl From<Impl> for Dependency {
    fn from(impl_: Impl) -> Self {
        Dependency::Impl(impl_)
    }
}

/// Check if a Dependency belongs to a local (workspace) crate.
pub fn is_local_dep(db: &dyn HirDatabase, dep: &Dependency) -> bool {
    match dep {
        Dependency::ModuleDef(def) => is_local(db, def),
        Dependency::Impl(impl_) => {
            // An impl is local if its containing module is in a local crate.
            impl_.module(db).krate().origin(db).is_local()
        }
    }
}

/// Check if a ModuleDef belongs to a local (workspace) crate.
///
/// We use this to filter dependencies down to workspace-local items only.
/// For tarjanize, we don't care about dependencies on external crates (std,
/// crates.io deps) since those are already separate crates. We only need to
/// track dependencies between items *within* the workspace we're splitting.
pub fn is_local(db: &dyn HirDatabase, def: &ModuleDef) -> bool {
    // Every item lives in a module, and every module belongs to a crate.
    // We get the module, then check if its crate is local (workspace member)
    // vs external (crates.io, std, etc.).
    //
    // Each ModuleDef variant has a different API for getting its module:
    // - Module IS the module
    // - Everything else has a .module(db) method
    // - BuiltinType (i32, bool, etc.) has no module - it's a language primitive
    let module = match def {
        ModuleDef::Module(m) => Some(*m),
        ModuleDef::Function(f) => Some(f.module(db)),
        ModuleDef::Adt(adt) => Some(adt.module(db)),
        ModuleDef::Variant(v) => Some(v.module(db)),
        ModuleDef::Const(c) => Some(c.module(db)),
        ModuleDef::Static(s) => Some(s.module(db)),
        ModuleDef::Trait(t) => Some(t.module(db)),
        ModuleDef::TraitAlias(t) => Some(t.module(db)),
        ModuleDef::TypeAlias(t) => Some(t.module(db)),
        ModuleDef::Macro(m) => Some(m.module(db)),
        ModuleDef::BuiltinType(_) => None,
    };

    // CrateOrigin tracks where a crate came from: Local (workspace), Library
    // (crates.io), Lang (std/core/alloc), or Rustc (compiler internals).
    module
        .map(|m| m.krate().origin(db).is_local())
        .unwrap_or(false)
}

/// Find all items that a given ModuleDef depends on.
///
/// This is the core of dependency analysis. Given an item (function, struct,
/// etc.), we find every other item it references in its definition.
///
/// ## What we capture
/// - Path references: `Foo`, `bar()`, `mod::Type` - covers ~95% of dependencies
/// - Method calls: `x.method()` - requires special handling (no visible path)
///
/// ## What we skip (intentionally)
/// - Implicit std trait deps: `await` (Future), `?` (Try), `for` (IntoIterator),
///   `[]` (Index), operators (Add/Sub/etc). These resolve to std library items
///   which are foreign and filtered out anyway.
/// - `#[derive(...)]`: Creates trait deps, but derive traits are typically foreign.
/// - `macro_rules!` bodies: Token trees without hygiene context; can't resolve.
///
/// ## Collapsing to containers
///
/// Per PLAN.md, we collapse associated items to their containers:
/// - Impl methods/consts/types → the impl block
/// - Trait methods/consts/types → the trait
/// - Enum variants → the enum
///
/// This ensures items that must stay together are treated atomically.
///
/// ## Impl blocks
///
/// Impl blocks are NOT ModuleDefs - they're anonymous. Use `find_impl_dependencies`
/// for impl analysis (self type, trait, and associated items).
pub fn find_dependencies(
    sema: &Semantics<'_, RootDatabase>,
    def: ModuleDef,
) -> Vec<Dependency> {
    let db = sema.db;
    let mut deps = Vec::new();

    // For each item type, we need to:
    // 1. Get its source syntax node via HasSource::source()
    // 2. Register the file with Semantics via parse_or_expand()
    // 3. Find our node in the registered tree via find_node_in_file()
    // 4. Walk the syntax tree collecting dependencies
    //
    // Why this dance? HasSource::source() returns a node from rust-analyzer's
    // internal tree, but Semantics::resolve_path() only works on nodes from
    // trees that Semantics has "seen" via parse() or parse_or_expand(). If we
    // try to resolve paths on the raw source node, we get a panic. So we:
    // - Call parse_or_expand() to register the file with Semantics
    // - Find our node in that registered tree by matching text ranges
    //
    // This is repetitive because each ModuleDef variant's source() returns a
    // different AST type (ast::Fn, ast::Struct, etc.), and Rust doesn't have
    // a trait that abstracts over "thing with syntax". A macro could reduce
    // the boilerplate, but wouldn't add clarity.
    match def {
        ModuleDef::Function(func) => {
            if let Some(source) = func.source(db) {
                let root = sema.parse_or_expand(source.file_id);
                if let Some(node) =
                    find_node_in_file(&root, source.value.syntax())
                {
                    collect_path_deps(sema, &node, &mut deps);
                }
            }
        }
        ModuleDef::Adt(adt) => match adt {
            ra_ap_hir::Adt::Struct(s) => {
                if let Some(source) = s.source(db) {
                    let root = sema.parse_or_expand(source.file_id);
                    if let Some(node) =
                        find_node_in_file(&root, source.value.syntax())
                    {
                        collect_path_deps(sema, &node, &mut deps);
                    }
                }
            }
            ra_ap_hir::Adt::Enum(e) => {
                if let Some(source) = e.source(db) {
                    let root = sema.parse_or_expand(source.file_id);
                    if let Some(node) =
                        find_node_in_file(&root, source.value.syntax())
                    {
                        collect_path_deps(sema, &node, &mut deps);
                    }
                }
            }
            ra_ap_hir::Adt::Union(u) => {
                if let Some(source) = u.source(db) {
                    let root = sema.parse_or_expand(source.file_id);
                    if let Some(node) =
                        find_node_in_file(&root, source.value.syntax())
                    {
                        collect_path_deps(sema, &node, &mut deps);
                    }
                }
            }
        },
        ModuleDef::Const(c) => {
            if let Some(source) = c.source(db) {
                let root = sema.parse_or_expand(source.file_id);
                if let Some(node) =
                    find_node_in_file(&root, source.value.syntax())
                {
                    collect_path_deps(sema, &node, &mut deps);
                }
            }
        }
        ModuleDef::Static(s) => {
            if let Some(source) = s.source(db) {
                let root = sema.parse_or_expand(source.file_id);
                if let Some(node) =
                    find_node_in_file(&root, source.value.syntax())
                {
                    collect_path_deps(sema, &node, &mut deps);
                }
            }
        }
        ModuleDef::Trait(t) => {
            if let Some(source) = t.source(db) {
                let root = sema.parse_or_expand(source.file_id);
                if let Some(node) =
                    find_node_in_file(&root, source.value.syntax())
                {
                    collect_path_deps(sema, &node, &mut deps);
                }
            }
        }
        ModuleDef::TypeAlias(t) => {
            if let Some(source) = t.source(db) {
                let root = sema.parse_or_expand(source.file_id);
                if let Some(node) =
                    find_node_in_file(&root, source.value.syntax())
                {
                    collect_path_deps(sema, &node, &mut deps);
                }
            }
        }
        ModuleDef::TraitAlias(t) => {
            if let Some(source) = t.source(db) {
                let root = sema.parse_or_expand(source.file_id);
                if let Some(node) =
                    find_node_in_file(&root, source.value.syntax())
                {
                    collect_path_deps(sema, &node, &mut deps);
                }
            }
        }

        // Modules are containers, not items with dependencies. We analyze their
        // contents individually. `pub use` re-exports don't create dependencies
        // for tarjanize purposes - they're just visibility aliases.
        ModuleDef::Module(_) => {}

        // Enum variants are analyzed as part of their parent Enum - we don't
        // need variant-level granularity since variants can't be split from
        // their enum anyway.
        ModuleDef::Variant(_) => {}

        // macro_rules! bodies are token trees without hygiene context - we can't
        // reliably resolve paths in them. Proc macros are separate crates.
        // Macro *invocations* are handled: rust-analyzer expands them and we
        // analyze the expanded code.
        ModuleDef::Macro(_) => {}

        // Builtin types (i32, bool, str, char, etc.) are language primitives.
        // They have no source code and no dependencies - they just exist.
        ModuleDef::BuiltinType(_) => {}
    }

    // Remove duplicates. A function might reference the same type multiple
    // times (e.g., in parameter and return position), but we only need to
    // record the dependency once.
    deps.sort_by_key(|d| format!("{:?}", d));
    deps.dedup_by_key(|d| format!("{:?}", d));
    deps
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
    root.descendants().find(|n| n.text_range() == range)
}

/// Walk a syntax tree and collect all references that resolve to dependencies.
///
/// This is the heart of dependency extraction. We use two strategies:
///
/// 1. **Path resolution**: Most references in Rust are paths (`foo::Bar`,
///    `some_fn()`, `Type`). Walking all Path nodes and resolving them catches
///    function calls, type annotations, trait bounds, use statements, pattern
///    matching, etc. This handles ~95% of dependencies.
///
///    Importantly, `descendants()` walks the ENTIRE syntax tree, so paths nested
///    inside patterns (`Some(x)`) and types (`Vec<Foo>`) are found and resolved.
///
/// 2. **Expression-specific handling**: Some expressions create dependencies
///    without a visible path. Method calls (`x.method()`) are the main example -
///    the method name isn't a path, it's resolved based on the receiver's type.
fn collect_path_deps(
    sema: &Semantics<'_, RootDatabase>,
    syntax: &SyntaxNode,
    deps: &mut Vec<Dependency>,
) {
    // Resolve all Path nodes. This is the workhorse that catches most
    // dependencies: function calls, type references, trait bounds, use
    // statements, enum variants in patterns, const references, etc.
    //
    // Because descendants() walks the entire tree, this also handles paths
    // nested inside patterns (e.g., `Some` in `Some(x)`) and types (e.g.,
    // `Vec` and `T` in `Vec<T>`).
    for path in syntax.descendants().filter_map(ast::Path::cast) {
        if let Some(resolution) = sema.resolve_path(&path) {
            // PathResolution tells us what a path refers to. It can be:
            // - Def(ModuleDef): An item (function, struct, trait, const, etc.)
            // - Local: A local variable (`let x = 1; x` - the second `x`)
            // - TypeParam: A generic parameter (`fn foo<T>(x: T)` - the `T` in `x: T`)
            // - ConstParam: A const generic (`fn foo<const N: usize>()`)
            // - SelfType: `Self` inside an impl block
            // - BuiltinAttr/ToolModule/DeriveHelper: Attribute-related
            //
            // For dependency analysis, we only care about Def - actual items that
            // live at module scope. The others are either local to this item
            // (Local, TypeParam, ConstParam, SelfType) or compiler internals.
            if let PathResolution::Def(module_def) = resolution {
                deps.push(module_def.into());
            }
        }
    }

    // Handle expressions that create dependencies without visible paths.
    for expr in syntax.descendants().filter_map(ast::Expr::cast) {
        collect_expr_deps(sema, &expr, deps);
    }

    // NOTE: Impl blocks are handled separately via find_impl_dependencies().
    // #[derive(...)] attributes typically reference foreign traits (Clone, Debug,
    // etc.) which we filter out anyway.
}

/// Exhaustively match on Expr variants to collect dependencies.
///
/// We use exhaustive matching so the compiler warns us if rust-analyzer adds
/// new expression types. Most expressions are handled by path resolution or
/// tree walking; only method calls need special handling.
fn collect_expr_deps(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
    deps: &mut Vec<Dependency>,
) {
    use ast::Expr;
    match expr {
        // Method calls: `x.method()` has no path - resolved based on receiver type.
        // Per PLAN.md, we collapse method calls to their container (impl or trait).
        Expr::MethodCallExpr(method_call) => {
            if let Some(func) = sema.resolve_method_call(method_call) {
                // Check if this is an associated item and collapse to container.
                if let Some(assoc_item) = func.as_assoc_item(sema.db) {
                    match assoc_item.container(sema.db) {
                        AssocItemContainer::Impl(impl_) => {
                            // For impl methods, record the impl itself as a
                            // dependency. This is now possible because Dependency
                            // can represent Impls.
                            deps.push(impl_.into());
                        }
                        AssocItemContainer::Trait(trait_) => {
                            // For trait methods (including default methods),
                            // collapse to the trait.
                            deps.push(ModuleDef::Trait(trait_).into());
                        }
                    }
                } else {
                    // Not an associated item (free function?) - record directly.
                    deps.push(ModuleDef::Function(func).into());
                }
            }
        }

        // Field access: technically depends on the struct, but if we're calling
        // methods or using the struct at all, we already have that dependency
        // via the type annotation or constructor.
        Expr::FieldExpr(_) => {}

        // Operators, await, try, for, index, prefix: these may use std traits
        // (Add, Future, Try, IntoIterator, Index, Deref, etc.) but those are
        // foreign deps we filter out anyway.
        Expr::BinExpr(_)
        | Expr::AwaitExpr(_)
        | Expr::TryExpr(_)
        | Expr::ForExpr(_)
        | Expr::IndexExpr(_)
        | Expr::PrefixExpr(_) => {}

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
        | Expr::YieldExpr(_) => {}
    }
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
pub fn find_impl_dependencies(
    db: &dyn HirDatabase,
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
pub struct ImplDependencies {
    /// Direct dependencies from the impl declaration itself (self type, trait).
    pub deps: Vec<ModuleDef>,

    /// Associated items in the impl that need their own dependency analysis.
    /// The caller should process these with `find_dependencies()`.
    pub items: Vec<ra_ap_hir::AssocItem>,
}

#[cfg(test)]
mod tests {
    use crate::extract::extract_symbol_graph;
    use crate::workspaces::load_workspace;

    /// Helper to check if an edge exists in the symbol graph.
    /// The `from` and `to` are substrings that must appear in the edge paths.
    fn has_edge(edges: &std::collections::HashSet<crate::schemas::Edge>, from: &str, to: &str) -> bool {
        edges.iter().any(|e| e.from.contains(from) && e.to.contains(to))
    }

    /// Helper to print all edges for debugging.
    #[allow(dead_code)]
    fn print_edges(edges: &std::collections::HashSet<crate::schemas::Edge>) {
        let mut sorted: Vec<_> = edges.iter().collect();
        sorted.sort_by(|a, b| (&a.from, &a.to).cmp(&(&b.from, &b.to)));
        for edge in sorted {
            println!("  {} -> {}", edge.from, edge.to);
        }
    }

    #[test]
    fn test_dependency_detection() {
        let db = load_workspace("tests/fixtures/dependency_detection")
            .expect("Failed to load fixture");
        let graph = extract_symbol_graph(&db, "dependency_detection");

        // Uncomment to debug:
        // println!("All edges:");
        // print_edges(&graph.edges);

        // =========================================================================
        // FUNCTION DEPENDENCIES
        // =========================================================================

        // Function calling another function
        assert!(
            has_edge(&graph.edges, "fn_calls_fn", "fn_target"),
            "fn_calls_fn should depend on fn_target"
        );

        // Function with parameter type
        assert!(
            has_edge(&graph.edges, "fn_param_type", "FnTargetType"),
            "fn_param_type should depend on FnTargetType"
        );

        // Function with return type
        assert!(
            has_edge(&graph.edges, "fn_return_type", "FnTargetType"),
            "fn_return_type should depend on FnTargetType"
        );

        // Function using type in body
        assert!(
            has_edge(&graph.edges, "fn_body_type", "FnTargetType"),
            "fn_body_type should depend on FnTargetType"
        );

        // Function with trait bound
        assert!(
            has_edge(&graph.edges, "fn_trait_bound", "FnBoundTrait"),
            "fn_trait_bound should depend on FnBoundTrait"
        );

        // Function with where clause
        assert!(
            has_edge(&graph.edges, "fn_where_clause", "FnWhereTrait"),
            "fn_where_clause should depend on FnWhereTrait"
        );

        // =========================================================================
        // STRUCT DEPENDENCIES
        // =========================================================================

        // Struct with field type
        assert!(
            has_edge(&graph.edges, "StructWithField", "StructFieldTarget"),
            "StructWithField should depend on StructFieldTarget"
        );

        // Struct with trait bound
        assert!(
            has_edge(&graph.edges, "StructWithBound", "StructBoundTrait"),
            "StructWithBound should depend on StructBoundTrait"
        );

        // Struct with where clause
        assert!(
            has_edge(&graph.edges, "StructWithWhere", "StructWhereTrait"),
            "StructWithWhere should depend on StructWhereTrait"
        );

        // =========================================================================
        // ENUM DEPENDENCIES
        // =========================================================================

        // Enum with tuple variant
        assert!(
            has_edge(&graph.edges, "EnumWithTupleVariant", "EnumVariantTarget"),
            "EnumWithTupleVariant should depend on EnumVariantTarget"
        );

        // Enum with struct variant
        assert!(
            has_edge(&graph.edges, "EnumWithStructVariant", "EnumVariantTarget"),
            "EnumWithStructVariant should depend on EnumVariantTarget"
        );

        // Enum with trait bound
        assert!(
            has_edge(&graph.edges, "EnumWithBound", "EnumBoundTrait"),
            "EnumWithBound should depend on EnumBoundTrait"
        );

        // =========================================================================
        // TRAIT DEPENDENCIES
        // =========================================================================

        // Trait with supertrait
        assert!(
            has_edge(&graph.edges, "TraitWithSuper", "Supertrait"),
            "TraitWithSuper should depend on Supertrait"
        );

        // Trait with associated type bound
        assert!(
            has_edge(&graph.edges, "TraitWithAssocTypeBound", "AssocTypeBound"),
            "TraitWithAssocTypeBound should depend on AssocTypeBound"
        );

        // Trait with default method - type in body
        assert!(
            has_edge(&graph.edges, "TraitWithDefaultMethod", "DefaultMethodTarget"),
            "TraitWithDefaultMethod should depend on DefaultMethodTarget"
        );

        // Trait with default method - function call
        assert!(
            has_edge(&graph.edges, "TraitWithDefaultMethod", "default_method_fn_target"),
            "TraitWithDefaultMethod should depend on default_method_fn_target"
        );

        // Trait with default const
        assert!(
            has_edge(&graph.edges, "TraitWithDefaultConst", "DefaultConstTarget"),
            "TraitWithDefaultConst should depend on DefaultConstTarget"
        );

        // =========================================================================
        // IMPL DEPENDENCIES (collapsed to impl block)
        // =========================================================================

        // Inherent impl - self type
        assert!(
            has_edge(&graph.edges, "impl ImplSelfType", "ImplSelfType"),
            "impl ImplSelfType should depend on ImplSelfType"
        );

        // Inherent impl - method body type (collapsed to impl)
        assert!(
            has_edge(&graph.edges, "impl ImplSelfType", "InherentImplBodyTarget"),
            "impl ImplSelfType should depend on InherentImplBodyTarget (from method body)"
        );

        // Inherent impl - method body function call (collapsed to impl)
        assert!(
            has_edge(&graph.edges, "impl ImplSelfType", "inherent_impl_fn_target"),
            "impl ImplSelfType should depend on inherent_impl_fn_target (from method body)"
        );

        // Trait impl - self type
        assert!(
            has_edge(&graph.edges, "impl ImplTrait for TraitImplType", "TraitImplType"),
            "impl ImplTrait for TraitImplType should depend on TraitImplType"
        );

        // Trait impl - trait
        assert!(
            has_edge(&graph.edges, "impl ImplTrait for TraitImplType", "ImplTrait"),
            "impl ImplTrait for TraitImplType should depend on ImplTrait"
        );

        // Trait impl - method body type (collapsed to impl)
        assert!(
            has_edge(&graph.edges, "impl ImplTrait for TraitImplType", "TraitImplBodyTarget"),
            "impl ImplTrait for TraitImplType should depend on TraitImplBodyTarget (from method body)"
        );

        // Trait impl - method body function call (collapsed to impl)
        assert!(
            has_edge(&graph.edges, "impl ImplTrait for TraitImplType", "trait_impl_fn_target"),
            "impl ImplTrait for TraitImplType should depend on trait_impl_fn_target (from method body)"
        );

        // Impl with associated type
        assert!(
            has_edge(&graph.edges, "impl TraitWithAssocType for AssocTypeImplType", "AssocTypeOutput"),
            "impl TraitWithAssocType for AssocTypeImplType should depend on AssocTypeOutput"
        );

        // =========================================================================
        // METHOD CALL RESOLUTION
        // =========================================================================

        // Inherent method call - should resolve to impl
        assert!(
            has_edge(&graph.edges, "method_call_inherent", "impl MethodCallTarget"),
            "method_call_inherent should depend on impl MethodCallTarget"
        );

        // Trait method call - should resolve to impl
        assert!(
            has_edge(&graph.edges, "method_call_trait", "impl MethodCallTrait for MethodCallTarget"),
            "method_call_trait should depend on impl MethodCallTrait for MethodCallTarget"
        );

        // =========================================================================
        // CONST AND STATIC DEPENDENCIES
        // =========================================================================

        // Const with type
        assert!(
            has_edge(&graph.edges, "CONST_WITH_TYPE", "ConstTypeTarget"),
            "CONST_WITH_TYPE should depend on ConstTypeTarget"
        );

        // Static with type
        assert!(
            has_edge(&graph.edges, "STATIC_WITH_TYPE", "StaticTypeTarget"),
            "STATIC_WITH_TYPE should depend on StaticTypeTarget"
        );

        // =========================================================================
        // TYPE ALIAS DEPENDENCIES
        // =========================================================================

        // Type alias
        assert!(
            has_edge(&graph.edges, "AliasOfTarget", "TypeAliasTarget"),
            "AliasOfTarget should depend on TypeAliasTarget"
        );

        // Note: Type alias bounds are not enforced by Rust, so we may or may not
        // detect AliasWithBound -> TypeAliasBoundTrait. This is a known limitation.

        // =========================================================================
        // EXTERNAL DEPENDENCY FILTERING
        // =========================================================================

        // Functions using only std types should have no edges to local items
        // (std dependencies are filtered out)
        let fn_std_edges: Vec<_> = graph.edges.iter()
            .filter(|e| e.from.contains("fn_uses_std_only"))
            .collect();
        assert!(
            fn_std_edges.is_empty(),
            "fn_uses_std_only should have no local dependencies, but found: {:?}",
            fn_std_edges
        );

        let struct_std_edges: Vec<_> = graph.edges.iter()
            .filter(|e| e.from.contains("StructUsesStdOnly"))
            .collect();
        assert!(
            struct_std_edges.is_empty(),
            "StructUsesStdOnly should have no local dependencies, but found: {:?}",
            struct_std_edges
        );
    }
}
