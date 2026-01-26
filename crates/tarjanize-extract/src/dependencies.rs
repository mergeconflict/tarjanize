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

use ra_ap_hir::db::HirDatabase;
use ra_ap_hir::{
    AsAssocItem, AssocItemContainer, HasSource, HirFileId, Impl, ModuleDef,
    PathResolution, Semantics,
};
use ra_ap_ide_db::RootDatabase;
use ra_ap_syntax::{AstNode, SyntaxNode, ast};
use tracing::warn;

use crate::modules::module_def_module;

/// A dependency target - either a module-level definition or an impl block.
///
/// We need this because method calls resolve to impl blocks, which are not
/// ModuleDefs. By using this enum, we can properly track dependencies on
/// impl blocks from method calls (per PLAN.md's "collapsing to containers").
/// Derives Copy because ModuleDef and Impl are both Copy (they're just IDs
/// wrapping integers). Derives Hash/Eq to enable deduplication via HashSet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Dependency {
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
pub(crate) fn is_local_dep(db: &dyn HirDatabase, dep: &Dependency) -> bool {
    match dep {
        Dependency::ModuleDef(def) => is_local(db, def),
        Dependency::Impl(impl_) => {
            // An impl is local if its containing module is in a local crate.
            impl_.module(db).krate(db).origin(db).is_local()
        }
    }
}

/// Check if a ModuleDef belongs to a local (workspace) crate.
///
/// We use this to filter dependencies down to workspace-local items only.
/// For tarjanize, we don't care about dependencies on external crates (std,
/// crates.io deps) since those are already separate crates. We only need to
/// track dependencies between items *within* the workspace we're splitting.
pub(crate) fn is_local(db: &dyn HirDatabase, def: &ModuleDef) -> bool {
    // CrateOrigin tracks where a crate came from: Local (workspace), Library
    // (crates.io), Lang (std/core/alloc), or Rustc (compiler internals).
    module_def_module(db, def)
        .map(|m| m.krate(db).origin(db).is_local())
        .unwrap_or(false)
}

/// Find all items that a given ModuleDef depends on.
///
/// This is the core of dependency analysis. Given an item (function, struct,
/// etc.), we find every other item it references in its definition.
///
/// ## What we capture
/// - Path references: `Foo`, `bar()`, `mod::Type`
/// - Method calls: `x.method()` - requires special handling (no visible path)
///
/// ## What we skip (intentionally)
/// - Implicit std trait deps: `await` (Future), `?` (Try), `for` (IntoIterator),
///   `[]` (Index), operators (Add/Sub/etc). These resolve to std library items
///   which are foreign and filtered out anyway.
/// - `#[derive(...)]` attributes: Not analyzed here, but the generated impl
///   blocks are captured separately via `module.impl_defs()` in `extract_symbols`.
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
///
/// # Example
///
/// ```ignore
/// // Get a function from the HIR
/// let func: ModuleDef = /* ... */;
///
/// // Find all dependencies
/// let deps = find_dependencies(&sema, func);
///
/// // Filter to local workspace items
/// for dep in deps {
///     if is_local_dep(db, &dep) {
///         println!("Depends on: {:?}", dep);
///     }
/// }
/// ```
pub fn find_dependencies(
    sema: &Semantics<'_, RootDatabase>,
    def: ModuleDef,
) -> HashSet<Dependency> {
    let db = sema.db;
    let mut deps = HashSet::new();

    // For each item type, we get its source and collect dependencies.
    // The collect_deps_from helper handles the parse_or_expand dance needed
    // to make Semantics work with the source nodes.
    //
    // Why different branches? Each ModuleDef variant's source() returns a
    // different AST type. We could use a macro, but explicit matching gives
    // clear compiler errors if rust-analyzer adds new variants.
    match def {
        ModuleDef::Function(func) => {
            if let Some(src) = func.source(db) {
                collect_deps_from(sema, src.file_id, &src.value, &mut deps);
            }
        }
        ModuleDef::Adt(adt) => match adt {
            ra_ap_hir::Adt::Struct(s) => {
                if let Some(src) = s.source(db) {
                    collect_deps_from(sema, src.file_id, &src.value, &mut deps);
                }
            }
            ra_ap_hir::Adt::Enum(e) => {
                if let Some(src) = e.source(db) {
                    collect_deps_from(sema, src.file_id, &src.value, &mut deps);
                }
            }
            ra_ap_hir::Adt::Union(u) => {
                if let Some(src) = u.source(db) {
                    collect_deps_from(sema, src.file_id, &src.value, &mut deps);
                }
            }
        },
        ModuleDef::Const(c) => {
            if let Some(src) = c.source(db) {
                collect_deps_from(sema, src.file_id, &src.value, &mut deps);
            }
        }
        ModuleDef::Static(s) => {
            if let Some(src) = s.source(db) {
                collect_deps_from(sema, src.file_id, &src.value, &mut deps);
            }
        }
        ModuleDef::Trait(t) => {
            if let Some(src) = t.source(db) {
                collect_deps_from(sema, src.file_id, &src.value, &mut deps);
            }
        }
        ModuleDef::TypeAlias(t) => {
            if let Some(src) = t.source(db) {
                collect_deps_from(sema, src.file_id, &src.value, &mut deps);
            }
        }

        // Modules are containers, not items with dependencies. We analyze their
        // contents individually. `pub use` re-exports don't create dependencies
        // for tarjanize purposes - they're just visibility aliases.
        ModuleDef::Module(_) => {}

        // Enum variants don't need separate analysis. When we analyze the parent
        // enum, we walk its entire syntax tree including all variant definitions,
        // so dependencies like `Foo` in `enum E { V(Foo) }` are captured there.
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

    // HashSet automatically deduplicates. A function might reference the same
    // type multiple times (e.g., in parameter and return position), but we
    // only need to record the dependency once.
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
fn collect_deps_from<T: AstNode>(
    sema: &Semantics<'_, RootDatabase>,
    file_id: HirFileId,
    syntax: &T,
    deps: &mut HashSet<Dependency>,
) {
    let root = sema.parse_or_expand(file_id);
    if let Some(node) = find_node_in_file(&root, syntax.syntax()) {
        collect_path_deps(sema, &node, deps);
    }
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
    deps: &mut HashSet<Dependency>,
) {
    let db = sema.db;

    // Resolve all Path nodes. This is the workhorse that catches most
    // dependencies: function calls, type references, trait bounds, use
    // statements, enum variants in patterns, const references, etc.
    //
    // Because descendants() walks the entire tree, this also handles paths
    // nested inside patterns (e.g., `Some` in `Some(x)`) and types (e.g.,
    // `Vec` and `T` in `Vec<T>`).
    for path in syntax.descendants().filter_map(ast::Path::cast) {
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
        if let Some(PathResolution::Def(module_def)) = sema.resolve_path(&path)
        {
            // Normalize the dependency: collapse variants to enums, skip
            // modules, and collapse associated items to their containers.
            if let Some(dep) = normalize_module_def(db, module_def) {
                deps.insert(dep);
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

/// Normalize a ModuleDef to the appropriate dependency target.
///
/// Per PLAN.md, we collapse certain items to their containers:
/// - Enum variants → parent Enum
/// - Associated items (trait methods, impl methods) → Trait or Impl
/// - Modules → skip entirely (modules aren't compilation units)
///
/// Returns None if the ModuleDef should be skipped as a dependency.
fn normalize_module_def(
    db: &dyn HirDatabase,
    def: ModuleDef,
) -> Option<Dependency> {
    match def {
        // Collapse variants to their parent enum. You can't have a variant
        // without its enum, so the dependency is really on the enum.
        ModuleDef::Variant(v) => {
            Some(ModuleDef::Adt(ra_ap_hir::Adt::Enum(v.parent_enum(db))).into())
        }

        // Skip modules. References to modules (like `use crate::foo;`) don't
        // create compile-time dependencies in the same way as types/functions.
        // Modules are organizational, not compilation units.
        ModuleDef::Module(_) => None,

        // For functions, constants, and type aliases, check if they're
        // associated items and collapse to their container.
        ModuleDef::Function(f) => {
            if let Some(assoc) = f.as_assoc_item(db) {
                Some(collapse_assoc_item(db, assoc))
            } else {
                Some(def.into())
            }
        }
        ModuleDef::Const(c) => {
            if let Some(assoc) = c.as_assoc_item(db) {
                Some(collapse_assoc_item(db, assoc))
            } else {
                Some(def.into())
            }
        }
        ModuleDef::TypeAlias(t) => {
            if let Some(assoc) = t.as_assoc_item(db) {
                Some(collapse_assoc_item(db, assoc))
            } else {
                Some(def.into())
            }
        }

        // These items are never associated items, record them directly.
        ModuleDef::Adt(_)
        | ModuleDef::Static(_)
        | ModuleDef::Trait(_)
        | ModuleDef::Macro(_) => Some(def.into()),

        // BuiltinTypes are language primitives (i32, bool, etc.).
        // They shouldn't create dependencies.
        ModuleDef::BuiltinType(_) => None,
    }
}

/// Collapse an associated item to its container (Impl or Trait).
fn collapse_assoc_item(
    db: &dyn HirDatabase,
    assoc: ra_ap_hir::AssocItem,
) -> Dependency {
    match assoc.container(db) {
        AssocItemContainer::Impl(impl_) => impl_.into(),
        AssocItemContainer::Trait(trait_) => ModuleDef::Trait(trait_).into(),
    }
}

/// Exhaustively match on Expr variants to collect dependencies.
///
/// We use exhaustive matching so the compiler warns us if rust-analyzer adds
/// new expression types. Most expressions are handled by path resolution or
/// tree walking; only method calls need special handling.
fn collect_expr_deps(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
    deps: &mut HashSet<Dependency>,
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
                            deps.insert(impl_.into());
                        }
                        AssocItemContainer::Trait(trait_) => {
                            // For trait methods (including default methods),
                            // collapse to the trait.
                            deps.insert(ModuleDef::Trait(trait_).into());
                        }
                    }
                } else {
                    // Not an associated item (free function?) - record directly.
                    deps.insert(ModuleDef::Function(func).into());
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
///
/// # Example
///
/// ```ignore
/// let impl_deps = find_impl_dependencies(db, impl_block);
///
/// // Process direct dependencies (self type, trait)
/// for dep in impl_deps.deps {
///     if is_local(db, &dep) {
///         // Record edge from impl to self type/trait
///     }
/// }
///
/// // Process associated items (methods need their own analysis)
/// for item in impl_deps.items {
///     let item_def: ModuleDef = item.into();
///     let method_deps = find_dependencies(&sema, item_def);
///     // ... process method dependencies
/// }
/// ```
pub(crate) fn find_impl_dependencies(
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
#[derive(Debug)]
pub(crate) struct ImplDependencies {
    /// Direct dependencies from the impl declaration itself (self type, trait).
    pub deps: Vec<ModuleDef>,

    /// Associated items in the impl that need their own dependency analysis.
    /// The caller should process these with `find_dependencies()`.
    pub items: Vec<ra_ap_hir::AssocItem>,
}

#[cfg(test)]
mod tests {
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;
    use tarjanize_schemas::Edge;
    use tracing::debug;

    use crate::extract_symbol_graph;

    /// Helper to check if an edge exists in the symbol graph.
    /// The `from` and `to` are substrings that must appear in the edge paths.
    fn has_edge(
        edges: &std::collections::HashSet<Edge>,
        from: &str,
        to: &str,
    ) -> bool {
        edges
            .iter()
            .any(|e| e.from.contains(from) && e.to.contains(to))
    }

    /// Helper to log all edges for debugging.
    #[expect(
        dead_code,
        reason = "debugging helper, uncomment print_edges call to use"
    )]
    fn print_edges(edges: &std::collections::HashSet<Edge>) {
        let mut sorted: Vec<_> = edges.iter().collect();
        sorted.sort_by(|a, b| (&a.from, &a.to).cmp(&(&b.from, &b.to)));
        for edge in sorted {
            debug!("  {} -> {}", edge.from, edge.to);
        }
    }

    // =========================================================================
    // FUNCTION DEPENDENCY TESTS
    // =========================================================================

    /// Test function call dependencies.
    #[test]
    fn test_fixture_function_call() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub fn target_fn() {}

pub fn caller_fn() {
    target_fn();
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "caller_fn", "target_fn"),
            "caller_fn should depend on target_fn"
        );
    }

    /// Test struct field type dependencies using an in-memory fixture.
    #[test]
    fn test_fixture_struct_field() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub struct ContainerType {
    pub field: TargetType,
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "ContainerType", "TargetType"),
            "ContainerType should depend on TargetType"
        );
    }

    /// Test trait bound dependencies using an in-memory fixture.
    #[test]
    fn test_fixture_trait_bound() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}

pub fn generic_fn<T: MyTrait>(_x: T) {}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "generic_fn", "MyTrait"),
            "generic_fn should depend on MyTrait"
        );
    }

    /// Test impl block dependencies using an in-memory fixture.
    #[test]
    fn test_fixture_impl_block() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // The impl should depend on both the trait and the self type
        assert!(
            has_edge(&graph.edges, "impl MyTrait for MyType", "MyTrait"),
            "impl should depend on MyTrait"
        );
        assert!(
            has_edge(&graph.edges, "impl MyTrait for MyType", "MyType"),
            "impl should depend on MyType"
        );
    }

    /// Test cross-crate dependencies using an in-memory fixture.
    #[test]
    fn test_fixture_cross_crate() {
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
        let graph = extract_symbol_graph(&db, "test_workspace");

        // Should have edges to the external crate items
        assert!(
            has_edge(&graph.edges, "UsesDepType", "dep_crate::DepType"),
            "UsesDepType should depend on dep_crate::DepType"
        );
        assert!(
            has_edge(&graph.edges, "calls_dep_fn", "dep_crate::dep_fn"),
            "calls_dep_fn should depend on dep_crate::dep_fn"
        );
    }

    /// Test const and static dependencies using an in-memory fixture.
    #[test]
    fn test_fixture_const_static() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub const MY_CONST: Option<TargetType> = None;

pub static MY_STATIC: Option<TargetType> = None;
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Const and static should depend on their type
        assert!(
            has_edge(&graph.edges, "MY_CONST", "TargetType"),
            "MY_CONST should depend on TargetType"
        );
        assert!(
            has_edge(&graph.edges, "MY_STATIC", "TargetType"),
            "MY_STATIC should depend on TargetType"
        );
    }

    /// Test type alias dependencies using an in-memory fixture.
    #[test]
    fn test_fixture_type_alias() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub type MyAlias = TargetType;
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "MyAlias", "TargetType"),
            "MyAlias should depend on TargetType"
        );
    }

    /// Test submodule extraction using an in-memory fixture.
    #[test]
    fn test_fixture_submodule() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Function should depend on the inner module's type
        assert!(
            has_edge(
                &graph.edges,
                "uses_inner",
                "test_crate::inner::InnerType"
            ),
            "uses_inner should depend on test_crate::inner::InnerType"
        );

        // Check that submodule is extracted
        assert_eq!(graph.crates.len(), 1);
        let root = &graph.crates[0];
        assert!(
            root.submodules.is_some(),
            "Root module should have submodules"
        );
    }

    /// Test inherent impl (no trait) using an in-memory fixture.
    #[test]
    fn test_fixture_inherent_impl() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct MyType;

impl MyType {
    pub fn method(&self) {}
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Find the impl symbol
        let root = &graph.crates[0];
        let has_inherent_impl =
            root.symbols.iter().any(|s| s.name == "impl MyType");
        assert!(has_inherent_impl, "Should have inherent impl 'impl MyType'");

        // The inherent impl should depend on its self type
        assert!(
            has_edge(&graph.edges, "impl MyType", "MyType"),
            "inherent impl should depend on MyType"
        );
    }

    /// Test visibility extraction using an in-memory fixture.
    #[test]
    fn test_fixture_visibility() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        let root = &graph.crates[0];

        // Find the public function
        let pub_fn = root.symbols.iter().find(|s| s.name == "public_fn");
        assert!(pub_fn.is_some(), "Should have public_fn");
        if let tarjanize_schemas::SymbolKind::ModuleDef { visibility, .. } =
            &pub_fn.unwrap().kind
        {
            assert_eq!(visibility.as_deref(), Some("pub"));
        }

        // Find the pub(crate) function
        let pub_crate = root.symbols.iter().find(|s| s.name == "pub_crate_fn");
        assert!(pub_crate.is_some(), "Should have pub_crate_fn");
        if let tarjanize_schemas::SymbolKind::ModuleDef { visibility, .. } =
            &pub_crate.unwrap().kind
        {
            assert_eq!(visibility.as_deref(), Some("pub(crate)"));
        }

        // Find the private function (should have no visibility)
        let priv_fn = root.symbols.iter().find(|s| s.name == "private_fn");
        assert!(priv_fn.is_some(), "Should have private_fn");
        if let tarjanize_schemas::SymbolKind::ModuleDef { visibility, .. } =
            &priv_fn.unwrap().kind
        {
            assert!(
                visibility.is_none(),
                "private_fn should have no visibility"
            );
        }

        // Find the pub(super) function in inner module
        let inner = root
            .submodules
            .as_ref()
            .unwrap()
            .iter()
            .find(|m| m.name == "inner");
        assert!(inner.is_some(), "Should have inner module");
        let pub_super = inner
            .unwrap()
            .symbols
            .iter()
            .find(|s| s.name == "pub_super_fn");
        assert!(pub_super.is_some(), "Should have pub_super_fn");
        if let tarjanize_schemas::SymbolKind::ModuleDef { visibility, .. } =
            &pub_super.unwrap().kind
        {
            assert_eq!(
                visibility.as_deref(),
                Some("pub(restricted)"),
                "pub_super_fn should have restricted visibility"
            );
        }
    }

    // =========================================================================
    // FUNCTION DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_fixture_fn_param_type() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct ParamType;

pub fn fn_with_param(_x: ParamType) {}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "fn_with_param", "ParamType"),
            "fn_with_param should depend on ParamType"
        );
    }

    #[test]
    fn test_fixture_fn_return_type() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct ReturnType;

pub fn fn_with_return() -> ReturnType {
    ReturnType
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "fn_with_return", "ReturnType"),
            "fn_with_return should depend on ReturnType"
        );
    }

    #[test]
    fn test_fixture_fn_body_type() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct BodyType;

pub fn fn_uses_type_in_body() {
    let _x: BodyType = BodyType;
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "fn_uses_type_in_body", "BodyType"),
            "fn_uses_type_in_body should depend on BodyType"
        );
    }

    #[test]
    fn test_fixture_fn_where_clause() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "fn_with_where", "WhereTrait"),
            "fn_with_where should depend on WhereTrait"
        );
    }

    // =========================================================================
    // STRUCT DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_fixture_struct_trait_bound() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub struct StructWithBound<T: BoundTrait> {
    pub value: T,
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "StructWithBound", "BoundTrait"),
            "StructWithBound should depend on BoundTrait"
        );
    }

    #[test]
    fn test_fixture_struct_where_clause() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "StructWithWhere", "WhereTrait"),
            "StructWithWhere should depend on WhereTrait"
        );
    }

    // =========================================================================
    // ENUM DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_fixture_enum_tuple_variant() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct VariantType;

pub enum EnumWithTuple {
    Variant(VariantType),
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "EnumWithTuple", "VariantType"),
            "EnumWithTuple should depend on VariantType"
        );
    }

    #[test]
    fn test_fixture_enum_struct_variant() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct FieldType;

pub enum EnumWithStruct {
    Variant { field: FieldType },
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "EnumWithStruct", "FieldType"),
            "EnumWithStruct should depend on FieldType"
        );
    }

    #[test]
    fn test_fixture_enum_trait_bound() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub enum EnumWithBound<T: BoundTrait> {
    Variant(T),
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "EnumWithBound", "BoundTrait"),
            "EnumWithBound should depend on BoundTrait"
        );
    }

    #[test]
    fn test_fixture_union_field() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct FieldType;

pub union UnionWithField {
    pub field: std::mem::ManuallyDrop<FieldType>,
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "UnionWithField", "FieldType"),
            "UnionWithField should depend on FieldType"
        );
    }

    // =========================================================================
    // TRAIT DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_fixture_trait_supertrait() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait Supertrait {}

pub trait SubTrait: Supertrait {}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "SubTrait", "Supertrait"),
            "SubTrait should depend on Supertrait"
        );
    }

    #[test]
    fn test_fixture_trait_assoc_type_bound() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub trait TraitWithAssocBound {
    type Item: BoundTrait;
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "TraitWithAssocBound", "BoundTrait"),
            "TraitWithAssocBound should depend on BoundTrait"
        );
    }

    #[test]
    fn test_fixture_trait_default_method() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Default method body type
        assert!(
            has_edge(&graph.edges, "TraitWithDefault", "MethodBodyType"),
            "TraitWithDefault should depend on MethodBodyType"
        );
        // Default method body function call
        assert!(
            has_edge(&graph.edges, "TraitWithDefault", "helper_fn"),
            "TraitWithDefault should depend on helper_fn"
        );
    }

    #[test]
    fn test_fixture_trait_default_const() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct ConstType;

pub trait TraitWithDefaultConst {
    const DEFAULT: Option<ConstType> = None;
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(&graph.edges, "TraitWithDefaultConst", "ConstType"),
            "TraitWithDefaultConst should depend on ConstType"
        );
    }

    // =========================================================================
    // IMPL DEPENDENCY VARIATIONS
    // =========================================================================

    #[test]
    fn test_fixture_impl_method_body_deps() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Method body type (collapsed to impl)
        assert!(
            has_edge(&graph.edges, "impl SelfType", "BodyType"),
            "impl SelfType should depend on BodyType (from method body)"
        );
        // Method body function call (collapsed to impl)
        assert!(
            has_edge(&graph.edges, "impl SelfType", "body_fn"),
            "impl SelfType should depend on body_fn (from method body)"
        );
    }

    #[test]
    fn test_fixture_impl_assoc_type() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        assert!(
            has_edge(
                &graph.edges,
                "impl TraitWithAssoc for ImplType",
                "OutputType"
            ),
            "impl should depend on OutputType (associated type)"
        );
    }

    // =========================================================================
    // METHOD CALL RESOLUTION
    // =========================================================================

    #[test]
    fn test_fixture_method_call_inherent() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Method call should resolve to the impl, not the method directly
        assert!(
            has_edge(&graph.edges, "caller", "impl Target"),
            "caller should depend on impl Target (via method call)"
        );
    }

    #[test]
    fn test_fixture_method_call_trait() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Method call should resolve to the impl
        assert!(
            has_edge(&graph.edges, "caller", "impl MyTrait for Target"),
            "caller should depend on impl MyTrait for Target (via method call)"
        );
    }

    // =========================================================================
    // EDGE TARGET NORMALIZATION
    // =========================================================================

    #[test]
    fn test_fixture_enum_variant_collapses() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Edges should be to MyEnum, NOT to MyEnum::VariantA or MyEnum::VariantB
        assert!(
            has_edge(&graph.edges, "uses_variant", "MyEnum"),
            "uses_variant should depend on MyEnum"
        );
        assert!(
            has_edge(&graph.edges, "constructs_variant", "MyEnum"),
            "constructs_variant should depend on MyEnum"
        );

        // Verify NO edges to variants directly
        let variant_edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.to.contains("VariantA") || e.to.contains("VariantB"))
            .collect();
        assert!(
            variant_edges.is_empty(),
            "No edges should target enum variants directly: {:?}",
            variant_edges
        );
    }

    #[test]
    fn test_fixture_module_not_edge_target() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Should have edges to inner items
        assert!(
            has_edge(&graph.edges, "uses_inner_type", "InnerType"),
            "uses_inner_type should depend on InnerType"
        );
        assert!(
            has_edge(&graph.edges, "calls_inner_fn", "inner_fn"),
            "calls_inner_fn should depend on inner_fn"
        );

        // Should NOT have edges to the module itself
        let module_edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.to.ends_with("::inner"))
            .collect();
        assert!(
            module_edges.is_empty(),
            "No edges should target modules directly: {:?}",
            module_edges
        );
    }

    #[test]
    fn test_fixture_trait_assoc_const_collapses() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Should depend on the trait, not the const directly
        assert!(
            has_edge(&graph.edges, "uses_assoc_const", "MyTrait"),
            "uses_assoc_const should depend on MyTrait"
        );

        // Verify NO edges to the const directly
        let const_edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.to.contains("TRAIT_CONST"))
            .collect();
        assert!(
            const_edges.is_empty(),
            "No edges should target trait consts directly: {:?}",
            const_edges
        );
    }

    // =========================================================================
    // NON-ADT IMPL TYPES
    // =========================================================================

    #[test]
    fn test_fixture_impl_for_reference() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Should have proper impl name with &
        assert!(
            has_edge(
                &graph.edges,
                "calls_ref_impl",
                "impl MyTrait for &Target"
            ),
            "calls_ref_impl should depend on impl MyTrait for &Target"
        );
    }

    #[test]
    fn test_fixture_impl_for_mut_reference() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Should have proper impl name with &mut
        assert!(
            has_edge(
                &graph.edges,
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
    fn test_fixture_std_only_no_local_deps() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Functions/structs using only std types should have no local dependencies
        let fn_edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.from.contains("uses_std_only"))
            .collect();
        assert!(
            fn_edges.is_empty(),
            "uses_std_only should have no local dependencies: {:?}",
            fn_edges
        );

        let struct_edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.from.contains("UsesStdOnly"))
            .collect();
        assert!(
            struct_edges.is_empty(),
            "UsesStdOnly should have no local dependencies: {:?}",
            struct_edges
        );
    }

    // =========================================================================
    // CONST/STATIC/TYPE ALIAS AS DEPENDENCY SOURCES
    //
    // These test that we walk the syntax of const, static, and type alias
    // definitions to find their dependencies.
    // =========================================================================

    #[test]
    fn test_fixture_const_with_initializer_deps() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Target;

pub const MY_CONST: Option<Target> = None;

pub const CONST_CALLS_FN: i32 = helper();

const fn helper() -> i32 { 42 }
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Const type annotation creates dependency
        assert!(
            has_edge(&graph.edges, "MY_CONST", "Target"),
            "MY_CONST should depend on Target"
        );

        // Const initializer function call creates dependency
        assert!(
            has_edge(&graph.edges, "CONST_CALLS_FN", "helper"),
            "CONST_CALLS_FN should depend on helper"
        );
    }

    #[test]
    fn test_fixture_static_with_initializer_deps() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Target;

pub static MY_STATIC: Option<Target> = None;

pub static STATIC_CALLS_FN: i32 = helper();

const fn helper() -> i32 { 42 }
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Static type annotation creates dependency
        assert!(
            has_edge(&graph.edges, "MY_STATIC", "Target"),
            "MY_STATIC should depend on Target"
        );

        // Static initializer function call creates dependency
        assert!(
            has_edge(&graph.edges, "STATIC_CALLS_FN", "helper"),
            "STATIC_CALLS_FN should depend on helper"
        );
    }

    #[test]
    fn test_fixture_type_alias_with_generic_deps() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Inner;
pub struct Wrapper<T>(T);

pub type MyAlias = Wrapper<Inner>;
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Type alias depends on both the wrapper and inner type
        assert!(
            has_edge(&graph.edges, "MyAlias", "Wrapper"),
            "MyAlias should depend on Wrapper"
        );
        assert!(
            has_edge(&graph.edges, "MyAlias", "Inner"),
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
    fn test_fixture_trait_default_method_call() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Calling a default method resolves to the trait (since the impl
        // doesn't override it, the method lives on the trait)
        assert!(
            has_edge(&graph.edges, "caller", "MyTrait"),
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
    fn test_fixture_impl_assoc_const_as_dependency() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Using an associated const should collapse to the impl
        assert!(
            has_edge(&graph.edges, "uses_assoc_const", "impl MyType"),
            "uses_assoc_const should depend on impl MyType"
        );
    }

    #[test]
    fn test_fixture_impl_assoc_type_as_dependency() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Using an associated type should create dependency on the trait
        // (the path <MyType as MyTrait>::Output resolves through the trait)
        assert!(
            has_edge(&graph.edges, "uses_assoc_type", "MyTrait"),
            "uses_assoc_type should depend on MyTrait"
        );
    }

    #[test]
    fn test_fixture_trait_with_assoc_const_default() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct ConstType;

pub trait TraitWithConst {
    const DEFAULT_CONST: Option<ConstType> = None;
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Trait's default const value type should create dependency
        assert!(
            has_edge(&graph.edges, "TraitWithConst", "ConstType"),
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
    fn test_fixture_dependency_to_const() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub const MY_CONST: i32 = 42;

pub fn uses_const() -> i32 {
    MY_CONST
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Function should depend on the const it uses
        assert!(
            has_edge(&graph.edges, "uses_const", "MY_CONST"),
            "uses_const should depend on MY_CONST"
        );
    }

    #[test]
    fn test_fixture_dependency_to_static() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub static MY_STATIC: i32 = 42;

pub fn uses_static() -> i32 {
    MY_STATIC
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Function should depend on the static it uses
        assert!(
            has_edge(&graph.edges, "uses_static", "MY_STATIC"),
            "uses_static should depend on MY_STATIC"
        );
    }

    #[test]
    fn test_fixture_dependency_to_type_alias() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub type MyAlias = i32;

pub fn uses_alias(x: MyAlias) -> MyAlias {
    x
}
"#,
        );
        let graph = extract_symbol_graph(&db, "test_crate");

        // Function should depend on the type alias it uses
        assert!(
            has_edge(&graph.edges, "uses_alias", "MyAlias"),
            "uses_alias should depend on MyAlias"
        );
    }

    #[test]
    fn test_fixture_assoc_fn_via_path() {
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
        let graph = extract_symbol_graph(&db, "test_crate");

        // Calling associated function via path should depend on the impl
        assert!(
            has_edge(&graph.edges, "caller", "impl MyType"),
            "caller should depend on impl MyType"
        );
    }
}
