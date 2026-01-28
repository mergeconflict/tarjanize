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

use ra_ap_hir::{AsAssocItem, AssocItemContainer, Semantics};
use ra_ap_ide_db::RootDatabase;
use ra_ap_ide_db::defs::{Definition, NameRefClass};
use ra_ap_syntax::{AstNode, SyntaxNode, ast};

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

/// Walk a syntax tree and collect all references that resolve to dependencies.
///
/// The syntax node must come from a tree registered with Semantics (via
/// `sema.source()` or `sema.parse_or_expand()`). This ensures that Semantics'
/// internal cache can map the node to semantic information for resolution.
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
pub(crate) fn collect_path_deps(
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
                NameRefClass::Definition(def, _) => {
                    // Normal reference to a definition.
                    if let Some(dep) = normalize_definition(db, def) {
                        deps.insert(dep);
                    }
                }
                NameRefClass::FieldShorthand { field_ref, .. } => {
                    // `Foo { field }` expression as shorthand for
                    // `Foo { field: field }` - references both local AND field.
                    // The field reference creates a dependency on its parent struct/enum.
                    deps.insert(variant_def_to_adt(
                        db,
                        field_ref.parent_def(db),
                    ));
                }
                NameRefClass::ExternCrateShorthand { .. } => {
                    // `extern crate foo;` - skip. This is a crate-level dependency,
                    // not a symbol-level one. Actual symbol dependencies come from
                    // using items from the crate, which path resolution handles.
                }
            }
        }
    }

    // NOTE: Method calls like `x.method()` are handled by NameRefClass::classify -
    // the method name IS a NameRef, and classify() has a MethodCallExpr match arm.
    //
    // Impl blocks are handled separately via impls::find_dependencies().
    // #[derive(...)] attributes typically reference foreign traits (Clone, Debug,
    // etc.) which we filter out anyway.
    deps
}

/// Normalize a Definition to filter and collapse to valid dependency targets.
///
/// Per PLAN.md, we collapse certain items to their containers:
/// - Enum variants → parent Enum
/// - Associated items (trait methods, impl methods) → Trait or Impl
/// - Fields → parent ADT
///
/// Returns None for items that aren't valid dependency targets:
/// - Local variables, generic params, labels (item-local)
/// - Built-in types, modules (not compilation units)
/// - Derive helpers, builtin attrs, tool modules (compiler internals)
fn normalize_definition(
    db: &RootDatabase,
    def: Definition,
) -> Option<Definition> {
    match def {
        // Function, Const, TypeAlias can be either free-standing or associated.
        // If associated, collapse to the container (Impl or Trait).
        Definition::Function(f) => Some(collapse_if_assoc(db, f, def)),
        Definition::Const(c) => Some(collapse_if_assoc(db, c, def)),
        Definition::TypeAlias(ta) => Some(collapse_if_assoc(db, ta, def)),

        // Variants collapse to their parent enum.
        Definition::Variant(v) => {
            Some(Definition::Adt(ra_ap_hir::Adt::Enum(v.parent_enum(db))))
        }

        // Fields collapse to their parent ADT (struct, union, or enum).
        Definition::Field(f) => Some(variant_def_to_adt(db, f.parent_def(db))),

        // These are valid dependency targets as-is.
        Definition::Adt(_)
        | Definition::Static(_)
        | Definition::Trait(_)
        | Definition::Macro(_)
        | Definition::SelfType(_) => Some(def),

        // Skip items that aren't module-level dependencies.
        Definition::Module(_)         // organizational unit, not a compilation unit
        | Definition::BuiltinType(_)  // language primitives (i32, bool, str, etc.)
        | Definition::BuiltinLifetime(_) // 'static
        | Definition::Local(_)        // local variable binding
        | Definition::GenericParam(_) // type/const/lifetime parameter
        | Definition::Label(_)        // loop label ('label: loop {})
        | Definition::Crate(_)        // crate root module
        | Definition::DeriveHelper(_) // #[derive] helper attribute
        | Definition::BuiltinAttr(_)  // #[cfg], #[allow], etc.
        | Definition::ToolModule(_)   // rustfmt::, clippy::, etc.
        | Definition::ExternCrateDecl(_) // extern crate statement
        | Definition::InlineAsmRegOrRegClass(_) // asm! register class
        | Definition::InlineAsmOperand(_) // asm! operand
        | Definition::TupleField(_) => None, // tuple.0, tuple.1, etc.
    }
}

/// Collapse an item to its container if it's an associated item.
///
/// Function, Const, and TypeAlias can appear either at module scope (free)
/// or inside an impl/trait block (associated). Associated items collapse to
/// their container since they can't be split independently.
fn collapse_if_assoc<T: AsAssocItem>(
    db: &RootDatabase,
    item: T,
    def: Definition,
) -> Definition {
    match item.as_assoc_item(db) {
        Some(assoc) => match assoc.container(db) {
            AssocItemContainer::Impl(impl_) => Definition::SelfType(impl_),
            AssocItemContainer::Trait(trait_) => Definition::Trait(trait_),
        },
        None => def,
    }
}

/// Convert a VariantDef (parent of a field) to an Adt Definition.
///
/// VariantDef can be Struct, Union, or Variant (enum variant with fields).
/// For enum variants, we return the parent enum since that's the actual
/// type definition we depend on.
fn variant_def_to_adt(
    db: &RootDatabase,
    variant_def: ra_ap_hir::VariantDef,
) -> Definition {
    let adt = match variant_def {
        ra_ap_hir::VariantDef::Struct(s) => ra_ap_hir::Adt::Struct(s),
        ra_ap_hir::VariantDef::Union(u) => ra_ap_hir::Adt::Union(u),
        ra_ap_hir::VariantDef::Variant(v) => {
            ra_ap_hir::Adt::Enum(v.parent_enum(db))
        }
    };
    Definition::Adt(adt)
}

#[cfg(test)]
mod tests {
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;
    use tarjanize_schemas::{Module, SymbolGraph};

    use crate::extract_symbol_graph;

    /// Create a symbol graph from a test fixture string.
    ///
    /// Combines `RootDatabase::with_files()` and `extract_symbol_graph()` into
    /// a single call for test convenience.
    fn extract_graph(fixture: &str) -> SymbolGraph {
        let db = RootDatabase::with_files(fixture);
        extract_symbol_graph(db)
    }

    /// Assert that a dependency edge exists in the graph.
    ///
    /// Panics with a descriptive message if `from` doesn't depend on `to`.
    fn assert_has_edge(graph: &SymbolGraph, from: &str, to: &str) {
        assert!(
            has_edge(graph, from, to),
            "{} should depend on {}",
            from,
            to
        );
    }

    /// Helper to check if a symbol has a dependency on another symbol.
    ///
    /// Uses `ends_with` matching for precision:
    /// - `from` matches symbol names ending with the pattern
    /// - `to` matches dependency paths ending with the pattern
    fn has_edge(graph: &SymbolGraph, from: &str, to: &str) -> bool {
        fn go(module: &Module, from: &str, to: &str) -> bool {
            // Check symbols in this module
            for (name, symbol) in &module.symbols {
                if name.ends_with(from)
                    && symbol.dependencies.iter().any(|dep| dep.ends_with(to))
                {
                    return true;
                }
            }
            // Check submodules
            for submodule in module.submodules.values() {
                if go(submodule, from, to) {
                    return true;
                }
            }
            false
        }

        // Search all crates for a symbol matching `from`
        for module in graph.crates.values() {
            if go(module, from, to) {
                return true;
            }
        }
        false
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

    /// Assert that no dependency in the graph ends with a suffix.
    ///
    /// Used to verify that certain items (modules, variants, locals, etc.)
    /// are NOT dependency targets.
    fn assert_no_edge_to(graph: &SymbolGraph, to: &str, description: &str) {
        let all_deps = all_dependencies(graph);
        let bad_deps: Vec<_> =
            all_deps.iter().filter(|d| d.ends_with(to)).collect();
        assert!(
            bad_deps.is_empty(),
            "No dependencies should target {}: {:?}",
            description,
            bad_deps
        );
    }

    // =========================================================================
    // BASIC DEPENDENCY TESTS
    //
    // Smoke tests verifying that each kind of symbol can be a dependency
    // source. Each test shows one source kind producing edges; detailed
    // testing of specific locations within each source kind is in
    // subsequent sections.
    // =========================================================================

    /// Function body calling another function creates an edge.
    #[test]
    fn test_fn_call() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn target_fn() {}

pub fn caller_fn() {
    target_fn();
}
"#,
        );
        assert_has_edge(&graph, "caller_fn", "target_fn");
    }

    /// Struct field type creates an edge.
    #[test]
    fn test_struct_field() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub struct ContainerType {
    pub field: TargetType,
}
"#,
        );
        assert_has_edge(&graph, "ContainerType", "TargetType");
    }

    /// Enum variant field type creates an edge.
    #[test]
    fn test_enum_field() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct FieldType;

pub enum MyEnum {
    Variant(FieldType),
}
"#,
        );
        assert_has_edge(&graph, "MyEnum", "FieldType");
    }

    /// Union field type creates an edge.
    #[test]
    fn test_union_field() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct FieldType;
pub union MyUnion { pub field: FieldType }
"#,
        );
        assert_has_edge(&graph, "MyUnion", "FieldType");
    }

    /// Trait supertrait creates an edge.
    #[test]
    fn test_trait_supertrait() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Supertrait {}

pub trait Subtrait: Supertrait {}
"#,
        );
        assert_has_edge(&graph, "Subtrait", "Supertrait");
    }

    /// Trait impl creates edges to both the trait and the self type.
    #[test]
    fn test_trait_impl() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}
pub struct MyType;
impl MyTrait for MyType {}
"#,
        );
        assert_has_edge(&graph, "impl MyTrait for MyType", "MyTrait");
        assert_has_edge(&graph, "impl MyTrait for MyType", "MyType");
    }

    /// Inherent impl creates an edge to the self type.
    #[test]
    fn test_inherent_impl() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct MyType;
impl MyType {}
"#,
        );
        assert_has_edge(&graph, "impl MyType", "MyType");
    }

    /// Const type annotation creates an edge.
    #[test]
    fn test_const_type() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub const MY_CONST: TargetType = TargetType;
"#,
        );
        assert_has_edge(&graph, "MY_CONST", "TargetType");
    }

    /// Static type annotation creates an edge.
    #[test]
    fn test_static_type() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub static MY_STATIC: TargetType = TargetType;
"#,
        );
        assert_has_edge(&graph, "MY_STATIC", "TargetType");
    }

    /// Type alias creates an edge.
    #[test]
    fn test_type_alias() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct TargetType;

pub type MyAlias = TargetType;
"#,
        );
        assert_has_edge(&graph, "MyAlias", "TargetType");
    }

    /// Macro invocation creates an edge.
    #[test]
    fn test_macro_invocation() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
macro_rules! my_macro {
    () => { 42 };
}

pub fn uses_macro() -> i32 {
    my_macro!()
}
"#,
        );
        assert_has_edge(&graph, "uses_macro", "my_macro");
    }

    // =========================================================================
    // FUNCTION DEPENDENCY LOCATIONS
    //
    // Tests for the different locations within a function where dependencies
    // can appear: parameter types, return types, body expressions, generic
    // bounds, and where clauses.
    // =========================================================================

    /// Function parameter type creates an edge.
    #[test]
    fn test_fn_param_type() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct ParamType;

pub fn fn_with_param(_x: ParamType) {}
"#,
        );
        assert_has_edge(&graph, "fn_with_param", "ParamType");
    }

    /// `impl Trait` argument type creates a dependency on the trait.
    #[test]
    fn test_impl_trait_arg() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr {}
pub fn takes_impl(_: impl Tr) {}
"#,
        );
        assert_has_edge(&graph, "takes_impl", "Tr");
    }

    /// Function pointer types create dependencies on param and return types.
    #[test]
    fn test_fn_ptr_type() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct A;
pub struct B;
pub fn takes_fn_ptr(_: fn(A) -> B) {}
"#,
        );
        assert_has_edge(&graph, "takes_fn_ptr", "A");
        assert_has_edge(&graph, "takes_fn_ptr", "B");
    }

    /// Function return type creates an edge.
    #[test]
    fn test_fn_return_type() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct ReturnType;

pub fn fn_with_return() -> ReturnType { ReturnType }
"#,
        );
        assert_has_edge(&graph, "fn_with_return", "ReturnType");
    }

    /// `impl Trait` return type creates a dependency on the trait.
    #[test]
    fn test_impl_trait_return() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr {}
pub struct S;
impl Tr for S {}
pub fn returns_impl() -> impl Tr { S }
"#,
        );
        assert_has_edge(&graph, "returns_impl", "Tr");
    }

    /// Nested generic types create dependencies on all types.
    #[test]
    fn test_nested_generics() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct Outer<T>(T);
pub struct Inner;
pub fn nested() -> Outer<Outer<Inner>> { todo!() }
"#,
        );
        assert_has_edge(&graph, "nested", "Outer");
        assert_has_edge(&graph, "nested", "Inner");
    }

    /// Function body expression creates an edge.
    #[test]
    fn test_fn_body() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct BodyType;

pub fn fn_uses_body() { let _x: BodyType = BodyType; }
"#,
        );
        assert_has_edge(&graph, "fn_uses_body", "BodyType");
    }

    /// Function generic bound `<T: Trait>` creates an edge.
    #[test]
    fn test_fn_generic_bound() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait MyTrait {}

pub fn generic_fn<T: MyTrait>(_x: T) {}
"#,
        );
        assert_has_edge(&graph, "generic_fn", "MyTrait");
    }

    /// Associated type constraints in bounds create dependencies.
    #[test]
    fn test_assoc_type_constraint() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait HasItem { type Item; }
pub struct ItemType;
pub fn constrained<T: HasItem<Item = ItemType>>(_: T) {}
"#,
        );
        assert_has_edge(&graph, "constrained", "HasItem");
        assert_has_edge(&graph, "constrained", "ItemType");
    }

    /// Function where clause `where T: Trait` creates an edge.
    #[test]
    fn test_fn_where_clause() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait WhereTrait {}

pub fn fn_with_where<T>(_x: T) where T: WhereTrait {}
"#,
        );
        assert_has_edge(&graph, "fn_with_where", "WhereTrait");
    }

    /// Higher-ranked trait bounds create dependencies.
    #[test]
    fn test_hrtb() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr<'a> {}
pub fn hrtb<T>(_: T) where T: for<'a> Tr<'a> {}
"#,
        );
        assert_has_edge(&graph, "hrtb", "Tr");
    }

    // =========================================================================
    // ADT DEPENDENCY LOCATIONS
    //
    // Tests for different locations within ADTs (structs, enums, unions)
    // where dependencies can appear: fields, generic bounds, where clauses.
    // Note: basic field tests are in BASIC section; these test bounds/where.
    // =========================================================================

    /// Struct generic bound `<T: Trait>` creates an edge.
    #[test]
    fn test_struct_generic_bound() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub struct StructWithBound<T: BoundTrait> { pub value: T }
"#,
        );
        assert_has_edge(&graph, "StructWithBound", "BoundTrait");
    }

    /// Struct where clause creates an edge.
    #[test]
    fn test_struct_where_clause() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait WhereTrait {}

pub struct StructWithWhere<T> where T: WhereTrait { pub value: T }
"#,
        );
        assert_has_edge(&graph, "StructWithWhere", "WhereTrait");
    }

    /// Default type parameters create dependencies.
    #[test]
    fn test_default_type_param() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct Default;
pub struct WithDefault<T = Default>(T);
"#,
        );
        assert_has_edge(&graph, "WithDefault", "Default");
    }

    /// Enum tuple variant field creates an edge.
    #[test]
    fn test_enum_tuple_variant_field() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct TupleType;

pub enum EnumWithTuple {
    Variant(TupleType),
}
"#,
        );
        assert_has_edge(&graph, "EnumWithTuple", "TupleType");
    }

    /// Enum struct variant field creates an edge.
    #[test]
    fn test_enum_struct_variant_field() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct FieldType;

pub enum EnumWithStruct {
    Variant { field: FieldType },
}
"#,
        );
        assert_has_edge(&graph, "EnumWithStruct", "FieldType");
    }

    /// Enum generic bound creates an edge.
    #[test]
    fn test_enum_generic_bound() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub enum EnumWithBound<T: BoundTrait> {
    Variant(T),
}
"#,
        );
        assert_has_edge(&graph, "EnumWithBound", "BoundTrait");
    }

    // =========================================================================
    // TRAIT DEPENDENCY LOCATIONS
    //
    // Tests for different locations within trait definitions where
    // dependencies can appear: associated type bounds, default method bodies,
    // default const values. Note: supertrait is in BASIC section.
    // =========================================================================

    /// Trait associated type bound creates an edge.
    #[test]
    fn test_trait_assoc_type_bound() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait BoundTrait {}

pub trait TraitWithAssocBound {
    type Item: BoundTrait;
}
"#,
        );
        assert_has_edge(&graph, "TraitWithAssocBound", "BoundTrait");
    }

    /// Trait default method body creates an edge.
    #[test]
    fn test_trait_default_method_body() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn helper() {}
pub trait MyTrait {
    fn default_method(&self) { helper(); }
}
"#,
        );
        assert_has_edge(&graph, "MyTrait", "helper");
    }

    /// Trait default const creates an edge.
    #[test]
    fn test_trait_default_const() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct ConstType;

pub trait TraitWithDefaultConst {
    const DEFAULT: ConstType = ConstType;
}
"#,
        );
        assert_has_edge(&graph, "TraitWithDefaultConst", "ConstType");
    }

    // =========================================================================
    // IMPL AND ITEM DEPENDENCY LOCATIONS
    //
    // Tests for dependency locations in impl blocks, consts, statics, and
    // type aliases: method bodies, associated type definitions, initializers,
    // and generic arguments.
    // =========================================================================

    /// Impl method body creates an edge (collapsed to the impl).
    #[test]
    fn test_impl_method_body() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct MyType;
pub fn helper() {}
impl MyType {
    pub fn method(&self) { helper(); }
}
"#,
        );
        assert_has_edge(&graph, "impl MyType", "helper");
    }

    /// Impl associated type definition creates an edge.
    #[test]
    fn test_impl_assoc_type_definition() {
        let graph = extract_graph(
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
        assert_has_edge(
            &graph,
            "impl TraitWithAssoc for ImplType",
            "OutputType",
        );
    }

    /// Const initializer function call creates an edge.
    #[test]
    fn test_const_initializer() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
const fn helper() -> i32 { 42 }

pub const MY_CONST: i32 = helper();
"#,
        );
        assert_has_edge(&graph, "MY_CONST", "helper");
    }

    /// Static initializer function call creates an edge.
    #[test]
    fn test_static_initializer() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
const fn helper() -> i32 { 42 }

pub static MY_STATIC: i32 = helper();
"#,
        );
        assert_has_edge(&graph, "MY_STATIC", "helper");
    }

    /// Type alias with generic arguments creates edges to all types.
    #[test]
    fn test_type_alias_generics() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct Inner;
pub struct Wrapper<T>(T);

pub type MyAlias = Wrapper<Inner>;
"#,
        );
        assert_has_edge(&graph, "MyAlias", "Wrapper");
        assert_has_edge(&graph, "MyAlias", "Inner");
    }

    // =========================================================================
    // EDGE TARGET NORMALIZATION
    //
    // Tests that references to "sub-items" (enum variants, associated items)
    // are normalized to their containers (enum, impl, trait). This collapsing
    // ensures edges point to independently-compilable units.
    //
    // Normalization happens regardless of how the item is accessed: via path
    // (Enum::Variant, Type::method), method call (t.method()), or pattern
    // match (let Variant { .. } = x).
    // =========================================================================

    // --- Enum variants normalize to Enum ---

    /// Enum variant via path normalizes to the enum.
    #[test]
    fn test_variant_via_path() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub enum MyEnum { Variant }
pub fn uses_variant() -> MyEnum { MyEnum::Variant }
"#,
        );
        assert_has_edge(&graph, "uses_variant", "MyEnum");
        assert_no_edge_to(&graph, "::Variant", "enum variants");
    }

    /// Enum variant via pattern normalizes to the enum.
    #[test]
    fn test_variant_via_pattern() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub enum MyEnum { Variant(i32) }
pub fn matcher(x: MyEnum) -> i32 {
    match x { MyEnum::Variant(n) => n }
}
"#,
        );
        assert_has_edge(&graph, "matcher", "MyEnum");
    }

    /// Field shorthand in enum variant pattern normalizes to the enum.
    #[test]
    fn test_variant_field_shorthand() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub enum MyEnum { Variant { x: i32 } }
pub fn shorthand() -> MyEnum {
    let x = 1;
    MyEnum::Variant { x }
}
"#,
        );
        assert_has_edge(&graph, "shorthand", "MyEnum");
    }

    // --- Impl associated items normalize to Impl ---

    /// Inherent method call normalizes to the impl.
    #[test]
    fn test_inherent_method_call() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct T;
impl T { pub fn method(&self) {} }
pub fn caller(t: T) { t.method(); }
"#,
        );
        assert_has_edge(&graph, "caller", "impl T");
    }

    /// Trait method call normalizes to the trait impl.
    #[test]
    fn test_trait_method_call() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr { fn method(&self); }
pub struct T;
impl Tr for T { fn method(&self) {} }
pub fn caller(t: T) { t.method(); }
"#,
        );
        assert_has_edge(&graph, "caller", "impl Tr for T");
    }

    /// Method call on &T normalizes to impl for &T.
    #[test]
    fn test_method_call_on_ref() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr { fn method(&self); }
pub struct T;
impl Tr for &T { fn method(&self) {} }
pub fn caller(t: &T) { t.method(); }
"#,
        );
        assert_has_edge(&graph, "caller", "impl Tr for &T");
    }

    /// Method call on &mut T normalizes to impl for &mut T.
    #[test]
    fn test_method_call_on_mut_ref() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr { fn method(&self); }
pub struct T;
impl Tr for &mut T { fn method(&self) {} }
pub fn caller(t: &mut T) { t.method(); }
"#,
        );
        assert_has_edge(&graph, "caller", "impl Tr for &mut T");
    }

    /// Associated function via path normalizes to the impl.
    #[test]
    fn test_assoc_fn_via_path() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct T;
impl T { pub fn f() {} }
pub fn caller() { T::f(); }
"#,
        );
        assert_has_edge(&graph, "caller", "impl T");
    }

    /// Associated const via path normalizes to the impl.
    #[test]
    fn test_assoc_const_via_path() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct T;
impl T { pub const C: i32 = 0; }
pub fn caller() -> i32 { T::C }
"#,
        );
        assert_has_edge(&graph, "caller", "impl T");
    }

    // --- Trait associated items normalize to Trait ---

    /// Calling a trait's default method normalizes to the trait.
    #[test]
    fn test_trait_default_method_call() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr { fn default_method(&self) {} }
pub struct T;
impl Tr for T {}
pub fn caller(t: T) { t.default_method(); }
"#,
        );
        assert_has_edge(&graph, "caller", "Tr");
    }

    /// Trait associated const via qualified path normalizes to the trait.
    #[test]
    fn test_trait_assoc_const_via_qualified_path() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr { const C: i32; }
pub struct T;
impl Tr for T { const C: i32 = 0; }
pub fn caller() -> i32 { <T as Tr>::C }
"#,
        );
        assert_has_edge(&graph, "caller", "Tr");
        assert_no_edge_to(&graph, "::C", "trait consts");
    }

    /// Trait associated type via qualified path normalizes to the trait.
    #[test]
    fn test_trait_assoc_type_via_qualified_path() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr { type Out; }
pub struct T;
pub struct O;
impl Tr for T { type Out = O; }
pub fn caller() -> <T as Tr>::Out { O }
"#,
        );
        assert_has_edge(&graph, "caller", "Tr");
    }

    // =========================================================================
    // OTHER RESOLUTION MECHANISMS
    //
    // Tests for various ways references get resolved that don't fit neatly
    // into the above categories: pattern destructuring, closures, async/await,
    // dyn traits, Self keyword, cross-module paths, callable fields, and
    // references to module-level items.
    // =========================================================================

    /// Pattern destructuring creates a dependency on the type.
    #[test]
    fn test_pattern_destructuring() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct S { pub x: i32 }
pub fn destruct(s: S) -> i32 { let S { x } = s; x }
"#,
        );
        assert_has_edge(&graph, "destruct", "S");
    }

    /// Struct expressions with field init shorthand.
    #[test]
    fn test_struct_init_shorthand() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct S { pub x: i32 }
pub fn caller() -> S { let x = 1; S { x } }
"#,
        );
        assert_has_edge(&graph, "caller", "S");
    }

    /// Slice patterns detect the type.
    #[test]
    fn test_slice_pattern() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct S;
pub fn caller(arr: [S; 2]) { let [a, b] = arr; }
"#,
        );
        assert_has_edge(&graph, "caller", "S");
    }

    /// Closures capture dependencies on items they reference.
    #[test]
    fn test_closure_captures() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn target() {}
pub fn caller() { let f = || target(); f(); }
"#,
        );
        assert_has_edge(&graph, "caller", "target");
    }

    /// Turbofish syntax creates dependencies.
    #[test]
    fn test_turbofish() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct S;
pub fn generic<T>() {}
pub fn caller() { generic::<S>(); }
"#,
        );
        assert_has_edge(&graph, "caller", "generic");
        assert_has_edge(&graph, "caller", "S");
    }

    /// Array repeat expressions detect the type.
    #[test]
    fn test_array_repeat() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct S;
impl S { pub const fn new() -> S { S } }
pub fn caller() -> [S; 3] { [S::new(); 3] }
"#,
        );
        assert_has_edge(&graph, "caller", "impl S");
    }

    /// Async functions and .await create dependencies.
    #[test]
    fn test_async_await() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub async fn producer() {}
pub async fn consumer() { producer().await }
"#,
        );
        assert_has_edge(&graph, "consumer", "producer");
    }

    /// Async closures create dependencies.
    #[test]
    fn test_async_closure() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub async fn target() {}
pub fn caller() { let _ = async || { target().await }; }
"#,
        );
        assert_has_edge(&graph, "caller", "target");
    }

    /// Async closure nested inside sync closure.
    ///
    /// In debug builds, rust-analyzer panics due to a bug. In release builds,
    /// we detect the dependency correctly.
    /// See: https://github.com/rust-lang/rust-analyzer/issues/21539
    ///
    /// TODO(rust-analyzer#21539): When the bug is fixed, remove should_panic.
    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "closure type is always closure")
    )]
    fn test_async_closure_nested() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub async fn target() {}
pub fn caller() { let _ = || { async || { target().await } }; }
"#,
        );
        assert_has_edge(&graph, "caller", "target");
    }

    /// dyn Trait creates a dependency on the trait.
    #[test]
    fn test_dyn_trait() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr {}
pub fn takes_dyn(_: &dyn Tr) {}
"#,
        );
        assert_has_edge(&graph, "takes_dyn", "Tr");
    }

    /// Multiple trait bounds on dyn create dependencies on all traits.
    #[test]
    fn test_dyn_multi_trait() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub trait Tr1 {}
pub trait Tr2 {}
pub fn takes_dyn(_: &(dyn Tr1 + Tr2)) {}
"#,
        );
        assert_has_edge(&graph, "takes_dyn", "Tr1");
        assert_has_edge(&graph, "takes_dyn", "Tr2");
    }

    /// Self keyword in impl resolves to the self type.
    #[test]
    fn test_self_keyword() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct T;
impl T { pub fn new() -> Self { Self } }
"#,
        );
        assert_has_edge(&graph, "impl T", "T");
    }

    /// Cross-module paths resolve to the item, not the module.
    #[test]
    fn test_cross_module_path() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub mod inner { pub struct S; }
pub fn caller() -> inner::S { inner::S }
"#,
        );
        assert_has_edge(&graph, "caller", "S");
        assert_no_edge_to(&graph, "::inner", "modules");
        assert!(!graph.crates["test_crate"].submodules.is_empty());
    }

    /// Callable fields create dependencies via struct type.
    #[test]
    fn test_callable_field() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct S { pub f: fn() }
pub fn caller(s: S) { (s.f)(); }
"#,
        );
        assert_has_edge(&graph, "caller", "S");
    }

    /// References to module-level const works.
    #[test]
    fn test_ref_to_const() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub const C: i32 = 0;
pub fn caller() -> i32 { C }
"#,
        );
        assert_has_edge(&graph, "caller", "C");
    }

    /// References to module-level static works.
    #[test]
    fn test_ref_to_static() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub static S: i32 = 0;
pub fn caller() -> i32 { S }
"#,
        );
        assert_has_edge(&graph, "caller", "S");
    }

    /// References to type alias works.
    #[test]
    fn test_ref_to_type_alias() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub type A = i32;
pub fn caller(_: A) {}
"#,
        );
        assert_has_edge(&graph, "caller", "A");
    }

    // =========================================================================
    // EDGE TARGET FILTERING
    //
    // Negative tests verifying that certain items are NOT valid edge targets.
    // These items are filtered out by normalize_definition because they're
    // either internal to a symbol (locals, labels, lifetimes), not
    // independently-compilable units (modules, generic params), or built-in
    // language features (built-in attributes).
    // =========================================================================

    /// Modules are not edge targets; edges go to items within modules.
    #[test]
    fn test_module_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub mod m { pub struct S; }
pub fn caller() -> m::S { m::S }
"#,
        );
        assert_has_edge(&graph, "caller", "S");
        assert_no_edge_to(&graph, "::m", "modules");
    }

    /// Generic type parameters are not edge targets.
    #[test]
    fn test_generic_type_param_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn generic<T>(x: T) -> T { x }
"#,
        );
        assert_no_edge_to(&graph, "::T", "generic type params");
    }

    /// Const generic parameters are not edge targets.
    #[test]
    fn test_const_generic_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct Arr<const N: usize>;
pub fn caller() -> Arr<10> { Arr }
"#,
        );
        assert_has_edge(&graph, "caller", "Arr");
        assert_no_edge_to(&graph, "::N", "const generic params");
    }

    /// Tuple field access (.0, .1) does not create edges.
    #[test]
    fn test_tuple_field_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct T(pub i32);
pub fn caller(t: T) -> i32 { t.0 }
"#,
        );
        assert_has_edge(&graph, "caller", "T");
        assert_no_edge_to(&graph, ".0", "tuple fields");
    }

    /// Local variables are not edge targets.
    #[test]
    fn test_local_var_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub struct S;
pub fn caller() { let x = S; }
"#,
        );
        assert_has_edge(&graph, "caller", "S");
        assert_no_edge_to(&graph, "::x", "local variables");
    }

    /// Loop labels are not edge targets.
    #[test]
    fn test_label_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn caller() { 'lbl: loop { break 'lbl; } }
"#,
        );
        assert_no_edge_to(&graph, "lbl", "labels");
    }

    /// Lifetime parameters are not edge targets.
    #[test]
    fn test_lifetime_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn caller<'a>(s: &'a str) -> &'a str { s }
"#,
        );
        assert_no_edge_to(&graph, "'a", "lifetime params");
    }

    /// The 'static lifetime is not an edge target.
    #[test]
    fn test_static_lifetime_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn caller() -> &'static str { "" }
"#,
        );
        assert_no_edge_to(&graph, "static", "'static lifetime");
    }

    /// The crate:: path prefix is not an edge target.
    #[test]
    fn test_crate_keyword_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn target() {}
pub fn caller() { crate::target(); }
"#,
        );
        assert_has_edge(&graph, "caller", "target");
        assert_no_edge_to(&graph, "crate", "crate keyword");
    }

    /// Extern crate declarations are not edge targets.
    #[test]
    fn test_extern_crate_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate deps:dep
extern crate dep;
//- /dep.rs crate:dep
"#,
        );
        assert_no_edge_to(&graph, "dep", "extern crate");
    }

    /// Built-in attributes are not edge targets.
    #[test]
    fn test_builtin_attr_not_target() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
#[inline]
pub fn caller() {}
"#,
        );
        assert_no_edge_to(&graph, "inline", "built-in attributes");
    }

    // =========================================================================
    // EXTERNAL DEPENDENCY FILTERING
    //
    // Tests that only workspace-local dependencies are included in the graph.
    // References to external crates (std, third-party dependencies) are
    // filtered out since they can't be reorganized.
    // =========================================================================

    /// Cross-crate dependencies to workspace crates are included.
    #[test]
    fn test_cross_crate_local_deps() {
        let graph = extract_graph(
            r#"
//- /dep.rs crate:dep
pub fn target() {}
//- /main.rs crate:main deps:dep
pub fn caller() { dep::target(); }
"#,
        );
        assert_has_edge(&graph, "caller", "dep::target");
    }

    /// Code using only std types has no local dependencies.
    #[test]
    fn test_std_only_no_deps() {
        let graph = extract_graph(
            r#"
//- /lib.rs crate:test_crate
pub fn caller() -> String { String::new() }
"#,
        );
        assert!(
            graph.crates["test_crate"].symbols["caller"]
                .dependencies
                .is_empty()
        );
    }
}
