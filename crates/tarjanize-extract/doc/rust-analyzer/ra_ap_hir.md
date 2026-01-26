# ra_ap_hir

High-level Intermediate Representation (HIR) providing object-oriented access to Rust code. This is the public API of all compiler logic above syntax trees.

## Overview

**Design Philosophy:**
- HIR is bound to a particular crate instance (cfg flags and features applied)
- Syntax to HIR is many-to-one (same syntax can produce different HIR depending on cfg)
- Written in "OO" style - each type is self-contained, knows its parents and full context
- Insulates IDE features from compiler implementation details

**Architecture:**
- `hir_*` crates: ECS-style compiler implementation
- `hir`: Clean API boundary for IDE use

## Re-exports

Key re-exports from internal crates (use sparingly to avoid breaking encapsulation):

```rust
pub use cfg::{CfgAtom, CfgExpr, CfgOptions};
pub use hir_def::{
    visibility::Visibility, per_ns::Namespace, type_ref::Mutability,
    nameres::{DefMap, ModuleSource}, find_path::PrefixKind,
    lang_item::LangItem, import_map, GenericParamId, ModuleDefId, TraitId,
};
pub use hir_expand::{
    EditionedFileId, HirFileId, MacroCallId, MacroKind,
    files::{FilePosition, FileRange, InFile, ...},
    mod_path::{ModPath, PathKind}, name::Name, ProcMacros,
};
pub use hir_ty::{
    display::{DisplayTarget, HirDisplay}, LayoutError, mir,
    consteval::ConstEvalError, ...
};
pub use intern::{Symbol, sym};
```

---

## Core Types

### Crate

Represents a single crate instance.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Crate {
    pub(crate) id: base_db::Crate,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `base` | `fn(self) -> base_db::Crate` | Get underlying base_db crate |
| `origin` | `fn(self, db) -> CrateOrigin` | Local, Library, Lang, or Rustc |
| `is_builtin` | `fn(self, db) -> bool` | Is this std/core/alloc? |
| `dependencies` | `fn(self, db) -> Vec<CrateDependency>` | Direct dependencies |
| `reverse_dependencies` | `fn(self, db) -> Vec<Crate>` | Crates depending on this |
| `transitive_reverse_dependencies` | `fn(self, db) -> impl Iterator<Item = Crate>` | All reverse deps |
| `root_module` | `fn(self, db) -> Module` | Get crate root module |
| `modules` | `fn(self, db) -> Vec<Module>` | All modules in crate |
| `root_file` | `fn(self, db) -> FileId` | Root file (lib.rs/main.rs) |
| `edition` | `fn(self, db) -> Edition` | Rust edition |
| `version` | `fn(self, db) -> Option<String>` | Package version |
| `display_name` | `fn(self, db) -> Option<CrateDisplayName>` | Display name |
| `cfg` | `fn(self, db) -> &CfgOptions` | Active cfg options |
| `all` | `fn(db) -> Vec<Crate>` | All crates in workspace |

### CrateDependency

```rust
pub struct CrateDependency {
    pub krate: Crate,
    pub name: Name,
}
```

---

### Module

Represents a module within a crate.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Module {
    pub(crate) id: ModuleId,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `name` | `fn(self, db) -> Option<Name>` | Module name (None for crate root) |
| `krate` | `fn(self, db) -> Crate` | Containing crate |
| `crate_root` | `fn(self, db) -> Module` | Top-level module |
| `is_crate_root` | `fn(self, db) -> bool` | Is this the crate root? |
| `children` | `fn(self, db) -> impl Iterator<Item = Module>` | Child modules |
| `parent` | `fn(self, db) -> Option<Module>` | Parent module |
| `path_to_root` | `fn(self, db) -> Vec<Module>` | Path from this to root |
| `declarations` | `fn(self, db) -> Vec<ModuleDef>` | Items declared here |
| `scope` | `fn(self, db, visible_from) -> Vec<(Name, ScopeDef)>` | Visible items |
| `impl_defs` | `fn(self, db) -> Vec<Impl>` | Impl blocks |
| `diagnostics` | `fn(self, db, acc, style_lints)` | Collect diagnostics |

**Source Methods (not `HasSource`):**

| Method | Description |
|--------|-------------|
| `definition_source(db)` | File or `mod foo {}` with items |
| `definition_source_range(db)` | Range of definition |
| `declaration_source(db)` | `mod foo;` or `mod foo {}` declaration |
| `is_inline(db)` | True if `mod foo { ... }` |
| `as_source_file_id(db)` | File ID if file-based module |

---

### ModuleDef

Enum of all items that can be visible in a module scope.

```rust
pub enum ModuleDef {
    Module(Module),
    Function(Function),
    Adt(Adt),
    Variant(Variant),
    Const(Const),
    Static(Static),
    Trait(Trait),
    TypeAlias(TypeAlias),
    BuiltinType(BuiltinType),
    Macro(Macro),
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `module` | `fn(self, db) -> Option<Module>` | Containing module |
| `canonical_path` | `fn(self, db, edition) -> Option<String>` | Full path from crate root |
| `name` | `fn(self, db) -> Option<Name>` | Item name |
| `as_def_with_body` | `fn(self) -> Option<DefWithBody>` | If has a body |
| `attrs` | `fn(self, db) -> Option<AttrsWithOwner>` | Attributes |
| `diagnostics` | `fn(self, db, style_lints) -> Vec<AnyDiagnostic>` | Item diagnostics |

---

### Adt (Algebraic Data Types)

```rust
pub enum Adt {
    Struct(Struct),
    Union(Union),
    Enum(Enum),
}
```

Each variant has its own type with methods for:
- `name(db)` - Type name
- `module(db)` - Containing module
- `fields(db)` - Struct/variant fields
- `ty(db)` - The `Type` representation
- `repr(db)` - Repr attribute options

### Struct, Union, Enum

```rust
pub struct Struct { pub(crate) id: StructId }
pub struct Union { pub(crate) id: UnionId }
pub struct Enum { pub(crate) id: EnumId }
```

**Enum-specific methods:**

| Method | Description |
|--------|-------------|
| `variants(db)` | All enum variants |
| `is_data_carrying(db)` | Any variant has fields |

### Variant

```rust
pub struct Variant { pub(crate) id: EnumVariantId }
```

| Method | Description |
|--------|-------------|
| `name(db)` | Variant name |
| `parent_enum(db)` | Parent enum |
| `fields(db)` | Variant fields |
| `kind(db)` | Unit, Tuple, or Record |

---

### Function

```rust
pub struct Function { pub(crate) id: AnyFunctionId }
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `name` | `fn(self, db) -> Name` | Function name |
| `module` | `fn(self, db) -> Module` | Containing module |
| `self_param` | `fn(self, db) -> Option<SelfParam>` | Self parameter |
| `assoc_fn_params` | `fn(self, db) -> Vec<Param>` | All params including self |
| `num_params` | `fn(self, db) -> usize` | Number of parameters |
| `ret_type` | `fn(self, db) -> Type` | Return type |
| `is_async` | `fn(self, db) -> bool` | Is async fn |
| `is_const` | `fn(self, db) -> bool` | Is const fn |
| `is_unsafe` | `fn(self, db) -> IsUnsafe` | Safety |
| `as_assoc_item` | `fn(self, db) -> Option<AssocItem>` | If method |
| `as_proc_macro` | `fn(self, db) -> Option<Macro>` | If proc-macro |

---

### Trait

```rust
pub struct Trait { pub(crate) id: TraitId }
```

| Method | Description |
|--------|-------------|
| `name(db)` | Trait name |
| `module(db)` | Containing module |
| `items(db)` | Associated items |
| `items_with_supertraits(db)` | Including inherited items |
| `is_auto(db)` | Auto trait? |
| `is_unsafe(db)` | Unsafe trait? |
| `is_dyn_compatible(db)` | Object-safe? |

---

### Impl

```rust
pub struct Impl { pub(crate) id: AnyImplId }
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `self_ty` | `fn(self, db) -> Type` | The implementing type (e.g., `Foo` in `impl Foo`) |
| `trait_` | `fn(self, db) -> Option<Trait>` | Implemented trait (None for inherent impls) |
| `items` | `fn(self, db) -> Vec<AssocItem>` | Associated items (methods, consts, types) |
| `module` | `fn(self, db) -> Module` | Containing module |
| `is_negative` | `fn(self, db) -> bool` | Negative impl (`impl !Trait`) |
| `is_unsafe` | `fn(self, db) -> bool` | Unsafe impl |

**Common Patterns:**

```rust
// Check if impl is in a local crate
fn is_local_impl(db: &RootDatabase, impl_: Impl) -> bool {
    impl_.module(db).krate(db).origin(db).is_local()
}

// Get the ADT from the self type (if it's a struct/enum/union)
if let Some(adt) = impl_.self_ty(db).as_adt() {
    // adt is Adt::Struct, Adt::Enum, or Adt::Union
}

// Build a display name for the impl
fn impl_display_name(db: &RootDatabase, impl_: &Impl) -> String {
    let self_ty = impl_.self_ty(db);
    let display_target = DisplayTarget::from_crate(db, impl_.module(db).krate(db).into());
    let self_ty_name = self_ty.display(db, display_target).to_string();

    if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    }
}
```

---

### AssocItemContainer

Indicates whether an associated item lives in a trait or an impl.

```rust
pub enum AssocItemContainer {
    Trait(Trait),
    Impl(Impl),
}
```

**Usage Pattern (collapsing associated items to containers):**

```rust
use ra_ap_hir::{AsAssocItem, AssocItemContainer, Function, ModuleDef};

// When analyzing dependencies, collapse associated items to their containers
fn collapse_to_container(db: &RootDatabase, func: Function) -> Option<ModuleDef> {
    // Check if this function is an associated item
    if let Some(assoc) = func.as_assoc_item(db) {
        match assoc.container(db) {
            AssocItemContainer::Impl(impl_) => {
                // Method in an impl - the dependency is on the impl block
                // Note: Impl is NOT a ModuleDef, handle separately
                None
            }
            AssocItemContainer::Trait(trait_) => {
                // Method in a trait - the dependency is on the trait
                Some(ModuleDef::Trait(trait_))
            }
        }
    } else {
        // Free function - return as-is
        Some(ModuleDef::Function(func))
    }
}
```

This pattern is essential for tarjanize because:
- Impl methods can't exist without their impl block
- Trait methods can't exist without their trait
- When tracking dependencies, we collapse to the container to keep atomic units together

---

### Const, Static

```rust
pub struct Const { pub(crate) id: ConstId }
pub struct Static { pub(crate) id: StaticId }
```

| Method | Description |
|--------|-------------|
| `name(db)` | Item name |
| `module(db)` | Containing module |
| `ty(db)` | Type |
| `value(db)` | Const: evaluated value |
| `is_mut(db)` | Static: is mutable? |
| `is_extern(db)` | Static: is extern? |

---

### TypeAlias

```rust
pub struct TypeAlias { pub(crate) id: TypeAliasId }
```

| Method | Description |
|--------|-------------|
| `name(db)` | Alias name |
| `module(db)` | Containing module |
| `ty(db)` | Aliased type |
| `as_assoc_item(db)` | If associated type |

---

### Macro

```rust
pub struct Macro { pub(crate) id: MacroId }
```

| Method | Description |
|--------|-------------|
| `name(db)` | Macro name |
| `module(db)` | Containing module |
| `kind(db)` | Declarative, ProcMacro, Derive, Attr |
| `is_proc_macro(db)` | Is proc-macro? |
| `is_attr(db)` | Is attribute macro? |
| `is_derive(db)` | Is derive macro? |

---

### Field

```rust
pub struct Field {
    pub(crate) parent: VariantDef,
    pub(crate) id: LocalFieldId,
}
```

| Method | Description |
|--------|-------------|
| `name(db)` | Field name |
| `ty(db)` | Field type |
| `parent_def(db)` | Parent struct/union/variant |
| `index()` | Field index |

---

### Local, Label

```rust
pub struct Local { pub(crate) parent: DefWithBodyId, pub(crate) binding_id: BindingId }
pub struct Label { pub(crate) parent: DefWithBodyId, pub(crate) label_id: LabelId }
```

For local variables and labels within function bodies.

---

## HasSource Trait

Fetches the syntax node for a HIR item.

```rust
pub trait HasSource: Sized {
    type Ast: AstNode;

    /// Get the source syntax node
    fn source(self, db: &dyn HirDatabase) -> Option<InFile<Self::Ast>>;

    /// Get source with range (some items have range but no node)
    fn source_with_range(self, db: &dyn HirDatabase)
        -> Option<InFile<(TextRange, Option<Self::Ast>)>>;
}
```

**Implementations:**

| Type | Ast Type |
|------|----------|
| `Field` | `FieldSource` |
| `Adt` | `ast::Adt` |
| `Struct` | `ast::Struct` |
| `Union` | `ast::Union` |
| `Enum` | `ast::Enum` |
| `Variant` | `ast::Variant` |
| `Function` | `ast::Fn` |
| `Const` | `ast::Const` |
| `Static` | `ast::Static` |
| `Trait` | `ast::Trait` |
| `TypeAlias` | `ast::TypeAlias` |
| `Macro` | `Either<ast::Macro, ast::Fn>` |
| `Impl` | `ast::Impl` |

**Note**: `Module` is NOT `HasSource` because it has two source nodes (definition and declaration).

---

## Semantics (Deep Dive)

`Semantics` is the bridge between syntax trees and semantic analysis. It's the primary way to go from "what text says" to "what it means."

### Architecture

```rust
pub struct Semantics<'db, DB> {
    pub db: &'db DB,
    imp: SemanticsImpl<'db>,  // Internal implementation
}

// Internally, Semantics manages:
struct SemanticsImpl<'db> {
    db: &'db dyn HirDatabase,
    s2d_cache: RefCell<SourceToDefCache>,     // Syntax-to-definition cache
    macro_call_cache: RefCell<FxHashMap<...>>, // Macro expansion cache
}
```

**Why the cache matters**: Syntax trees are transient. When you call `sema.parse(file)`, the tree is cached. Later calls like `resolve_path` look up nodes in this cache. If you parse a file outside of `Semantics` and then try to resolve something from that tree, you'll get a panic because the node isn't in the cache.

### Creation

```rust
// Strong typing - knows the concrete database type
let sema = Semantics::new(&db);

// Weak typing - works with trait objects (for use in salsa functions)
let sema = Semantics::new_dyn(&db);
```

### Parsing and Caching

```rust
// Parse a file - ALWAYS use this, not db.parse() directly
let source_file = sema.parse(editioned_file_id);

// Parse guessing the edition (when crate is unknown)
let source_file = sema.parse_guess_edition(file_id);

// Parse or expand a HirFileId (handles macro files too)
let syntax_node = sema.parse_or_expand(hir_file_id);
```

### Converting Syntax to HIR

The `to_xxx_def` methods convert syntax nodes to their HIR equivalents:

```rust
// Each item type has a specific method
sema.to_fn_def(&ast_fn)           -> Option<Function>
sema.to_struct_def(&ast_struct)   -> Option<Struct>
sema.to_enum_def(&ast_enum)       -> Option<Enum>
sema.to_trait_def(&ast_trait)     -> Option<Trait>
sema.to_impl_def(&ast_impl)       -> Option<Impl>
sema.to_module_def(&ast_module)   -> Option<Module>
sema.to_const_def(&ast_const)     -> Option<Const>
sema.to_static_def(&ast_static)   -> Option<Static>
sema.to_type_alias_def(&ast_ta)   -> Option<TypeAlias>
sema.to_macro_def(&ast_macro)     -> Option<Macro>
sema.to_adt_def(&ast_adt)         -> Option<Adt>
sema.to_enum_variant_def(&ast_v)  -> Option<Variant>

// Generic version (requires node to be in cache)
sema.to_def(&syntax_node)         -> Option<T::Def>
```

### Path Resolution

**How it works internally**:

1. First, check if type inference already resolved this path (for paths in expressions/patterns)
2. If not, lower the AST path to a HIR path using `ExprCollector::lower_path`
3. Resolve through the `Resolver`:
   - Check type namespace (types, ADTs, traits, type aliases, modules)
   - Check value namespace (functions, consts, statics, local bindings)
   - Check macro namespace
4. Handle special cases (attributes, use trees, visibility paths, associated items)

```rust
// Basic resolution - returns first match
sema.resolve_path(&path) -> Option<PathResolution>

// Resolution with generic substitution info
sema.resolve_path_with_subst(&path) -> Option<(PathResolution, Option<GenericSubstitution>)>

// Resolution in all namespaces separately
sema.resolve_path_per_ns(&path) -> Option<PathResolutionPerNs>
```

**PathResolution variants**:

| Variant | When Used |
|---------|-----------|
| `Def(ModuleDef)` | Module-level items (functions, structs, etc.) |
| `Local(Local)` | Local variables |
| `TypeParam(TypeParam)` | Generic type parameters |
| `ConstParam(ConstParam)` | Const generic parameters |
| `SelfType(Impl)` | `Self` in impl blocks |
| `BuiltinAttr(BuiltinAttr)` | Built-in attributes like `#[derive]` |
| `ToolModule(ToolModule)` | Tool attributes like `#[rustfmt::skip]` |
| `DeriveHelper(DeriveHelper)` | Derive helper attributes |

### Method and Field Resolution

```rust
// Resolve method call to the actual function
sema.resolve_method_call(&method_call) -> Option<Function>

// With fallback to field (for UFCS-style calls)
sema.resolve_method_call_fallback(&method_call)
    -> Option<(Either<Function, Field>, Option<GenericSubstitution>)>

// Resolve field access
sema.resolve_field(&field_expr) -> Option<Either<Field, TupleField>>

// Record field in struct literal
sema.resolve_record_field(&record_field) -> Option<(Field, Option<Local>, Type)>

// Record field in pattern
sema.resolve_record_pat_field(&record_pat_field) -> Option<(Field, Type)>
```

### Type Information

```rust
// Get type of expression (with adjustment info for coercions)
sema.type_of_expr(&expr) -> Option<TypeInfo>

// Get type of pattern
sema.type_of_pat(&pat) -> Option<TypeInfo>

// Get type of self parameter
sema.type_of_self(&self_param) -> Option<Type>

// Resolve a type syntax node
sema.resolve_type(&ast_type) -> Option<Type>
```

### Macro Handling

Macros complicate everything. A token in your source might expand to multiple locations.

```rust
// Descend a token into all macro expansions
sema.descend_into_macros(token) -> SmallVec<[SyntaxToken; 1]>

// Descend, filtering to exact matches (same text and kind)
sema.descend_into_macros_exact(token) -> SmallVec<[SyntaxToken; 1]>

// Check if a node is inside a macro call
sema.is_inside_macro_call(token) -> bool

// Expand a macro call
sema.expand_macro_call(&macro_call) -> Option<InFile<SyntaxNode>>
```

### Scope and Analysis

```rust
// Get semantic scope at a syntax node
sema.scope(&syntax_node) -> Option<SemanticsScope>

// Get scope at a specific offset
sema.scope_at_offset(&node, offset) -> Option<SemanticsScope>

// Walk ancestors, crossing macro boundaries
sema.ancestors_with_macros(node) -> impl Iterator<Item = SyntaxNode>
```

### File/Module Mapping

```rust
// Get module(s) for a file
sema.file_to_module_def(file_id) -> Option<Module>
sema.file_to_module_defs(file_id) -> impl Iterator<Item = Module>

// Map node to original file range (handles macro expansions)
sema.original_range(&node) -> FileRange
sema.original_range_opt(&node) -> Option<FileRange>
```

### Internal: SourceAnalyzer

Under the hood, `Semantics` creates `SourceAnalyzer` instances for resolution. The analyzer holds:
- The file ID
- A `Resolver` (core name resolution machinery)
- Optionally, body/inference information for expressions

The `Resolver` is the actual workhorse - it knows about:
- Module scope (items visible in the current module)
- Expression scopes (local variables)
- Generic parameters
- Self type in impls

```rust
// The Resolver does the heavy lifting
resolver.resolve_path_in_type_ns(db, path)   // Types
resolver.resolve_path_in_value_ns(db, path)  // Values
resolver.resolve_path_as_macro(db, path)     // Macros
```

---

## PathResolution

Result of resolving a path.

```rust
pub enum PathResolution {
    Def(ModuleDef),
    Local(Local),
    TypeParam(TypeParam),
    ConstParam(ConstParam),
    SelfType(Impl),
    BuiltinAttr(BuiltinAttr),
    ToolModule(ToolModule),
    DeriveHelper(DeriveHelper),
}
```

---

## AssocItem

Associated items (in traits or impls).

```rust
pub enum AssocItem {
    Function(Function),
    Const(Const),
    TypeAlias(TypeAlias),
}
```

| Method | Description |
|--------|-------------|
| `name(db)` | Item name |
| `module(db)` | Containing module |
| `containing_trait(db)` | Trait if trait method |
| `containing_trait_impl(db)` | Impl if impl method |
| `container(db)` | Trait or Impl container |

---

## GenericParam

```rust
pub enum GenericParam {
    TypeParam(TypeParam),
    ConstParam(ConstParam),
    LifetimeParam(LifetimeParam),
}
```

---

## Type

Represents a resolved type.

```rust
pub struct Type { ... }
```

**Key Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_unit()` | `fn(&self) -> bool` | Is `()` |
| `is_bool()` | `fn(&self) -> bool` | Is `bool` |
| `is_fn()` | `fn(&self) -> bool` | Is function type |
| `is_closure()` | `fn(&self) -> bool` | Is closure |
| `is_reference()` | `fn(&self) -> bool` | Is `&T` or `&mut T` |
| `is_mutable_reference()` | `fn(&self) -> bool` | Is `&mut T` |
| `is_raw_ptr()` | `fn(&self) -> bool` | Is `*const T` or `*mut T` |
| `as_adt()` | `fn(&self) -> Option<Adt>` | Extract ADT (struct/enum/union) |
| `as_callable(db)` | `fn(&self, db) -> Option<Callable>` | Get callable info |
| `display(db, target)` | `fn(&self, db, DisplayTarget) -> impl Display` | Format for display |

**Type::as_adt() Pattern:**

When working with impl blocks, you often need to extract the ADT from the self type:

```rust
// Get the ADT from an impl's self type
let impl_: Impl = ...;
let self_ty = impl_.self_ty(db);

// as_adt() returns Some only for struct/enum/union types
// Returns None for references, tuples, slices, trait objects, etc.
if let Some(adt) = self_ty.as_adt() {
    match adt {
        Adt::Struct(s) => { /* impl for struct */ }
        Adt::Enum(e) => { /* impl for enum */ }
        Adt::Union(u) => { /* impl for union */ }
    }
}

// Note: For `impl Trait for &Type`, as_adt() returns None
// because the self type is a reference, not an ADT.
```

---

## DisplayTarget and HirDisplay

`DisplayTarget` controls how types are formatted for display. `HirDisplay` is the trait that enables formatting.

```rust
pub trait HirDisplay {
    fn display<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        target: DisplayTarget,
    ) -> impl fmt::Display + 'a;
}
```

**Creating a DisplayTarget:**

```rust
use ra_ap_hir::{DisplayTarget, HirDisplay};

// From a crate (most common)
let display_target = DisplayTarget::from_crate(db, krate.into());

// Or from a module
let display_target = DisplayTarget::from_module(db, module);
```

**Usage Pattern:**

```rust
fn format_type_name(db: &RootDatabase, impl_: &Impl) -> String {
    let self_ty = impl_.self_ty(db);
    let krate = impl_.module(db).krate(db);
    let display_target = DisplayTarget::from_crate(db, krate.into());

    // HirDisplay::display returns something that implements Display
    self_ty.display(db, display_target).to_string()
}
```

This is essential for generating unique names for impl blocks like:
- `impl Foo` (inherent impl)
- `impl Clone for Foo` (trait impl)
- `impl Clone for &Foo` (impl for reference type)
- `impl<T> Clone for Vec<T>` (generic impl)

---

## Common Traits

### HasVisibility

```rust
pub trait HasVisibility {
    fn visibility(&self, db: &dyn HirDatabase) -> Visibility;
}
```

Implemented by: `ModuleDef`, `Module`, `Field`, `Function`, etc.

### Visibility Enum

```rust
pub enum Visibility {
    /// Visible everywhere (`pub`)
    Public,
    /// Visible within the crate (`pub(crate)`)
    PubCrate(Crate),
    /// Visible within a specific module path (`pub(in path)`, `pub(super)`, or private)
    Module(ModuleId, VisExplicitness),
}

pub enum VisExplicitness {
    Explicit,  // pub(super), pub(in path)
    Implicit,  // No visibility annotation (private)
}
```

**Usage Pattern:**

```rust
use ra_ap_hir::{HasVisibility, Visibility};

fn visibility_to_string(db: &RootDatabase, def: &ModuleDef) -> Option<String> {
    let visibility = def.visibility(db);

    match visibility {
        Visibility::Public => Some("pub".to_string()),
        Visibility::PubCrate(_) => Some("pub(crate)".to_string()),
        Visibility::Module(_, vis_explicitness) => {
            // Module visibility covers pub(super), pub(in path), and private
            if vis_explicitness.is_explicit() {
                // pub(super) or pub(in path) - it's restricted but explicit
                Some("pub(restricted)".to_string())
            } else {
                // Implicit = private (no visibility keyword)
                None
            }
        }
    }
}
```

### HasCrate

```rust
pub trait HasCrate {
    fn krate(&self, db: &dyn HirDatabase) -> Crate;
}
```

### HasAttrs

```rust
pub trait HasAttrs {
    fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner;
}
```

---

## Database Traits

```rust
pub mod db {
    pub trait DefDatabase: ... { }
    pub trait HirDatabase: DefDatabase { }
}
```

`HirDatabase` is the main trait for semantic queries, extending `DefDatabase` with type inference and other analysis.

---

## InFile<T>

Wrapper combining a value with its file location.

```rust
pub struct InFile<T> {
    pub file_id: HirFileId,
    pub value: T,
}
```

Used for all source-located results. `HirFileId` can be a real file or macro expansion.

---

## HirFileId and Macro Expansion Tracing

`HirFileId` represents either a real file or a macro expansion. This is critical for handling macro-generated code.

```rust
pub struct HirFileId { ... }
```

**Tracing back to the original file:**

When a symbol is macro-generated, its `source()` returns a `HirFileId` pointing to the macro expansion, not the source file. To get the actual source file:

```rust
use ra_ap_hir::{HasSource, HirFileId};
use ra_ap_base_db::FileId;

fn get_original_file(db: &RootDatabase, func: Function) -> Option<FileId> {
    let source = func.source(db)?;
    let hir_file_id: HirFileId = source.file_id;

    // original_file() traces through macro expansions to the source file
    // containing the macro invocation
    let original = hir_file_id.original_file(db);

    // file_id(db) converts from EditionedFileId to FileId
    Some(original.file_id(db))
}
```

**The full pattern (as used in tarjanize-extract):**

```rust
fn compute_relative_file_path(
    db: &RootDatabase,
    crate_root: &VfsPath,
    hir_file_id: HirFileId,
) -> String {
    // 1. Trace back through macro expansions
    let file_id = hir_file_id.original_file(db).file_id(db);

    // 2. Look up the path in the VFS
    let vfs_path = db.file_path(file_id);

    // 3. Make relative to crate root
    vfs_path
        .strip_prefix(crate_root)
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| vfs_path.to_string())
}
```

---

## SymbolCollector (symbols module)

`SymbolCollector` gathers all symbols defined in a module. This is an alternative to `Module::declarations()` that captures more detail.

```rust
pub mod symbols {
    pub struct FileSymbol {
        pub name: Name,
        pub def: ModuleDef,
        pub loc: DeclarationLocation,
        pub is_alias: bool,
        pub is_assoc: bool,
    }

    pub struct DeclarationLocation {
        pub hir_file_id: HirFileId,
        pub ptr: SyntaxNodePtr,
    }
}
```

**Creating a SymbolCollector:**

```rust
use ra_ap_hir::symbols::SymbolCollector;

// Collect all symbols in a module (including private ones)
let symbols = SymbolCollector::new_module(db, module, /* include_private */ false);

// The second argument controls filtering:
// - false: collect all items (pub and private)
// - true: collect only items visible from outside the module
```

**Why use SymbolCollector instead of declarations()?**

1. **Location information**: Each `FileSymbol` includes `DeclarationLocation` with the exact file and syntax pointer
2. **Associated item tracking**: `is_assoc` tells you if it's a trait/impl item
3. **Alias detection**: `is_alias` identifies re-exports
4. **Efficiency**: Single pass collection with all metadata

**Usage Pattern (as in tarjanize-extract):**

```rust
fn extract_symbols(
    sema: &Semantics<'_, RootDatabase>,
    module: &Module,
) -> HashMap<String, Symbol> {
    let db = sema.db;
    let mut symbols = HashMap::new();

    // Collect all symbols defined in this module
    let collected = SymbolCollector::new_module(db, *module, false);

    for sym in collected {
        let name = sym.name.as_str().to_owned();

        // Get file path from the location
        let file = compute_relative_file_path(db, &crate_root, sym.loc.hir_file_id);

        // Get cost (byte size) from the syntax pointer
        let cost: f64 = u32::from(sym.loc.ptr.text_range().len()).into();

        symbols.insert(name, Symbol { file, cost, ... });
    }

    symbols
}
```

Note: `SymbolCollector` does NOT include impl blocks. Use `module.impl_defs(db)` separately for those.

---

## Usage in tarjanize

tarjanize-extract uses ra_ap_hir for:

1. **Crate iteration**: `Crate::all(db)` to get workspace crates
2. **Module traversal**: `Module::declarations(db)` for items
3. **Item inspection**: `Function`, `Struct`, `Enum`, etc. for symbol data
4. **Source access**: `HasSource::source(db)` to get AST nodes
5. **Dependency analysis**: Walking syntax with `Semantics` for resolution

Key patterns:
```rust
// Get all crates
for krate in Crate::all(db) {
    if krate.origin(db).is_local() {
        // Process workspace member
        let root = krate.root_module(db);
        process_module(db, root);
    }
}

// Process module items
fn process_module(db: &RootDatabase, module: Module) {
    for decl in module.declarations(db) {
        match decl {
            ModuleDef::Function(f) => { ... }
            ModuleDef::Struct(s) => { ... }
            ModuleDef::Module(m) => process_module(db, m),
            _ => {}
        }
    }
}

// Get source for an item
if let Some(source) = func.source(db) {
    let syntax = source.value.syntax();
    // Use syntax node
}
```
