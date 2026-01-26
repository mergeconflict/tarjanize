# Usage Patterns for rust-analyzer APIs

This document explains how to effectively use rust-analyzer's APIs for symbol extraction and dependency analysis. These patterns are derived from studying rust-analyzer's own IDE features (goto_definition, find_references, call_hierarchy).

**Sources:**
- [goto_definition.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide/src/goto_definition.rs)
- [references.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide/src/references.rs)
- [call_hierarchy.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide/src/call_hierarchy.rs)
- [defs.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide-db/src/defs.rs)
- [search.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide-db/src/search.rs)

## Core Concepts

### The Two-Layer Architecture

rust-analyzer has a strict separation between:

1. **Concrete Syntax Tree (CST)** - In `ra_ap_syntax`, provides full-fidelity representation of source text
2. **High-level IR (HIR)** - In `ra_ap_hir`, provides semantically-resolved view of code

The bridge between them is `Semantics`, which lets you go from syntax nodes to semantic information.

### The Database Pattern

Everything flows through the `RootDatabase`, which is a salsa database. All queries are memoized. Key traits:

```
RootDatabase
    └── SourceDatabase (file text, source roots)
        └── RootQueryDb (parsing)
            └── DefDatabase (definitions)
                └── HirDatabase (type inference, trait solving)
```

### File vs Semantic Identity

A critical concept: **files and semantic items are not 1:1**. Due to macros and cfg attributes:
- One file can produce multiple HIR items (cfg branches)
- One HIR item can span multiple files (macro expansions)

The `HirFileId` type captures this - it can be either a real file or a macro expansion.

---

## Setting Up

### Loading a Workspace

```rust
use ra_ap_load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use ra_ap_project_model::CargoConfig;

let cargo_config = CargoConfig::default();
let load_config = LoadCargoConfig {
    load_out_dirs_from_check: true,  // Run cargo check for OUT_DIR
    with_proc_macro_server: ProcMacroServerChoice::Sysroot,
    prefill_caches: false,
};

let (db, vfs, _) = load_workspace_at(
    Path::new("."),
    &cargo_config,
    &load_config,
    &|msg| eprintln!("{}", msg),
)?;
```

---

## Traversing Code Structure

### Iterating Over Crates

```rust
use ra_ap_hir::Crate;

for krate in Crate::all(&db) {
    // Filter to workspace members only (skip dependencies)
    if !krate.origin(&db).is_local() {
        continue;
    }

    let root_module = krate.root_module(&db);
    process_module(&db, root_module);
}
```

### Traversing Modules

There are two approaches to get items from a module:

#### Option 1: Module::declarations() (Simple)

```rust
use ra_ap_hir::{Module, ModuleDef};

fn process_module(db: &RootDatabase, module: Module) {
    // Get all items declared directly in this module
    for decl in module.declarations(db) {
        match decl {
            ModuleDef::Function(f) => process_function(db, f),
            ModuleDef::Struct(s) => process_struct(db, s),
            ModuleDef::Enum(e) => process_enum(db, e),
            ModuleDef::Trait(t) => process_trait(db, t),
            ModuleDef::TypeAlias(ta) => process_type_alias(db, ta),
            ModuleDef::Const(c) => process_const(db, c),
            ModuleDef::Static(s) => process_static(db, s),
            ModuleDef::Module(m) => process_module(db, m), // Recurse into submodules
            ModuleDef::Macro(m) => process_macro(db, m),
            ModuleDef::BuiltinType(_) | ModuleDef::Variant(_) => {
                // BuiltinType shouldn't appear in declarations
                // Variant is accessed through its parent Enum
            }
        }
    }

    // Also process impl blocks
    for impl_ in module.impl_defs(db) {
        process_impl(db, impl_);
    }
}
```

#### Option 2: SymbolCollector (With Location Info)

When you need source locations and syntax pointers:

```rust
use ra_ap_hir::symbols::SymbolCollector;

fn process_module_with_locations(db: &RootDatabase, module: Module) {
    // Collect all symbols with their locations
    // Second arg: false = include private items, true = public only
    let symbols = SymbolCollector::new_module(db, module, false);

    for sym in symbols {
        let name = sym.name.as_str();
        let def = sym.def;  // The ModuleDef

        // Location info: file and syntax pointer
        let file_id = sym.loc.hir_file_id;
        let text_range = sym.loc.ptr.text_range();
        let byte_size = u32::from(text_range.len());

        // sym.is_assoc: true if this is an associated item (trait/impl member)
        // sym.is_alias: true if this is a re-export

        process_symbol(name, def, file_id, byte_size);
    }

    // Note: SymbolCollector does NOT include impl blocks
    // Use module.impl_defs(db) separately
    for impl_ in module.impl_defs(db) {
        process_impl(db, impl_);
    }
}
```

### Getting Source Location

```rust
use ra_ap_hir::HasSource;

fn get_source_info(db: &RootDatabase, func: Function) {
    // Get the syntax node for this function
    if let Some(source) = func.source(db) {
        let file_id = source.file_id;
        let ast_fn: ast::Fn = source.value;

        // Get text range within the file
        let range = ast_fn.syntax().text_range();

        // Map back to original file (handles macro expansions)
        if let Some(real_file) = file_id.file_id() {
            let file = real_file.file_id(db);
            // file is the actual FileId
        }
    }
}
```

---

## Name Classification (NameClass/NameRefClass)

rust-analyzer uses a two-stage classification system for names. This is the canonical way to determine what a name refers to, used throughout IDE features like goto_definition and find_references.

### NameClass - For Definition Sites

When you have an `ast::Name` (a definition site like `fn foo` or `struct Bar`):

```rust
use ra_ap_ide_db::defs::{NameClass, Definition};

fn classify_definition(sema: &Semantics<'_, RootDatabase>, name: &ast::Name) {
    if let Some(class) = NameClass::classify(sema, name) {
        match class {
            NameClass::Definition(def) => {
                // Standard definition (function, struct, etc.)
                // def is a Definition enum
            }
            NameClass::ConstReference(def) => {
                // Name in a pattern that refers to a const (like `None` in `match x { None => }`)
            }
            NameClass::PatFieldShorthand { local_def, field_ref } => {
                // Shorthand field pattern like `let Foo { field } = foo`
                // Both defines a local AND references a field
            }
        }
    }
}
```

### NameRefClass - For Reference Sites

When you have an `ast::NameRef` (a reference to something defined elsewhere):

```rust
use ra_ap_ide_db::defs::{NameRefClass, Definition};

fn classify_reference(sema: &Semantics<'_, RootDatabase>, name_ref: &ast::NameRef) {
    if let Some(class) = NameRefClass::classify(sema, name_ref) {
        match class {
            NameRefClass::Definition(def, subst) => {
                // Reference to a definition
                // def: what it refers to
                // subst: generic substitution info (Option<GenericSubstitution>)
            }
            NameRefClass::FieldShorthand { local_ref, field_ref } => {
                // Shorthand field in struct literal: `Foo { field }` where `field` is a local
            }
            NameRefClass::ExternCrateShorthand { decl, krate } => {
                // `extern crate foo;` - the `foo` name
            }
        }
    }
}
```

### Why Use Classification Instead of resolve_path?

Classification handles edge cases that raw `resolve_path` misses:

1. **Pattern constants**: `match x { None => }` - `None` is a NameRef but not a path
2. **Field shorthands**: `Foo { field }` - both defines local AND references field
3. **Extern crates**: Different resolution rules
4. **Derive macro inputs**: Special handling required

**Canonical pattern from goto_definition.rs:**

```rust
// 1. Get the token at cursor
// 2. Find the name or name_ref that contains it
// 3. Classify syntactically, then semantically

for name_ref in syntax.descendants().filter_map(ast::NameRef::cast) {
    if let Some(NameRefClass::Definition(def, _)) = NameRefClass::classify(sema, &name_ref) {
        // def is a Definition - convert to your dependency type
        process_definition(def);
    }
}

for name in syntax.descendants().filter_map(ast::Name::cast) {
    if let Some(NameClass::Definition(def)) = NameClass::classify(sema, &name) {
        // This is a definition site
        process_definition(def);
    }
}
```

### The Definition Type

`Definition` is a unified enum for all nameable things:

```rust
pub enum Definition {
    Macro(Macro),
    Field(Field),
    Module(Module),
    Function(Function),
    Adt(Adt),
    Variant(Variant),
    Const(Const),
    Static(Static),
    Trait(Trait),
    TraitAlias(TraitAlias),
    TypeAlias(TypeAlias),
    BuiltinType(BuiltinType),
    SelfType(Impl),
    Local(Local),
    GenericParam(GenericParam),
    Label(Label),
    DeriveHelper(DeriveHelper),
    BuiltinAttr(BuiltinAttr),
    ToolModule(ToolModule),
    ExternCrateDecl(ExternCrateDecl),
    InlineAsmRegOrRegClass(()),
    InlineAsmOperand(InlineAsmOperand),
}
```

---

## Semantic Analysis with `Semantics`

`Semantics` is the bridge between syntax and semantics. It caches syntax trees and provides semantic information.

### Creating Semantics

```rust
use ra_ap_hir::Semantics;

let sema = Semantics::new(&db);
```

### Parsing a File

```rust
use ra_ap_hir::EditionedFileId;

// Get an EditionedFileId (file + edition + crate)
let file_id: FileId = ...;
let editioned = sema.attach_first_edition(file_id);

// Parse the file
let source_file = sema.parse(editioned);

// Now you can walk the syntax tree
for item in source_file.items() {
    // ...
}
```

### Converting Syntax to HIR

```rust
// From syntax to HIR (use to_xxx_def methods)
let ast_fn: ast::Fn = ...;
if let Some(hir_fn) = sema.to_fn_def(&ast_fn) {
    // Now you have the semantic Function
}

// The general pattern:
// sema.to_fn_def(&ast::Fn) -> Option<Function>
// sema.to_struct_def(&ast::Struct) -> Option<Struct>
// sema.to_enum_def(&ast::Enum) -> Option<Enum>
// sema.to_trait_def(&ast::Trait) -> Option<Trait>
// etc.
```

### Resolving Paths

This is the core of dependency analysis - finding what a name refers to.

```rust
use ra_ap_hir::PathResolution;

fn resolve_path(sema: &Semantics<'_, RootDatabase>, path: &ast::Path) {
    if let Some(resolution) = sema.resolve_path(path) {
        match resolution {
            PathResolution::Def(module_def) => {
                // Resolved to a module-level definition
                // module_def is a ModuleDef (Function, Struct, Enum, etc.)
            }
            PathResolution::Local(local) => {
                // Resolved to a local variable
            }
            PathResolution::TypeParam(type_param) => {
                // Resolved to a type parameter
            }
            PathResolution::ConstParam(const_param) => {
                // Resolved to a const parameter
            }
            PathResolution::SelfType(impl_) => {
                // Resolved to Self in an impl block
            }
            _ => {
                // Other cases: BuiltinAttr, ToolModule, DeriveHelper
            }
        }
    }
}
```

### Resolving Method Calls

#### Basic Method Resolution

```rust
fn resolve_method(sema: &Semantics<'_, RootDatabase>, call: &ast::MethodCallExpr) {
    if let Some(func) = sema.resolve_method_call(call) {
        // func is the resolved Function
        let name = func.name(db);
        let module = func.module(db);
        // ...
    }
}
```

#### Handling UFCS and Field Access Fallback

Sometimes what looks like a method call is actually a field access (for callable fields) or UFCS syntax. Use `resolve_method_call_fallback`:

```rust
use either::Either;

fn resolve_method_or_field(
    sema: &Semantics<'_, RootDatabase>,
    call: &ast::MethodCallExpr,
) {
    // resolve_method_call_fallback handles:
    // 1. Normal method calls: foo.bar() -> Function
    // 2. Callable fields: foo.callback() where callback is a field of fn type -> Field
    if let Some((either, subst)) = sema.resolve_method_call_fallback(call) {
        match either {
            Either::Left(func) => {
                // Normal method call
                let function: Function = func;
            }
            Either::Right(field) => {
                // Field access with call syntax
                // e.g., struct Foo { callback: fn() }; foo.callback()
                let field: Field = field;
            }
        }
        // subst contains generic substitution info if applicable
    }
}
```

#### Getting Callable Information

For analyzing return types and parameters of a call:

```rust
fn analyze_call(sema: &Semantics<'_, RootDatabase>, call: &ast::MethodCallExpr) {
    if let Some(callable) = sema.resolve_method_call_as_callable(call) {
        // Callable provides:
        // - return_type()
        // - params()
        // - n_params()
        let ret_ty = callable.return_type();
    }
}
```

### Resolving Field Access

```rust
use either::Either;

fn resolve_field(sema: &Semantics<'_, RootDatabase>, field_expr: &ast::FieldExpr) {
    if let Some(field) = sema.resolve_field(field_expr) {
        match field {
            Either::Left(named_field) => {
                // Named field (struct.field)
                let name = named_field.name(db);
            }
            Either::Right(tuple_field) => {
                // Tuple field (tuple.0)
                let index = tuple_field.index;
            }
        }
    }
}
```

### Getting Type Information

```rust
fn get_expr_type(sema: &Semantics<'_, RootDatabase>, expr: &ast::Expr) {
    if let Some(type_info) = sema.type_of_expr(expr) {
        let original_ty = type_info.original;
        let adjusted_ty = type_info.adjusted(); // After coercions

        // Display the type
        let display_target = DisplayTarget::from_crate(&db, krate.into());
        println!("Type: {}", adjusted_ty.display(&db, display_target));
    }
}
```

---

## Walking Syntax Trees for Dependencies

To find all dependencies (references to other symbols), you need to walk the syntax tree and resolve each reference.

### The parse_or_expand and find_node_in_file Pattern

**Problem**: `HasSource::source()` returns a syntax node, but `Semantics::resolve_path()` only works on nodes from trees that Semantics has cached. The two trees have identical content but are different allocations.

**Solution**: Use `sema.parse_or_expand()` to get a cached tree, then find the equivalent node by matching text ranges.

```rust
use ra_ap_hir::{HasSource, HirFileId, Semantics};
use ra_ap_syntax::{AstNode, SyntaxNode};

/// Find a node in a syntax tree by matching its text range.
/// The source tree and parsed tree have identical content but different allocations.
fn find_node_in_file(root: &SyntaxNode, target: &SyntaxNode) -> Option<SyntaxNode> {
    let range = target.text_range();
    root.descendants().find(|n| n.text_range() == range)
}

/// Collect dependencies from a source node.
fn collect_deps_from<T: AstNode>(
    sema: &Semantics<'_, RootDatabase>,
    file_id: HirFileId,
    syntax: &T,
    deps: &mut HashSet<ModuleDef>,
) {
    // 1. Get the cached tree from Semantics (handles real files and macro expansions)
    let root = sema.parse_or_expand(file_id);

    // 2. Find the equivalent node in the cached tree
    if let Some(node) = find_node_in_file(&root, syntax.syntax()) {
        // 3. Now we can safely resolve paths
        for path in node.descendants().filter_map(ast::Path::cast) {
            if let Some(PathResolution::Def(def)) = sema.resolve_path(&path) {
                deps.insert(def);
            }
        }
    }
}

// Usage with HasSource:
fn find_dependencies(sema: &Semantics<'_, RootDatabase>, func: Function) -> HashSet<ModuleDef> {
    let mut deps = HashSet::new();

    if let Some(src) = func.source(sema.db) {
        // src.file_id is a HirFileId (might be macro expansion)
        // src.value is an ast::Fn from a different tree allocation
        collect_deps_from(sema, src.file_id, &src.value, &mut deps);
    }

    deps
}
```

This pattern is essential when:
- Walking syntax trees obtained from `HasSource::source()`
- Resolving paths in macro-expanded code
- Analyzing function bodies, struct definitions, etc.

### Basic Pattern

```rust
fn find_dependencies_in_item(sema: &Semantics<'_, RootDatabase>, item: &ast::Item) {
    let mut dependencies = Vec::new();

    // Walk all descendant nodes
    for node in item.syntax().descendants() {
        // Check for paths (the most common reference type)
        if let Some(path) = ast::Path::cast(node.clone()) {
            if let Some(resolution) = sema.resolve_path(&path) {
                if let PathResolution::Def(def) = resolution {
                    dependencies.push(def);
                }
            }
        }

        // Check for method calls
        if let Some(method_call) = ast::MethodCallExpr::cast(node.clone()) {
            if let Some(func) = sema.resolve_method_call(&method_call) {
                dependencies.push(ModuleDef::Function(func));
            }
        }

        // Check for field access
        if let Some(field_expr) = ast::FieldExpr::cast(node.clone()) {
            if let Some(Either::Left(field)) = sema.resolve_field(&field_expr) {
                // Field access creates a dependency on the struct
                let parent = field.parent_def(db);
                dependencies.push(parent.into());
            }
        }
    }

    dependencies
}
```

### Handling Macro Expansions

Macros are one of the most complex aspects of Rust analysis. rust-analyzer provides several methods for working with macro-expanded code.

#### descend_into_macros - The Core Pattern

When you have a token from source code that might be inside a macro invocation, use `descend_into_macros` to find where it ends up after expansion:

```rust
// From references.rs - finding all usages of a name
fn find_refs_in_macro(sema: &Semantics<'_, RootDatabase>, token: SyntaxToken) {
    // descend_into_macros returns all locations the token maps to
    // (can be multiple due to macro hygiene)
    for descended in sema.descend_into_macros(token.clone()) {
        // Each descended token is in the expanded syntax tree
        // Now you can walk ancestors or resolve paths
        if let Some(name_ref) = descended.parent_ancestors().find_map(ast::NameRef::cast) {
            if let Some(NameRefClass::Definition(def, _)) = NameRefClass::classify(sema, &name_ref) {
                // Found a reference in macro-expanded code
            }
        }
    }
}
```

#### Variants of descend_into_macros

```rust
// Basic descent - returns all expansions
sema.descend_into_macros(token) -> SmallVec<[SyntaxToken; 1]>

// Exact matching - only tokens with same text and kind
sema.descend_into_macros_exact(token) -> SmallVec<[SyntaxToken; 1]>

// Excludes opaque macro contexts (proc macros)
sema.descend_into_macros_no_opaque(token, always_descend_into_derives: bool)
    -> SmallVec<[InFile<SyntaxToken>; 1]>
```

#### Walking Ancestors Through Macro Boundaries

When you need to find the containing function/impl/module of something in expanded code:

```rust
// From call_hierarchy.rs - finding the function that contains a call
fn find_containing_function(sema: &Semantics<'_, RootDatabase>, node: SyntaxNode) {
    // ancestors_with_macros crosses macro expansion boundaries
    for ancestor in sema.ancestors_with_macros(node) {
        if let Some(func) = ast::Fn::cast(ancestor) {
            // Found the containing function
            if let Some(hir_func) = sema.to_fn_def(&func) {
                return Some(hir_func);
            }
        }
    }
    None
}
```

#### Token Ancestors Through Macros

```rust
// Similar but starts from a token
for ancestor in sema.token_ancestors_with_macros(token) {
    // Walk up through macro expansions
}
```

#### The InFile Pattern

When working with macro expansions, results often come as `InFile<T>`:

```rust
pub struct InFile<T> {
    pub file_id: HirFileId,  // Might be a macro expansion
    pub value: T,
}

// Check if it's a real file or macro expansion
if file_id.is_macro() {
    // This is in expanded code
}

// Get the original file (traces through expansions)
let original = file_id.original_file(db);
```

---

## Working with Impl Blocks

Impl blocks require special handling because they're NOT part of `ModuleDef` - they're anonymous (you can't write a path to them like `my_crate::SomeImpl`).

### Getting Impl Blocks

```rust
// Get all impl blocks in a module
for impl_ in module.impl_defs(db) {
    process_impl(db, impl_);
}
```

### Extracting Impl Dependencies

An impl block has two kinds of dependencies:
1. **Declaration dependencies**: The self type and trait (if any)
2. **Body dependencies**: What the methods reference

```rust
use ra_ap_hir::{Impl, ModuleDef, Type};

fn find_impl_dependencies(db: &RootDatabase, impl_: Impl) -> Vec<ModuleDef> {
    let mut deps = Vec::new();

    // 1. Self type dependency
    let self_ty: Type = impl_.self_ty(db);
    if let Some(adt) = self_ty.as_adt() {
        deps.push(ModuleDef::Adt(adt));
    }
    // Note: as_adt() returns None for reference types like &Foo,
    // tuples, slices, trait objects, etc.

    // 2. Trait dependency (for trait impls)
    if let Some(trait_) = impl_.trait_(db) {
        deps.push(ModuleDef::Trait(trait_));
    }

    // 3. Associated items need separate analysis
    for item in impl_.items(db) {
        // item is an AssocItem (Function, Const, or TypeAlias)
        // Analyze each one for its body dependencies
        let item_deps = find_dependencies_in_assoc_item(sema, item);
        deps.extend(item_deps);
    }

    deps
}
```

### Building Impl Names

Impl blocks need unique names for identification. Use `HirDisplay`:

```rust
use ra_ap_hir::{DisplayTarget, HirDisplay, Impl};

fn impl_name(db: &RootDatabase, impl_: &Impl) -> String {
    let self_ty = impl_.self_ty(db);
    let krate = impl_.module(db).krate(db);
    let display_target = DisplayTarget::from_crate(db, krate.into());

    // HirDisplay gives proper type names including generics
    let self_ty_name = self_ty.display(db, display_target).to_string();

    if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    }
}

// Examples of generated names:
// - "impl MyStruct"
// - "impl Clone for MyStruct"
// - "impl Clone for &MyStruct"
// - "impl<T> Clone for Vec<T>"
```

### Checking If an Impl Is Local

```rust
fn is_local_impl(db: &RootDatabase, impl_: Impl) -> bool {
    impl_.module(db).krate(db).origin(db).is_local()
}
```

---

## Collapsing Associated Items to Containers

For dependency analysis, associated items (methods, associated types, associated consts) should often be collapsed to their containers (impl blocks or traits). You can't have a method without its impl block, and they can't be split apart.

### The AsAssocItem Pattern

```rust
use ra_ap_hir::{AsAssocItem, AssocItemContainer, Function, ModuleDef};

/// Check if a resolved function is an associated item and get its container.
fn get_container_for_method(
    db: &RootDatabase,
    func: Function,
) -> Option<AssocItemContainer> {
    // as_assoc_item returns None for free functions
    let assoc = func.as_assoc_item(db)?;
    Some(assoc.container(db))
}

/// When resolving method calls, collapse to the impl/trait container.
fn collect_method_call_deps(
    sema: &Semantics<'_, RootDatabase>,
    method_call: &ast::MethodCallExpr,
    deps: &mut HashSet<Dependency>,
) {
    if let Some(func) = sema.resolve_method_call(method_call) {
        if let Some(assoc) = func.as_assoc_item(sema.db) {
            match assoc.container(sema.db) {
                AssocItemContainer::Impl(impl_) => {
                    // Method lives in an impl block
                    // Record dependency on the impl, not the method
                    deps.insert(Dependency::Impl(impl_));
                }
                AssocItemContainer::Trait(trait_) => {
                    // Method lives in a trait (including default methods)
                    deps.insert(Dependency::Trait(trait_));
                }
            }
        } else {
            // Free function (not associated)
            deps.insert(Dependency::Function(func));
        }
    }
}
```

### Why This Matters

Consider `impl Clone for Foo { fn clone(&self) { ... } }`:
- The method `clone` cannot exist without the impl block
- If we split `clone` into a separate crate, we'd break the code
- Dependencies on `clone()` are really dependencies on `impl Clone for Foo`

Similarly for trait methods:
- Default methods in traits cannot be separated from their trait
- Dependencies resolve to the trait, not individual methods

### Types That Support AsAssocItem

The following types implement `AsAssocItem`:
- `Function` - methods
- `Const` - associated consts
- `TypeAlias` - associated types

```rust
// All of these can be associated items
func.as_assoc_item(db)        // Option<AssocItem>
const_.as_assoc_item(db)      // Option<AssocItem>
type_alias.as_assoc_item(db)  // Option<AssocItem>
```

---

## Filtering to Workspace Members

A common requirement is to only track dependencies within the workspace, not to external crates.

```rust
fn is_local_def(db: &RootDatabase, def: ModuleDef) -> bool {
    match def {
        ModuleDef::Module(m) => m.krate(db).origin(db).is_local(),
        ModuleDef::Function(f) => f.module(db).krate(db).origin(db).is_local(),
        ModuleDef::Adt(a) => {
            let module = match a {
                Adt::Struct(s) => s.module(db),
                Adt::Union(u) => u.module(db),
                Adt::Enum(e) => e.module(db),
            };
            module.krate(db).origin(db).is_local()
        }
        ModuleDef::Variant(v) => v.parent_enum(db).module(db).krate(db).origin(db).is_local(),
        ModuleDef::Const(c) => c.module(db).krate(db).origin(db).is_local(),
        ModuleDef::Static(s) => s.module(db).krate(db).origin(db).is_local(),
        ModuleDef::Trait(t) => t.module(db).krate(db).origin(db).is_local(),
        ModuleDef::TypeAlias(ta) => ta.module(db).krate(db).origin(db).is_local(),
        ModuleDef::Macro(m) => m.module(db).krate(db).origin(db).is_local(),
        ModuleDef::BuiltinType(_) => false, // Builtins are never local
    }
}
```

---

## The Resolver

For advanced use cases, you might need direct access to the `Resolver`. The resolver is the core name resolution machinery.

### Getting a Resolver

```rust
// From a scope (most common)
let scope = sema.scope(some_syntax_node)?;
let resolver = scope.resolver;

// Or from module
let module: Module = ...;
let resolver = module.id.resolver(db);
```

### Using the Resolver Directly

```rust
use hir_def::resolver::{TypeNs, ValueNs};

// Resolve in type namespace
let path: Path = ...; // This is hir_def::expr_store::path::Path, not ast::Path
let (type_ns, remaining, _) = resolver.resolve_path_in_type_ns(db, &path)?;

match type_ns {
    TypeNs::AdtId(adt) => { /* struct/enum/union */ }
    TypeNs::TraitId(trait_) => { /* trait */ }
    TypeNs::TypeAliasId(alias) => { /* type alias */ }
    TypeNs::SelfType(impl_) => { /* Self type */ }
    TypeNs::GenericParam(param) => { /* type parameter */ }
    TypeNs::BuiltinType(builtin) => { /* i32, bool, etc */ }
    TypeNs::ModuleId(module) => { /* module */ }
    TypeNs::EnumVariantId(variant) => { /* enum variant in type position */ }
    TypeNs::AdtSelfType(adt) => { /* Self in ADT methods */ }
}

// Resolve in value namespace
let value_ns = resolver.resolve_path_in_value_ns_fully(db, &path, hygiene)?;

match value_ns {
    ValueNs::LocalBinding(binding) => { /* local variable */ }
    ValueNs::FunctionId(func) => { /* function */ }
    ValueNs::ConstId(const_) => { /* const */ }
    ValueNs::StaticId(static_) => { /* static */ }
    ValueNs::StructId(struct_) => { /* struct constructor */ }
    ValueNs::EnumVariantId(variant) => { /* enum variant constructor */ }
    ValueNs::ImplSelf(impl_) => { /* Self in impl */ }
    ValueNs::GenericParam(param) => { /* const generic parameter */ }
}
```

---

## Common Pitfalls

### 1. Not Caching Syntax Trees

`Semantics` caches syntax trees internally. If you parse the same file multiple times without going through `Semantics`, you'll break the cache and get panics.

```rust
// WRONG: Parsing outside of Semantics
let tree = db.parse(file_id);
sema.resolve_path(&some_path); // PANIC: "node not found in Semantics"

// RIGHT: Always go through Semantics
let tree = sema.parse(file_id);
```

### 2. Mixing Files and Semantics Instances

Each `Semantics` instance has its own cache. Don't use syntax nodes from one `Semantics` with another.

### 3. Forgetting About Macros

A path like `foo!()::Bar` won't have a simple resolution. You need to handle macro expansions.

### 4. Edition Mismatches

Different crates can have different Rust editions. Always use `EditionedFileId` when parsing.

### 5. Assuming Unique Resolution

Due to glob imports and re-exports, one name might resolve to multiple definitions in different namespaces. Use `resolve_path_per_ns` for complete information.

---

## Finding All Usages (Two-Phase Search)

rust-analyzer's "find references" feature uses a **two-phase search** for efficiency. This pattern is in [search.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide-db/src/search.rs).

### The Two Phases

1. **Fast text search**: Use `memchr` or similar to find potential matches by name
2. **Precise semantic resolution**: Confirm each match using name resolution

```rust
// Conceptual pattern from search.rs:
// "first, we run a fast text search to get a super-set of matches.
// Then, we confirm each match using precise name resolution."

fn find_usages(db: &RootDatabase, def: Definition) -> Vec<FileRange> {
    let name = def.name(db)?;
    let name_str = name.as_str();

    let mut usages = Vec::new();

    // Phase 1: Fast text search
    for file_id in search_scope.files() {
        let text = db.file_text(file_id);

        // Find all positions where the name appears
        for offset in find_all_occurrences(&text, name_str) {
            // Phase 2: Precise resolution
            let token = find_token_at_offset(db, file_id, offset);
            if let Some(name_ref) = token.parent().and_then(ast::NameRef::cast) {
                if let Some(NameRefClass::Definition(resolved, _)) =
                    NameRefClass::classify(&sema, &name_ref)
                {
                    if resolved == def {
                        usages.push(FileRange { file_id, range: name_ref.syntax().text_range() });
                    }
                }
            }
        }
    }

    usages
}
```

### Using the FindUsages API

rust-analyzer exposes this as a fluent API:

```rust
use ra_ap_ide_db::search::{SearchScope, UsageSearchResult};
use ra_ap_ide_db::defs::Definition;

fn find_all_references(
    sema: &Semantics<'_, RootDatabase>,
    def: Definition,
) -> UsageSearchResult {
    def.usages(sema)
        .set_scope(SearchScope::krate(db, krate))  // Limit to a crate
        .include_self_refs()                        // Include self-references
        .all()                                      // Execute search
}

// SearchScope options:
SearchScope::single_file(file_id)      // Just one file
SearchScope::krate(db, krate)          // One crate
SearchScope::crate_graph(db)           // All crates in graph
SearchScope::module_and_children(db, module)  // Module tree
```

### Why Two Phases?

- **Text search is O(n)** in file size, very fast with SIMD
- **Semantic resolution is expensive** - type inference, name resolution
- By filtering first, we only do expensive work on actual candidates

This is relevant for tarjanize because:
- Dependency analysis walks all syntax anyway (can't use text filter)
- But understanding the pattern helps when extending functionality
- The `Definition` type and `NameRefClass::classify` pattern applies directly

---

## Performance Considerations

1. **Batch operations**: Many queries are memoized. Querying the same information twice is cheap.

2. **Lazy inference**: Type inference is expensive. Use `analyze_no_infer` when you don't need types.

3. **Parallelism**: The database is thread-safe. You can use rayon for parallel analysis.

4. **Source roots**: Library crates are marked as `is_library = true` and assumed immutable. This enables better caching.

---

## Summary: Complete Workflow for Symbol Extraction

```rust
use ra_ap_hir::{Crate, Module, ModuleDef, Semantics, HasSource};
use ra_ap_ide_db::RootDatabase;
use ra_ap_load_cargo::{LoadCargoConfig, load_workspace_at};

fn extract_symbols(db: &RootDatabase) {
    let sema = Semantics::new(db);

    for krate in Crate::all(db) {
        if !krate.origin(db).is_local() {
            continue;  // Skip external dependencies
        }

        let root = krate.root_module(db);
        extract_from_module(db, &sema, root);
    }
}

fn extract_from_module(db: &RootDatabase, sema: &Semantics<'_, RootDatabase>, module: Module) {
    for decl in module.declarations(db) {
        // Extract symbol info
        let name = decl.name(db);

        // Get source location
        if let Some(source) = get_source(db, &decl) {
            let range = source.value.syntax().text_range();
            // ...
        }

        // Find dependencies by walking the syntax
        if let Some(source) = get_source(db, &decl) {
            for node in source.value.syntax().descendants() {
                if let Some(path) = ast::Path::cast(node) {
                    if let Some(PathResolution::Def(dep)) = sema.resolve_path(&path) {
                        if is_local_def(db, dep) {
                            // Record dependency: decl -> dep
                        }
                    }
                }
            }
        }

        // Recurse into submodules
        if let ModuleDef::Module(m) = decl {
            extract_from_module(db, sema, m);
        }
    }
}
```
