# rust-analyzer API Reference

This document explains how to use rust-analyzer's APIs for symbol extraction and dependency analysis. These patterns are derived from studying rust-analyzer's own IDE features (goto_definition, find_references, call_hierarchy).

**Sources:**
- [goto_definition.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide/src/goto_definition.rs)
- [references.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide/src/references.rs)
- [call_hierarchy.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide/src/call_hierarchy.rs)
- [defs.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide-db/src/defs.rs)

---

## Crate Overview

| Crate | Purpose |
|-------|---------|
| `ra_ap_ide_db` | `RootDatabase` (the main analysis database), `NameRefClass`/`NameClass` for name classification |
| `ra_ap_hir` | High-level IR: `Crate`, `Module`, `ModuleDef`, `Impl`, `Semantics` |
| `ra_ap_syntax` | Syntax trees: `SyntaxNode`, `AstNode`, AST types like `ast::Fn`, `ast::NameRef` |
| `ra_ap_load_cargo` | `load_workspace_at()` to load a Cargo workspace into a database |

---

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

**Canonical pattern for dependency analysis:**

```rust
for name_ref in syntax.descendants().filter_map(ast::NameRef::cast) {
    if let Some(NameRefClass::Definition(def, _)) = NameRefClass::classify(sema, &name_ref) {
        // def is a Definition - convert to your dependency type
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
    TypeAlias(TypeAlias),
    BuiltinType(BuiltinType),
    SelfType(Impl),
    Local(Local),
    GenericParam(GenericParam),
    // ... and more
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

### Resolving Paths

```rust
use ra_ap_hir::PathResolution;

fn resolve_path(sema: &Semantics<'_, RootDatabase>, path: &ast::Path) {
    if let Some(resolution) = sema.resolve_path(path) {
        match resolution {
            PathResolution::Def(module_def) => {
                // Resolved to a module-level definition
            }
            PathResolution::Local(local) => {
                // Resolved to a local variable
            }
            PathResolution::TypeParam(type_param) => {
                // Resolved to a type parameter
            }
            PathResolution::SelfType(impl_) => {
                // Resolved to Self in an impl block
            }
            _ => {}
        }
    }
}
```

---

## Walking Syntax Trees for Dependencies

To find all dependencies (references to other symbols), you need to walk the syntax tree and resolve each reference.

### The parse_or_expand Pattern

**Problem**: `HasSource::source()` returns a syntax node, but `Semantics::resolve_path()` only works on nodes from trees that Semantics has cached. The two trees have identical content but are different allocations.

**Solution**: Use `sema.parse_or_expand()` to get a cached tree, then find the equivalent node by matching text ranges.

```rust
use ra_ap_hir::{HasSource, HirFileId, Semantics};
use ra_ap_syntax::{AstNode, SyntaxNode};

/// Find a node in a syntax tree by matching its text range.
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
    // 1. Get the cached tree from Semantics
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
```

### Handling Macro Expansions

When working with macro expansions, results often come as `InFile<T>`:

```rust
pub struct InFile<T> {
    pub file_id: HirFileId,  // Might be a macro expansion
    pub value: T,
}

// Get the original file (traces through expansions)
let original = file_id.original_file(db);
```

---

## Working with Impl Blocks

Impl blocks require special handling because they're NOT part of `ModuleDef` - they're anonymous.

### Getting Impl Blocks

```rust
for impl_ in module.impl_defs(db) {
    process_impl(db, impl_);
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
    let self_ty_name = self_ty.display(db, display_target).to_string();

    if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    }
}
```

---

## Collapsing Associated Items to Containers

For dependency analysis, associated items should be collapsed to their containers (impl blocks or traits). You can't have a method without its impl block.

```rust
use ra_ap_hir::{AsAssocItem, AssocItemContainer, Function};

fn get_container_for_method(db: &RootDatabase, func: Function) -> Option<AssocItemContainer> {
    let assoc = func.as_assoc_item(db)?;
    Some(assoc.container(db))
}
```

Consider `impl Clone for Foo { fn clone(&self) { ... } }`:
- The method `clone` cannot exist without the impl block
- Dependencies on `clone()` are really dependencies on `impl Clone for Foo`

---

## Filtering to Workspace Members

Only track dependencies within the workspace, not to external crates:

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
        // ... etc for other variants
        ModuleDef::BuiltinType(_) => false,
    }
}
```

---

## Common Pitfalls

1. **Not Caching Syntax Trees**: Always parse through `Semantics`, not directly from the database.

2. **Mixing Semantics Instances**: Each `Semantics` has its own cache. Don't use nodes from one with another.

3. **Forgetting About Macros**: Use `original_file(db)` to trace back through macro expansions.

4. **Edition Mismatches**: Different crates can have different Rust editions.

---

## Performance Considerations

1. **Batch operations**: Queries are memoized. Querying the same information twice is cheap.
2. **Parallelism**: The database is thread-safe. You can use rayon for parallel analysis.
3. **Source roots**: Library crates are marked as immutable, enabling better caching.
