# ra_ap_ide_db

Defines the core data structure representing IDE state - `RootDatabase`. This is the main database type used throughout rust-analyzer, combining semantic analysis (`HirDatabase`) with fuzzy search (`SymbolsDatabase`).

## Overview

This crate provides:
1. **RootDatabase** - The concrete database type combining all rust-analyzer capabilities
2. **Definition** - A unified type representing any named definition in code
3. **NameClass / NameRefClass** - Classification of name definitions and references
4. **Symbol search and indexing** - Via the `symbol_index` module
5. **Source changes** - Text edits and file changes
6. **Helper utilities** - Path transforms, imports, rename logic, etc.

## Re-exports

```rust
pub use base_db::{self, FxIndexMap, FxIndexSet, LibraryRoots, LocalRoots};
pub use hir::{ChangeWithProcMacros, EditionedFileId};
pub use line_index;
pub use span::{self, FileId};
pub use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
```

---

## RootDatabase

The main database struct implementing all analysis traits.

```rust
#[salsa_macros::db]
pub struct RootDatabase {
    storage: ManuallyDrop<salsa::Storage<Self>>,
    files: Arc<Files>,
    crates_map: Arc<CratesMap>,
    nonce: Nonce,
}
```

**Implemented Traits:**
- `salsa::Database` - Salsa database
- `SourceDatabase` - File text and source roots
- `RootQueryDb` - Parsing queries
- `LineIndexDatabase` - Line/column mapping
- `DefDatabase` - Definition resolution (from hir)
- `ExpandDatabase` - Macro expansion (from hir)
- `HirDatabase` - High-level semantic analysis (from hir)

### Construction

```rust
impl RootDatabase {
    /// Create a new database with optional LRU capacity
    pub fn new(lru_capacity: Option<u16>) -> RootDatabase;
}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        RootDatabase::new(None)
    }
}
```

### Methods

| Method | Description |
|--------|-------------|
| `enable_proc_attr_macros()` | Enable proc-macro attribute expansion |
| `update_base_query_lru_capacities(cap)` | Update LRU cache sizes |
| `update_lru_capacities(capacities)` | Update specific LRU capacities by name |

---

## LineIndexDatabase

Query trait for line/column position mapping.

```rust
#[query_group::query_group]
pub trait LineIndexDatabase: base_db::RootQueryDb {
    fn line_index(&self, file_id: FileId) -> Arc<LineIndex>;
}
```

---

## Position Types

```rust
/// A position in a file (line, column)
pub type FilePosition = FilePositionWrapper<FileId>;

/// A range in a file (start, end)
pub type FileRange = FileRangeWrapper<FileId>;
```

---

## Definition

A unified enum representing any named definition in Rust code. This is the core type for "find definition", "find references", rename, and similar operations.

```rust
#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum Definition {
    Macro(Macro),
    Field(Field),
    TupleField(TupleField),
    Module(Module),
    Crate(Crate),
    Function(Function),
    Adt(Adt),
    Variant(Variant),
    Const(Const),
    Static(Static),
    Trait(Trait),
    TypeAlias(TypeAlias),
    SelfType(Impl),
    GenericParam(GenericParam),
    Local(Local),
    Label(Label),
    DeriveHelper(DeriveHelper),
    BuiltinType(BuiltinType),
    BuiltinLifetime(StaticLifetime),
    BuiltinAttr(BuiltinAttr),
    ToolModule(ToolModule),
    ExternCrateDecl(ExternCrateDecl),
    InlineAsmRegOrRegClass(()),
    InlineAsmOperand(InlineAsmOperand),
}
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `canonical_module_path` | `fn(&self, db) -> Option<impl Iterator<Item = Module>>` | Get path from crate root |
| `krate` | `fn(&self, db) -> Option<Crate>` | Get containing crate |
| `module` | `fn(&self, db) -> Option<Module>` | Get containing module (parent for modules) |
| `enclosing_definition` | `fn(&self, db) -> Option<Definition>` | Get enclosing definition |
| `visibility` | `fn(&self, db) -> Option<Visibility>` | Get visibility |
| `name` | `fn(&self, db) -> Option<Name>` | Get name |
| `docs` | `fn(&self, db, famous_defs, display_target) -> Option<Documentation>` | Get documentation |
| `label` | `fn(&self, db, display_target) -> String` | Get display label |

### Conversions

`Definition` implements `From` for many HIR types:
- `Field`, `Module`, `Function`, `Adt`, `Variant`, `Const`, `Static`, `Trait`, `TypeAlias`
- `BuiltinType`, `Local`, `GenericParam`, `Label`, `Macro`, `ExternCrateDecl`
- `Impl` (becomes `SelfType`)
- `PathResolution`, `ModuleDef`, `AssocItem`, `VariantDef`, `GenericDef`, `DocLinkDef`

---

## IdentClass

Classification of an identifier (name or reference) in code.

```rust
pub enum IdentClass<'db> {
    NameClass(NameClass<'db>),
    NameRefClass(NameRefClass<'db>),
    Operator(OperatorClass),
}
```

### Methods

| Method | Description |
|--------|-------------|
| `classify_node(sema, node)` | Classify a syntax node |
| `classify_token(sema, token)` | Classify a token (uses parent node) |
| `classify_lifetime(sema, lifetime)` | Classify a lifetime |
| `definitions()` | Get definitions with optional generic substitutions |
| `definitions_no_ops()` | Get definitions, excluding operators |

---

## NameClass

Classification of a name definition (an `ast::Name`). This is the canonical way rust-analyzer determines what a name defines.

```rust
pub enum NameClass<'db> {
    /// Simple definition
    Definition(Definition),

    /// A constant used in pattern position (e.g., `None` in `if let None = x`)
    ConstReference(Definition),

    /// Field shorthand in patterns (e.g., `field` in `Foo { field }`)
    PatFieldShorthand {
        local_def: Local,
        field_ref: Field,
        adt_subst: GenericSubstitution<'db>,
    },
}
```

### Methods

| Method | Description |
|--------|-------------|
| `defined()` | Get the definition (None for ConstReference) |
| `classify(sema, name)` | Classify an `ast::Name` |
| `classify_lifetime(sema, lifetime)` | Classify a lifetime definition |

### Canonical Usage (from goto_definition.rs)

```rust
// Find what a name defines
fn get_definition_at_name(
    sema: &Semantics<'_, RootDatabase>,
    name: &ast::Name,
) -> Option<Definition> {
    match NameClass::classify(sema, name)? {
        NameClass::Definition(def) => Some(def),
        NameClass::ConstReference(def) => {
            // e.g., `None` in pattern - points to the const
            Some(def)
        }
        NameClass::PatFieldShorthand { local_def, .. } => {
            // `Foo { field }` pattern - both defines local AND references field
            // Usually we want the local definition here
            Some(Definition::Local(local_def))
        }
    }
}
```

---

## NameRefClass

Classification of a name reference (an `ast::NameRef`). This is the canonical way rust-analyzer determines what a name refers to.

```rust
pub enum NameRefClass<'db> {
    /// Simple reference to a definition
    Definition(Definition, Option<GenericSubstitution<'db>>),

    /// Field shorthand in expressions (e.g., `field` in `Foo { field }`)
    FieldShorthand {
        local_ref: Local,
        field_ref: Field,
        adt_subst: GenericSubstitution<'db>,
    },

    /// Extern crate without rename (both declaration and reference)
    ExternCrateShorthand {
        decl: ExternCrateDecl,
        krate: Crate,
    },
}
```

### Methods

| Method | Description |
|--------|-------------|
| `classify(sema, name_ref)` | Classify an `ast::NameRef` |
| `classify_lifetime(sema, lifetime)` | Classify a lifetime reference |

### Canonical Usage (from references.rs)

```rust
// Find what a name reference refers to
fn find_referenced_definition(
    sema: &Semantics<'_, RootDatabase>,
    name_ref: &ast::NameRef,
) -> Option<Definition> {
    match NameRefClass::classify(sema, name_ref)? {
        NameRefClass::Definition(def, _subst) => {
            // Normal reference - def is what it points to
            // _subst contains generic substitution info (e.g., Vec<i32>)
            Some(def)
        }
        NameRefClass::FieldShorthand { field_ref, .. } => {
            // `Foo { field }` expression - references both local AND field
            // Usually we want the field for "go to definition"
            Some(Definition::Field(field_ref))
        }
        NameRefClass::ExternCrateShorthand { krate, .. } => {
            // `extern crate foo;` - the `foo` name
            Some(Definition::Crate(krate))
        }
    }
}
```

### Why Use NameRefClass Instead of resolve_path?

`NameRefClass::classify` handles cases that `Semantics::resolve_path` doesn't:

1. **Not all references are paths**: `Foo { field }` - `field` is a NameRef but not a path
2. **Pattern constants**: `match x { None => }` - `None` resolves specially
3. **Field shorthands**: Need to distinguish local vs field reference
4. **Extern crates**: Different resolution rules
5. **Generic substitutions**: Returns substitution info for generic types

**Recommended pattern for dependency analysis:**

```rust
fn collect_dependencies(sema: &Semantics<'_, RootDatabase>, syntax: &SyntaxNode) {
    // Collect all NameRef nodes (references to other things)
    for name_ref in syntax.descendants().filter_map(ast::NameRef::cast) {
        if let Some(NameRefClass::Definition(def, _)) = NameRefClass::classify(sema, &name_ref) {
            // def is the Definition this name refers to
            process_dependency(def);
        }
    }
}
```

---

## OperatorClass

Classification of operators that resolve to functions/types.

```rust
pub enum OperatorClass {
    Range(Struct),      // Range operator resolves to Range struct
    Await(Function),    // .await resolves to poll function
    Prefix(Function),   // Prefix operators (!, -, *)
    Index(Function),    // Index operator []
    Try(Function),      // ? operator
    Bin(Function),      // Binary operators
}
```

---

## SymbolKind

Classification of symbol types for display/filtering.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymbolKind {
    Attribute, BuiltinAttr, Const, ConstParam, CrateRoot, Derive,
    DeriveHelper, Enum, Field, Function, Method, Impl,
    InlineAsmRegOrRegClass, Label, LifetimeParam, Local, Macro,
    ProcMacro, Module, SelfParam, SelfType, Static, Struct,
    ToolModule, Trait, TypeAlias, TypeParam, Union, ValueParam, Variant,
}
```

### Methods

| Method | Description |
|--------|-------------|
| `from_module_def(db, ModuleDef)` | Create from module definition |

---

## Severity

Diagnostic severity levels.

```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Severity {
    Error,
    Warning,
    WeakWarning,
    Allow,
}
```

---

## SnippetCap

Capability marker for snippet support.

```rust
pub struct SnippetCap { _private: () }

impl SnippetCap {
    pub const fn new(allow_snippets: bool) -> Option<SnippetCap>;
}
```

---

## Ranker

Token ranking utility for finding the best token at a position.

```rust
pub struct Ranker<'a> {
    pub kind: SyntaxKind,
    pub text: &'a str,
    pub ident_kind: bool,
}

impl Ranker<'_> {
    pub const MAX_RANK: usize = 0b1110;
    pub fn from_token(token: &SyntaxToken) -> Self;
    pub fn rank_token(&self, tok: &SyntaxToken) -> usize;
}
```

---

## Public Modules

| Module | Description |
|--------|-------------|
| `active_parameter` | Find active parameter in function calls |
| `assists` | Assist/quick-fix infrastructure |
| `defs` | Definition classification (this document) |
| `documentation` | Documentation extraction |
| `famous_defs` | Well-known definitions (Option, Result, etc.) |
| `helpers` | Miscellaneous helpers |
| `items_locator` | Locate items by name |
| `label` | Label generation |
| `path_transform` | Path substitution/transformation |
| `prime_caches` | Cache warming |
| `ra_fixture` | Test fixture creation |
| `range_mapper` | Range mapping utilities |
| `rename` | Rename refactoring logic |
| `rust_doc` | Rust documentation link handling |
| `search` | Reference search |
| `source_change` | Source code modification |
| `symbol_index` | Symbol indexing for search |
| `text_edit` | Text editing primitives |
| `traits` | Trait-related utilities |
| `ty_filter` | Type filtering utilities |
| `use_trivial_constructor` | Trivial constructor detection |

### Submodules

**`imports`:**
- `import_assets` - Import path resolution
- `insert_use` - Use statement insertion
- `merge_imports` - Import merging

**`generated`:**
- `lints` - Generated lint definitions

**`syntax_helpers`:**
- `format_string` - Format string parsing
- `format_string_exprs` - Format string expression handling
- `tree_diff` - Syntax tree diffing
- `node_ext` - Node extension utilities
- `suggest_name` - Name suggestion utilities
- `prettify_macro_expansion` - Macro expansion formatting (from hir)
- `LexedStr` - Lexed string utilities (from parser)

---

## search Module - FindUsages

The `search` module provides the canonical way to find all references to a definition. It uses a **two-phase search** for efficiency.

### SearchScope

```rust
pub struct SearchScope {
    // Internally stores file_id -> text_range mappings
}

impl SearchScope {
    pub fn single_file(file_id: EditionedFileId) -> SearchScope;
    pub fn files(files: &[EditionedFileId]) -> SearchScope;
    pub fn module_and_children(db: &dyn HirDatabase, module: Module) -> SearchScope;
    pub fn crate_graph(db: &dyn HirDatabase) -> SearchScope;
    pub fn krate(db: &dyn HirDatabase, krate: Crate) -> SearchScope;
}
```

### FindUsages

Fluent API for finding all usages of a definition:

```rust
impl Definition {
    pub fn usages<'a>(&self, sema: &'a Semantics<'_, RootDatabase>) -> FindUsages<'a>;
}

impl<'a> FindUsages<'a> {
    pub fn set_scope(self, scope: SearchScope) -> Self;
    pub fn include_self_refs(self) -> Self;
    pub fn search_self_mod(self) -> Self;
    pub fn all(self) -> UsageSearchResult;
}
```

### UsageSearchResult

```rust
pub struct UsageSearchResult {
    pub references: FxHashMap<EditionedFileId, Vec<FileReference>>,
}

pub struct FileReference {
    pub range: TextRange,
    pub category: ReferenceCategory,
}

pub enum ReferenceCategory {
    Read,
    Write,
    Import,
    Test,
}
```

### Canonical Usage (from references.rs)

```rust
fn find_all_references(
    sema: &Semantics<'_, RootDatabase>,
    def: Definition,
    krate: Crate,
) -> Vec<FileRange> {
    def.usages(sema)
        .set_scope(SearchScope::krate(sema.db, krate))
        .include_self_refs()
        .all()
        .references
        .into_iter()
        .flat_map(|(file_id, refs)| {
            refs.into_iter().map(move |r| FileRange {
                file_id: file_id.into(),
                range: r.range,
            })
        })
        .collect()
}
```

### How It Works (Two-Phase Search)

1. **Fast text search**: Scans file text for the definition's name using `memchr`
2. **Precise semantic resolution**: For each potential match, uses `NameRefClass::classify` to confirm

This is much faster than walking the entire syntax tree for each file.

---

## Usage in tarjanize

tarjanize-extract uses ra_ap_ide_db for:

1. **RootDatabase** - The main database type for analysis
2. **Definition** - To represent symbols found during extraction
3. **NameClass/NameRefClass** - To classify identifiers when finding dependencies
4. **SymbolKind** - To categorize extracted symbols

The `RootDatabase` is typically obtained via `ra_ap_load_cargo::load_workspace()` and provides all semantic analysis capabilities.

### Potential Improvements

Based on rust-analyzer patterns, tarjanize-extract could:

1. **Use NameRefClass instead of resolve_path** for more robust name resolution
2. **Use FindUsages** for reverse dependency analysis (what depends on X?)
3. **Use the Definition type** as a unified representation instead of ModuleDef + Impl
