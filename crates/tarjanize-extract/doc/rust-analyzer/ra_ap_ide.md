# ra_ap_ide

IDE-centric APIs for rust-analyzer. This is the top-level crate that provides human-friendly analysis capabilities suitable for displaying to users in editors.

## Overview

**Design Philosophy:**
- Operates with files and text ranges, returns results as Strings for human display
- Powered by `RootDatabase` and `hir` for semantic analysis
- IDE-specific features (completion, hover, etc.) implemented here
- API designed to be language-server-protocol independent

---

## Crate Architecture: Where ra_ap_ide Fits

rust-analyzer is organized as a stack of crates, each building on the ones below:

```
┌─────────────────────────────────────────────────────────────┐
│  ra_ap_ide                                                  │  ← You are here
│  IDE features: hover, completion, goto, rename, etc.        │
│  Works with FilePosition, returns human-readable strings    │
├─────────────────────────────────────────────────────────────┤
│  ra_ap_ide_db                                               │
│  RootDatabase, Definition, search indexes                   │
│  Bridges hir types to IDE concepts                          │
├─────────────────────────────────────────────────────────────┤
│  ra_ap_hir                                                  │
│  High-level semantic model: Crate, Module, Function, etc.   │
│  OO-style API hiding compiler internals                     │
├─────────────────────────────────────────────────────────────┤
│  ra_ap_hir_def, ra_ap_hir_ty, ra_ap_hir_expand              │
│  Compiler internals: name resolution, type inference        │
│  ECS-style, not meant for direct use                        │
├─────────────────────────────────────────────────────────────┤
│  ra_ap_base_db, ra_ap_syntax                                │
│  Foundation: file storage, parsing, syntax trees            │
└─────────────────────────────────────────────────────────────┘
```

**Choosing the right crate depends on your task:**

| Task | Crate | Why |
|------|-------|-----|
| Build an IDE/editor plugin | `ra_ap_ide` | Provides ready-made features (hover, goto, completion) |
| Extract symbol information | `ra_ap_hir` + `ra_ap_ide` | hir for traversal, ide for source locations (`TryToNav`) |
| Build custom analysis tools | `ra_ap_hir` + `Semantics` | Fine-grained control over traversal and resolution |
| Implement new IDE features | `ra_ap_ide_db` | Access to Definition, search indexes, etc. |
| Work with syntax only | `ra_ap_syntax` | No semantic analysis needed |

**How they connect:**

1. **ra_ap_ide** wraps everything in `Analysis`/`AnalysisHost` and provides the public API
2. **ra_ap_ide** delegates to sub-crates for specific features:
   - `ra_ap_ide_assists` for code actions
   - `ra_ap_ide_completion` for autocomplete
   - `ra_ap_ide_diagnostics` for error reporting
3. All these use **ra_ap_ide_db**'s `RootDatabase` as their database
4. **ra_ap_ide_db** uses **ra_ap_hir** for semantic queries
5. **ra_ap_hir** provides the `Semantics` type that bridges syntax trees to semantic information

**For tarjanize specifically:**
- We use `ra_ap_load_cargo` to load the workspace into a `RootDatabase`
- We use `ra_ap_hir` (via `Semantics`) for module traversal and semantic analysis
- We use `ra_ap_ide` for `TryToNav` to get source locations (file paths, ranges for cost calculation)
- We don't use `ra_ap_ide`'s high-level features (hover, completion, etc.) - just the utilities

---

## The Incremental Computation Model

Understanding the `AnalysisHost`/`Analysis` split requires understanding salsa's incremental computation model.

**The problem**: rust-analyzer needs to answer semantic questions about code while the code is being modified. A long-running query (like "find all references") might be in progress when a new edit arrives. If we apply the edit immediately, the running query would see inconsistent state mid-computation (violating repeatable read).

**The solution**: All queries run against immutable **snapshots**. When an edit arrives, all pending queries are **cancelled** before the change is applied. Cancellation works by salsa unwinding with a special `Cancelled` value, which is caught at the API boundary.

### AnalysisHost vs Analysis

```rust
// AnalysisHost: The mutable owner of the database
pub struct AnalysisHost {
    db: RootDatabase,
}

// Analysis: An immutable snapshot for querying
pub struct Analysis {
    db: RootDatabase,  // Clone of the database at a point in time
}
```

**AnalysisHost** is the mutable container that holds the current state of the world. You apply changes to it, and it manages the underlying salsa database.

**Analysis** is an immutable snapshot obtained from `host.analysis()`. All semantic queries go through `Analysis`. If you call `host.apply_change()` while an `Analysis` snapshot exists, all queries on that snapshot will be cancelled.

**Why the split?**
- The host is `&mut self` for applying changes
- Analysis snapshots are `&self` and can be shared across threads
- Multiple threads can query the same snapshot concurrently
- Changes invalidate all outstanding snapshots

---

## Workflow: From Cargo Workspace to Analysis

Here's how you typically get from a Cargo workspace to making queries:

```rust
use ra_ap_load_cargo::{load_workspace, LoadCargoConfig, ProcMacroServerChoice};
use ra_ap_ide::{AnalysisHost, Analysis, FilePosition};
use ra_ap_project_model::{CargoConfig, ProjectManifest, ProjectWorkspace};

// 1. Find and parse the Cargo.toml
let manifest = ProjectManifest::discover_single(&workspace_path)?;

// 2. Load the project model (parses Cargo.toml, resolves dependencies)
let cargo_config = CargoConfig::default();
let workspace = ProjectWorkspace::load(manifest, &cargo_config, progress)?;

// 3. Convert to rust-analyzer's internal representation
let load_config = LoadCargoConfig {
    load_out_dirs_from_check: true,  // Run `cargo check` for build.rs outputs
    with_proc_macro_server: ProcMacroServerChoice::Sysroot,
    prefill_caches: false,
};
let (db, vfs, proc_macro_server) = load_workspace(workspace, &load_config)?;

// 4. Wrap in AnalysisHost
let host = AnalysisHost::with_database(db);

// 5. Get an Analysis snapshot for querying
let analysis: Analysis = host.analysis();

// 6. Make queries
let position = FilePosition { file_id, offset };
let hover = analysis.hover(&hover_config, position.into())?;
```

**Key types in the flow:**
- `FileId`: Opaque identifier for a file in the VFS (virtual file system)
- `FilePosition`: A `FileId` + byte `offset` within that file
- `FileRange`: A `FileId` + `TextRange` (start..end offsets)

---

## Cancellable<T> and Query Cancellation

```rust
pub type Cancellable<T> = Result<T, Cancelled>;
```

Almost every `Analysis` method returns `Cancellable<T>`. This is because:

1. Queries can be long-running (e.g., finding all references in a large codebase)
2. When `host.apply_change()` is called, all outstanding `Analysis` snapshots become invalid
3. Any query in progress will receive a `Cancelled` error instead of completing

**Handling cancellation:**

```rust
match analysis.goto_definition(position, &config) {
    Ok(Some(nav_targets)) => {
        // Success - use the navigation targets
    }
    Ok(None) => {
        // No definition found at this position
    }
    Err(Cancelled) => {
        // Query was cancelled due to a change - retry with fresh snapshot
        let analysis = host.analysis();
        // ... retry the query
    }
}
```

**When cancellation happens:**
- User types a character → edit applied → outstanding queries cancelled
- File saved → change applied → outstanding queries cancelled
- `host.trigger_cancellation()` called explicitly

---

## Core Types

### AnalysisHost

Stores the current mutable state of the world.

```rust
impl AnalysisHost {
    /// Create a new host with optional LRU cache capacity for queries
    pub fn new(lru_capacity: Option<u16>) -> AnalysisHost;

    /// Wrap an existing database
    pub fn with_database(db: RootDatabase) -> AnalysisHost;

    /// Get an immutable snapshot for querying.
    /// The snapshot remains valid until apply_change is called.
    pub fn analysis(&self) -> Analysis;

    /// Apply changes to the world state.
    /// This cancels all outstanding Analysis snapshots.
    pub fn apply_change(&mut self, change: ChangeWithProcMacros);

    /// Explicitly cancel all running queries.
    /// Useful when you know you're about to apply a change.
    pub fn trigger_cancellation(&mut self);

    /// Evict LRU cache entries and collect garbage.
    /// Also triggers cancellation.
    pub fn trigger_garbage_collection(&mut self);

    /// Access the underlying database directly (rarely needed)
    pub fn raw_database(&self) -> &RootDatabase;
    pub fn raw_database_mut(&mut self) -> &mut RootDatabase;
}
```

### Analysis

Immutable snapshot for semantic queries.

```rust
impl Analysis {
    /// Quick single-file analysis (no deps, no stdlib).
    /// Useful for testing or simple tools.
    pub fn from_single_file(text: String) -> (Analysis, FileId);
}
```

---

## Navigation Methods

These methods help navigate through code. Understanding when to use each:

### goto_definition

```rust
pub fn goto_definition(
    &self,
    position: FilePosition,
    config: &GotoDefinitionConfig<'_>,
) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>>
```

**What it does**: Navigates to where an identifier is *defined/implemented*.

**Examples:**
- On a function call `foo()` → goes to `fn foo() { ... }`
- On a type `Foo` → goes to `struct Foo { ... }` or `type Foo = ...`
- On an outline module `use foo::bar` → goes to `foo/bar.rs` (the module's source file)
- On a trait method call → goes to the impl method (not the trait declaration)

### goto_declaration

```rust
pub fn goto_declaration(
    &self,
    position: FilePosition,
    config: &GotoDefinitionConfig<'_>,
) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>>
```

**What it does**: Same as `goto_definition` with key exceptions for finding *declarations* rather than *implementations*:

**Differences from goto_definition:**
1. **Outline modules**: Goes to `mod foo;` declaration, not the module's source file
2. **Trait assoc items**: Goes to the trait declaration, not the impl
3. **Fields in patterns**: Goes to the struct field declaration

**Example:**
```rust
trait Speak {
    fn speak(&self);  // ← goto_declaration lands here
}
impl Speak for Dog {
    fn speak(&self) { println!("woof"); }  // ← goto_definition lands here
}
fn main() {
    dog.speak();  // cursor here
}
```

**When to use which:**
- `goto_definition`: "Show me the implementation"
- `goto_declaration`: "Show me the interface/contract"

### goto_type_definition

```rust
pub fn goto_type_definition(
    &self,
    position: FilePosition,
) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>>
```

**What it does**: Navigates to the *type* of an identifier, not the identifier itself.

**Examples:**
```rust
let x: Foo = ...;
//  ^ cursor here → goes to `struct Foo`

let y: &Bar = ...;
//  ^ cursor here → goes to `struct Bar` (strips references)

let z: Vec<Baz> = ...;
//  ^ cursor here → goes to `Vec`, `Baz` (all types in the signature)
```

**When to use**: "I have a variable, show me what type it is"

### goto_implementation

```rust
pub fn goto_implementation(
    &self,
    config: &GotoImplementationConfig,
    position: FilePosition,
) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>>
```

**What it does**: From a trait or abstract type, find all implementations.

**Examples:**
- On `trait Foo` → shows all `impl Foo for X` blocks
- On `struct Bar` → shows all `impl Bar` and `impl Trait for Bar` blocks

---

## UpmappingResult: Handling Macro Expansions

When navigating to something defined inside a macro, there are potentially two relevant locations:

```rust
pub struct UpmappingResult<T> {
    /// Location at the macro call site (where the macro was invoked)
    pub call_site: T,
    /// Location at the macro definition site (where the macro is defined)
    pub def_site: Option<T>,
}
```

**Why two locations?**

```rust
macro_rules! make_struct {
    ($name:ident) => {
        struct $name { value: i32 }  // ← def_site: inside macro definition
    };
}

make_struct!(Foo);  // ← call_site: where macro is invoked
```

When you "go to definition" of `Foo`:
- `call_site` points to `make_struct!(Foo);` - usually what you want
- `def_site` points to `struct $name` inside the macro - useful for understanding the macro

**When is def_site Some vs None?**
- `Some`: Item was created by a macro expansion
- `None`: Item is not macro-generated (normal code)

**Usage patterns:**

```rust
let nav = def.try_to_nav(sema)?;

// Most common: just use call site
let target = nav.call_site();

// Get all locations (for "find all definitions" style features)
let all_targets: Vec<_> = nav.collect();

// Iterate over both if present
for target in nav {
    // ...
}
```

---

## NavigationTarget

Represents a clickable UI element that navigates to code.

```rust
pub struct NavigationTarget {
    /// File containing the target
    pub file_id: FileId,

    /// Full range including doc comments, attributes, body.
    /// Use this to answer "is the cursor inside this element?"
    pub full_range: TextRange,

    /// Range of just the identifier (name).
    /// Place the cursor here when navigating.
    /// Contained within full_range.
    pub focus_range: Option<TextRange>,

    /// Display name of the item
    pub name: Symbol,

    /// Kind (function, struct, etc.)
    pub kind: Option<SymbolKind>,

    /// Name of containing item (e.g., "impl Foo" for a method)
    pub container_name: Option<Symbol>,

    /// Short description (typically the signature)
    pub description: Option<String>,

    /// Documentation
    pub docs: Option<Documentation<'static>>,

    /// Alternative name (for re-exports)
    pub alias: Option<Symbol>,
}
```

**The full_range vs focus_range distinction:**

```rust
/// This is a doc comment          ─┐
#[derive(Debug)]                   │ full_range
pub fn example_function() {        │
    // function body               │
}                                  ─┘
       ^^^^^^^^^^^^^^^^  ← focus_range (just the name)
```

- `full_range`: The entire syntactic element including docs, attributes, body
- `focus_range`: Just the identifier - where the cursor should land

```rust
impl NavigationTarget {
    /// Get focus_range if available, otherwise fall back to full_range
    pub fn focus_or_full_range(&self) -> TextRange;

    /// Create a NavigationTarget for a module's declaration (mod foo;)
    /// rather than its definition (the file contents)
    pub fn from_module_to_decl(db: &RootDatabase, module: Module) -> UpmappingResult<NavigationTarget>;
}
```

### TryToNav Trait

Convert HIR types to NavigationTarget:

```rust
pub trait TryToNav {
    fn try_to_nav(&self, sema: &Semantics<'_, RootDatabase>)
        -> Option<UpmappingResult<NavigationTarget>>;
}

// Implemented for: Definition, ModuleDef, Function, Struct, Enum,
// Trait, Impl, Const, Static, TypeAlias, Macro, Field, etc.
```

---

## StaticIndex: Complete Codebase Analysis

`StaticIndex` provides a complete, pre-computed index of all symbols and references in a codebase. It's designed for:
- Generating LSIF/SCIP indexes (for code intelligence in browsers)
- Read-only code browsers
- Batch analysis tools

### The Token Model

Despite the name, "token" in StaticIndex doesn't mean a lexical token. It means a **unique definition**:

```rust
pub struct TokenId(usize);  // Opaque handle to a definition

pub struct TokenStaticData {
    pub documentation: Option<Documentation<'static>>,
    pub hover: Option<HoverResult>,

    /// Range of the identifier (e.g., function name)
    pub definition: Option<FileRange>,

    /// Range of the entire definition (e.g., whole function including body)
    pub definition_body: Option<FileRange>,

    /// All references to this definition
    pub references: Vec<ReferenceData>,

    pub moniker: Option<MonikerResult>,
    pub display_name: Option<String>,
    pub signature: Option<String>,
    pub kind: SymbolInformationKind,
}

pub struct ReferenceData {
    pub range: FileRange,
    pub is_definition: bool,  // True if this reference IS the definition
}
```

**Key insight**: Each `Definition` (function, struct, etc.) gets exactly one `TokenId`. When the same definition is referenced multiple times, each reference is added to that token's `references` list. This is the deduplication.

### StaticIndexedFile

```rust
pub struct StaticIndexedFile {
    pub file_id: FileId,
    pub folds: Vec<Fold>,           // Code folding ranges
    pub inlay_hints: Vec<InlayHint>,
    pub tokens: Vec<(TextRange, TokenId)>,  // Each token occurrence in this file
}
```

### VendoredLibrariesConfig

Controls which files get indexed:

```rust
pub enum VendoredLibrariesConfig<'a> {
    /// Include vendored libraries (library files inside workspace_root)
    Included { workspace_root: &'a VfsPath },

    /// Exclude all library files (only local workspace code)
    Excluded,
}
```

**What this means:**
- External dependencies (from `~/.cargo/registry`) are always excluded
- "Vendored" libraries are dependencies copied into your workspace (e.g., `vendor/` directory)
- With `Included`, vendored libs are indexed; with `Excluded`, only your code is indexed

### Computing StaticIndex

```rust
let index = StaticIndex::compute(
    &analysis,
    VendoredLibrariesConfig::Excluded,  // Just workspace code
);

// Iterate over all indexed files
for file in &index.files {
    println!("File: {:?}", file.file_id);

    // Each token occurrence in this file
    for (range, token_id) in &file.tokens {
        let data = index.tokens.get(*token_id).unwrap();

        // Check if this occurrence is the definition
        let is_def = data.definition
            .map(|d| d.file_id == file.file_id && d.range == *range)
            .unwrap_or(false);

        if is_def {
            println!("  Definition: {:?} at {:?}", data.display_name, range);
        } else {
            println!("  Reference to {:?} at {:?}", data.display_name, range);
        }
    }
}

// Or iterate by definition
for (token_id, data) in index.tokens.iter() {
    println!("Symbol: {:?}", data.display_name);
    println!("  Defined at: {:?}", data.definition);
    println!("  {} references", data.references.len());
}
```

---

## Other Analysis Methods

### References

```rust
pub fn find_all_refs(
    &self,
    position: FilePosition,
    config: &FindAllRefsConfig<'_>,
) -> Cancellable<Option<Vec<ReferenceSearchResult>>>
```

Find all references to the symbol at the given position.

```rust
pub struct ReferenceSearchResult {
    /// The declaration (None for primitives like `i32`)
    pub declaration: Option<Declaration>,
    /// References grouped by file
    pub references: IntMap<FileId, Vec<(TextRange, ReferenceCategory)>>,
}

pub struct Declaration {
    pub nav: NavigationTarget,
    pub is_mut: bool,  // For mutable variables
}
```

### Hover

```rust
pub fn hover(
    &self,
    config: &HoverConfig<'_>,
    range: FileRange,
) -> Cancellable<Option<RangeInfo<HoverResult>>>
```

Get hover information (documentation, type signature, etc.) for the given range.

### Diagnostics

```rust
/// Parse errors only - fast, no semantic analysis
pub fn syntax_diagnostics(
    &self,
    config: &DiagnosticsConfig,
    file_id: FileId,
) -> Cancellable<Vec<Diagnostic>>

/// Full semantic diagnostics - type errors, unresolved references, etc.
pub fn semantic_diagnostics(
    &self,
    config: &DiagnosticsConfig,
    resolve: AssistResolveStrategy,
    file_id: FileId,
) -> Cancellable<Vec<Diagnostic>>

/// Both syntax and semantic diagnostics
pub fn full_diagnostics(
    &self,
    config: &DiagnosticsConfig,
    resolve: AssistResolveStrategy,
    file_id: FileId,
) -> Cancellable<Vec<Diagnostic>>
```

### Runnables

```rust
/// Find tests, benchmarks, and binaries in a file
pub fn runnables(&self, file_id: FileId) -> Cancellable<Vec<Runnable>>

/// Find tests that exercise the item at the given position
pub fn related_tests(
    &self,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Cancellable<Vec<Runnable>>
```

### Crate Information

```rust
/// Get crates that contain this file
pub fn crates_for(&self, file_id: FileId) -> Cancellable<Vec<Crate>>

/// Get the root file (lib.rs/main.rs) of a crate
pub fn crate_root(&self, crate_id: Crate) -> Cancellable<FileId>

/// Get the Rust edition of a crate
pub fn crate_edition(&self, crate_id: Crate) -> Cancellable<Edition>
```

---

## Re-exports

The crate re-exports many types for convenience:

```rust
// From ide_db
pub use ide_db::{
    FileId, FilePosition, FileRange, RootDatabase, SymbolKind,
    Crate, CrateGraphBuilder, FileChange, SourceRoot,
    LineCol, LineIndex, TextEdit,
};

// From hir
pub use hir::Semantics;

// From ide_assists
pub use ide_assists::{Assist, AssistConfig, AssistKind};

// From ide_completion
pub use ide_completion::{CompletionConfig, CompletionItem};

// From ide_diagnostics
pub use ide_diagnostics::{Diagnostic, DiagnosticsConfig};

// From syntax
pub use syntax::{TextRange, TextSize};
```

---

## Usage in tarjanize

tarjanize uses a combination of `ra_ap_hir` and `ra_ap_ide`, each for what it does best.

### What tarjanize uses from ra_ap_ide

**TryToNav** - Converting HIR types to source locations:

```rust
use ra_ap_ide::TryToNav;

fn extract_module_def(sema: &Semantics<'_, RootDatabase>, def: ModuleDef) -> Option<Symbol> {
    // TryToNav gives us NavigationTarget with file_id and ranges
    let nav = def.try_to_nav(sema)?.call_site;

    let file_id = nav.file_id;           // Which file contains this symbol
    let cost = nav.full_range.len();     // Byte size as compile-time proxy
    let focus = nav.focus_range;         // Where the name is (for UI)

    // ...
}
```

This is cleaner than using `HasSource` directly because:
- `TryToNav` handles macro expansions via `UpmappingResult` (call_site vs def_site)
- `NavigationTarget` provides both `full_range` (whole item) and `focus_range` (just the name)
- It's already implemented for all the HIR types we care about

### What tarjanize uses from ra_ap_hir

**Module hierarchy traversal** - The structure of the codebase:

```rust
use ra_ap_hir::{Crate, Module, ModuleDef, Impl};

for krate in Crate::all(db) {
    if krate.origin(db).is_local() {
        let root = krate.root_module(db);
        traverse_module(db, root);
    }
}

fn traverse_module(db: &RootDatabase, module: Module) {
    for decl in module.declarations(db) { /* ... */ }
    for impl_ in module.impl_defs(db) { /* ... */ }
    for child in module.children(db) { traverse_module(db, child); }
}
```

**Semantic analysis** - Understanding what code references:

```rust
use ra_ap_hir::Semantics;

// Semantics bridges syntax trees to semantic information
let sema = Semantics::new(db);
// ... resolve paths, find definitions, etc.
```

### Why NOT StaticIndex

`StaticIndex` seems appealing because it pre-computes all definitions and references. But it's designed for LSIF/SCIP output, which doesn't match tarjanize's needs:

- **Token-centric, not module-centric**: StaticIndex organizes by definition, but tarjanize needs the module hierarchy
- **No impl block grouping**: tarjanize collapses impl methods to their impl blocks (atomic units), but StaticIndex treats each method separately
- **Reference granularity**: StaticIndex tracks every mention, but tarjanize only needs dependency edges between symbols

### Summary

| Need | Crate | Type/Trait |
|------|-------|------------|
| Module/crate hierarchy | `ra_ap_hir` | `Crate`, `Module`, `ModuleDef` |
| Impl blocks | `ra_ap_hir` | `Impl`, `AssocItem` |
| Source locations | `ra_ap_ide` | `TryToNav`, `NavigationTarget` |
| Path resolution | `ra_ap_hir` | `Semantics` |
| Visibility | `ra_ap_hir` | `HasVisibility` |

See [ra_ap_hir.md](ra_ap_hir.md) for detailed documentation on the semantic model.
