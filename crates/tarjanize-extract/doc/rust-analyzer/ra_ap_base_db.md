# ra_ap_base_db

Defines the basic database traits and input types for rust-analyzer. This is the foundational layer that all other analysis is built upon - everything in rust-analyzer is derived from the inputs defined here.

## Overview

This crate provides:
1. **Salsa database infrastructure** - Re-exports salsa and defines query groups
2. **File and source root management** - `FileId`, `SourceRoot`, `SourceRootId`
3. **Crate graph** - `Crate`, `CrateGraphBuilder`, dependencies
4. **Edition-aware file IDs** - `EditionedFileId` for parsing with correct edition
5. **Change tracking** - `FileChange` for transactional updates

**Design principle**: This crate does NO actual IO. See `ra_ap_vfs` and `ra_ap_project_model` for how IO is done and lowered to these input types.

## Re-exports

```rust
pub use salsa;
pub use salsa_macros;
pub use semver::{BuildMetadata, Prerelease, Version, VersionReq};
pub use vfs::{AnchoredPath, AnchoredPathBuf, FileId, VfsPath, file_set::FileSet};
```

## Module Structure

| Module | Description |
|--------|-------------|
| `lib.rs` | Database traits, `Files` storage, query implementations |
| `input.rs` | Crate graph types, source roots, dependencies |
| `change.rs` | `FileChange` for transactional updates |
| `editioned_file_id.rs` | `EditionedFileId` - file ID with edition info |
| `target.rs` | Target architecture information |

---

## Database Traits

### SourceDatabase

The fundamental trait for accessing source files. All databases that work with source code must implement this.

```rust
#[salsa_macros::db]
pub trait SourceDatabase: salsa::Database {
    /// Get the text content of a file
    fn file_text(&self, file_id: vfs::FileId) -> FileText;

    /// Set file text
    fn set_file_text(&mut self, file_id: vfs::FileId, text: &str);

    /// Set file text with explicit durability
    fn set_file_text_with_durability(
        &mut self,
        file_id: vfs::FileId,
        text: &str,
        durability: Durability,
    );

    /// Get the source root containing a file
    fn source_root(&self, id: SourceRootId) -> SourceRootInput;

    /// Get which source root a file belongs to
    fn file_source_root(&self, id: vfs::FileId) -> FileSourceRootInput;

    /// Resolve a path relative to an anchor file
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId>;

    /// Get the crate map (internal)
    fn crates_map(&self) -> Arc<CratesMap>;

    /// Get nonce and revision for cache invalidation
    fn nonce_and_revision(&self) -> (Nonce, salsa::Revision);
}
```

### RootQueryDb

Extends `SourceDatabase` with parsing and crate queries.

```rust
#[query_group::query_group]
pub trait RootQueryDb: SourceDatabase + salsa::Database {
    /// Parse a file into a syntax tree (LRU cached, capacity 128)
    #[salsa::lru(128)]
    fn parse(&self, file_id: EditionedFileId) -> Parse<ast::SourceFile>;

    /// Get parse errors including validation errors
    fn parse_errors(&self, file_id: EditionedFileId) -> Option<&[SyntaxError]>;

    /// Get the toolchain release channel for a crate
    fn toolchain_channel(&self, krate: Crate) -> Option<ReleaseChannel>;

    /// Get all crates whose root file is in a source root
    fn source_root_crates(&self, id: SourceRootId) -> Arc<[Crate]>;

    /// Get crates relevant to a file (crates whose root is in same source root)
    fn relevant_crates(&self, file_id: FileId) -> Arc<[Crate]>;

    /// Get all crates in topological order
    /// WARNING: Do not use in hir-* crates - kills incrementality!
    #[salsa::input]
    fn all_crates(&self) -> Arc<Box<[Crate]>>;
}
```

---

## Source Roots

### SourceRootId

A simple wrapper around `u32` identifying a source root.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceRootId(pub u32);
```

### SourceRoot

A directory watched for changes. Files are grouped into source roots, and paths are always relative to the containing source root.

```rust
pub struct SourceRoot {
    /// True for sysroot/crates.io libraries (considered immutable)
    pub is_library: bool,
    file_set: FileSet,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new_local` | `fn new_local(file_set: FileSet) -> SourceRoot` | Create a local (mutable) source root |
| `new_library` | `fn new_library(file_set: FileSet) -> SourceRoot` | Create a library (immutable) source root |
| `path_for_file` | `fn path_for_file(&self, file: &FileId) -> Option<&VfsPath>` | Get path for a file |
| `file_for_path` | `fn file_for_path(&self, path: &VfsPath) -> Option<&FileId>` | Get file for a path |
| `resolve_path` | `fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId>` | Resolve relative path |
| `iter` | `fn iter(&self) -> impl Iterator<Item = FileId>` | Iterate over all files |

**Nesting**: Source roots can be nested. A file belongs to the nearest enclosing source root.

**Isolation**: Files in one source root cannot refer to files in another source root by path.

### Singleton Inputs

```rust
/// Set of source roots for crates.io libraries (assumed immutable)
#[salsa::input(singleton)]
pub struct LibraryRoots {
    pub roots: FxHashSet<SourceRootId>,
}

/// Set of local workspace roots (assumed to change frequently)
#[salsa::input(singleton)]
pub struct LocalRoots {
    pub roots: FxHashSet<SourceRootId>,
}
```

---

## Files Storage

### Files

A concurrent storage structure for file data, using `DashMap` for thread-safe access.

```rust
#[derive(Debug, Default)]
pub struct Files {
    files: Arc<DashMap<vfs::FileId, FileText, ...>>,
    source_roots: Arc<DashMap<SourceRootId, SourceRootInput, ...>>,
    file_source_roots: Arc<DashMap<vfs::FileId, FileSourceRootInput, ...>>,
}
```

**Methods:**

| Method | Description |
|--------|-------------|
| `file_text(file_id)` | Get file text (panics if not found) |
| `set_file_text(db, file_id, text)` | Set file text |
| `set_file_text_with_durability(...)` | Set file text with explicit durability |
| `source_root(source_root_id)` | Get source root input |
| `set_source_root_with_durability(...)` | Set source root |
| `file_source_root(id)` | Get file's source root input |
| `set_file_source_root_with_durability(...)` | Set file's source root |

### FileText (Salsa Input)

```rust
#[salsa_macros::input]
pub struct FileText {
    pub text: Arc<str>,      // The file content
    pub file_id: vfs::FileId, // For debugging
}
```

---

## Crate Graph

### Crate (Salsa Input)

The main crate type, a salsa input that stores all crate metadata.

```rust
#[salsa_macros::input]
pub struct Crate {
    /// Core crate data (root file, edition, dependencies, etc.)
    pub data: BuiltCrateData,

    /// Extra data not needed for analysis (display name, version)
    pub extra_data: ExtraCrateData,

    /// Workspace-level data (shared across workspace crates)
    pub workspace_data: Arc<CrateWorkspaceData>,

    /// Cfg options for conditional compilation
    pub cfg_options: CfgOptions,

    /// Environment variables
    pub env: Env,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `root_file_id` | `fn root_file_id(self, db) -> EditionedFileId` | Get the crate's root file |
| `transitive_deps` | `fn transitive_deps(self, db) -> Vec<Crate>` | All transitive dependencies including self |
| `transitive_rev_deps` | `fn transitive_rev_deps(self, db) -> Box<[Crate]>` | All reverse dependencies |

**Warning**: `transitive_deps` and `transitive_rev_deps` kill incrementality - don't use in `hir-*` crates!

### CrateData / BuiltCrateData

Core crate metadata. `CrateData<Id>` is generic over the dependency ID type.

```rust
pub struct CrateData<Id> {
    pub root_file_id: FileId,
    pub edition: Edition,
    pub dependencies: Vec<Dependency<Id>>,
    pub origin: CrateOrigin,
    pub crate_attrs: Box<[Box<str>]>,  // Extra crate-level attributes
    pub is_proc_macro: bool,
    pub proc_macro_cwd: Arc<AbsPathBuf>,
}

pub type CrateDataBuilder = CrateData<CrateBuilderId>;
pub type BuiltCrateData = CrateData<Crate>;
```

### CrateOrigin

Distinguishes where a crate comes from.

```rust
pub enum CrateOrigin {
    /// From the rustc workspace
    Rustc { name: Symbol },

    /// Workspace member (local development)
    Local { repo: Option<String>, name: Option<Symbol> },

    /// Non-member library (dependencies)
    Library { repo: Option<String>, name: Symbol },

    /// Language-provided (std, core, etc.)
    Lang(LangCrateOrigin),
}
```

**Methods:**
- `is_local()` - Returns `true` for `Local` variant
- `is_lib()` - Returns `true` for `Library` variant
- `is_lang()` - Returns `true` for `Lang` variant

### LangCrateOrigin

```rust
pub enum LangCrateOrigin {
    Alloc,
    Core,
    ProcMacro,
    Std,
    Test,
    Dependency,
    Other,
}
```

### CrateName

A crate name that cannot contain dashes.

```rust
pub struct CrateName(Symbol);
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(name: &str) -> Result<CrateName, &str>` | Create, rejecting dashes |
| `normalize_dashes` | `fn normalize_dashes(name: &str) -> CrateName` | Replace dashes with underscores |
| `symbol` | `fn symbol(&self) -> &Symbol` | Get the underlying symbol |

### CrateDisplayName

Display name with both normalized (`_`) and canonical (`-`) forms.

```rust
pub struct CrateDisplayName {
    crate_name: CrateName,       // With underscores
    canonical_name: Symbol,       // As in Cargo.toml (may have dashes)
}
```

**Methods:**
- `canonical_name()` - Get the Cargo.toml name
- `crate_name()` - Get the normalized name
- `from_canonical_name(name)` - Create from Cargo.toml name

### Dependency

```rust
pub struct Dependency<Id> {
    pub crate_id: Id,
    pub name: CrateName,
    prelude: bool,   // Add to extern prelude?
    sysroot: bool,   // Is this a sysroot-injected dep?
}

pub type DependencyBuilder = Dependency<CrateBuilderId>;
pub type BuiltDependency = Dependency<Crate>;
```

**Methods (BuiltDependency):**
- `is_prelude()` - Whether to add to extern prelude
- `is_sysroot()` - Whether this is sysroot-injected

### CrateGraphBuilder

Builder for constructing the crate graph before committing to the database.

```rust
#[derive(Default, Clone)]
pub struct CrateGraphBuilder {
    arena: Arena<CrateBuilder>,
}
```

**Methods:**

| Method | Description |
|--------|-------------|
| `add_crate_root(...)` | Add a new crate, returns `CrateBuilderId` |
| `add_dep(from, dep)` | Add dependency, returns error if cyclic |
| `set_in_db(db)` | Commit to database, returns `CratesIdMap` |
| `iter()` | Iterate over crate IDs |
| `transitive_deps(of)` | Get transitive dependencies |
| `extend(other, proc_macros)` | Merge another graph, deduplicating |
| `remove_crates_except(to_keep)` | Filter crates |
| `shrink_to_fit()` | Shrink internal storage |

### CyclicDependenciesError

Returned when adding a dependency would create a cycle.

```rust
pub struct CyclicDependenciesError {
    path: Vec<(CrateBuilderId, Option<CrateDisplayName>)>,
}
```

---

## EditionedFileId

An interned file ID that includes the Rust edition and owning crate. This is used for parsing with the correct edition.

```rust
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EditionedFileId(...);
```

**Construction:**

| Method | Description |
|--------|-------------|
| `new(db, file_id, edition, krate)` | Create with known crate |
| `from_span(db, editioned_file_id, krate)` | Create from span |
| `from_span_guess_origin(db, editioned_file_id)` | Create, guessing the crate |
| `current_edition_guess_origin(db, file_id)` | Use current edition, guess crate |

**Note on `guess_origin`**: Only use when you cannot determine the crate precisely:
1. File is not in the module tree
2. Latency-sensitive context (folding, on-enter)

**Accessors:**

| Method | Description |
|--------|-------------|
| `file_id(db)` | Get the underlying `FileId` |
| `edition(db)` | Get the Rust edition |
| `krate(db)` | Get the owning crate |
| `editioned_file_id(db)` | Get the span-level ID |
| `unpack(db)` | Get `(FileId, Edition)` tuple |

**Hashing Design**: The `EditionedFileId` hashes only the file+edition, not the crate. This allows reusing existing interned IDs when the crate isn't known precisely.

---

## FileChange

Encapsulates a batch of changes to apply to the database transactionally.

```rust
pub struct FileChange {
    pub roots: Option<Vec<SourceRoot>>,
    pub files_changed: Vec<(FileId, Option<String>)>,
    pub crate_graph: Option<CrateGraphBuilder>,
}
```

**Methods:**

| Method | Description |
|--------|-------------|
| `set_roots(roots)` | Set new source roots |
| `change_file(file_id, text)` | Queue a file change (None = reset to empty) |
| `set_crate_graph(graph)` | Set new crate graph |
| `apply(db)` | Apply all changes, returns `CratesIdMap` if graph changed |

**Durability**: Library files get `Durability::HIGH`, local files get `Durability::LOW`.

---

## Environment

### Env

Environment variables for a crate.

```rust
#[derive(Default, Clone, PartialEq, Eq)]
pub struct Env {
    entries: FxHashMap<String, String>,
}
```

**Methods:**
- `set(env, value)` - Set a variable
- `get(env)` - Get a variable
- `extend_from_other(other)` - Merge from another Env
- `is_empty()` - Check if empty
- `insert(k, v)` - Insert a variable
- `contains_key(arg)` - Check if key exists

Implements `Extend`, `FromIterator`, `IntoIterator`, `From<Env> for Vec<(String, String)>`.

---

## Target Information

### Arch

```rust
pub enum Arch {
    Wasm32,
    Wasm64,
    Other,
}
```

### TargetData

```rust
pub struct TargetData {
    pub data_layout: Box<str>,
    pub arch: Arch,
}
```

### TargetLoadError

Error type for target loading failures.

```rust
pub struct TargetLoadError(Arc<str>);

pub type TargetLoadResult = Result<TargetData, TargetLoadError>;
```

---

## Workspace Data

### CrateWorkspaceData

Data shared by all crates in a workspace.

```rust
pub struct CrateWorkspaceData {
    pub target: Result<target::TargetData, target::TargetLoadError>,
    pub toolchain: Option<Version>,
}
```

**Methods:**
- `is_atleast_187()` - Check if toolchain is >= 1.87

### ReleaseChannel

```rust
pub enum ReleaseChannel {
    Stable,
    Beta,
    Nightly,
}
```

**Methods:**
- `as_str()` - Get string representation
- `from_str(str)` - Parse from string

---

## Constants

```rust
pub const DEFAULT_FILE_TEXT_LRU_CAP: u16 = 16;
pub const DEFAULT_PARSE_LRU_CAP: u16 = 128;
pub const DEFAULT_BORROWCK_LRU_CAP: u16 = 2024;
```

---

## Utility Types

### Nonce

A unique identifier for cache invalidation.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Nonce(usize);
```

Uses an atomic counter to generate unique values.

### FxIndexSet / FxIndexMap

Type aliases for `indexmap` collections with FxHasher:

```rust
pub type FxIndexSet<T> = indexmap::IndexSet<T, rustc_hash::FxBuildHasher>;
pub type FxIndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<FxHasher>>;
```

---

## Macros

### impl_intern_key

Generates interned key types for salsa.

```rust
#[macro_export]
macro_rules! impl_intern_key {
    ($id:ident, $loc:ident) => { ... }
}
```

---

## Usage in tarjanize

tarjanize-extract uses ra_ap_base_db for:

1. **Crate identification**: `Crate` and `CrateOrigin::is_local()` to filter workspace members
2. **File access**: `FileId` and `SourceDatabase::file_text()` for source code
3. **Edition handling**: `EditionedFileId` ensures correct parsing
4. **Dependency traversal**: `Crate::data().dependencies` for the dependency graph

The crate graph from this module is the foundation for understanding which crates to analyze and their relationships.
