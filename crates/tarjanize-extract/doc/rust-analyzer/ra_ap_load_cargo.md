# ra_ap_load_cargo

Loads a Cargo project into a static instance of analysis, without support for incorporating changes. This is the main entry point for using rust-analyzer as a library.

## Overview

This crate provides the high-level functions to:
1. **Discover** a project workspace from a path
2. **Load** the workspace into a `RootDatabase`
3. **Configure** proc-macro expansion
4. **Set up** the VFS (Virtual File System) for source file access

**Note**: This API is consumed by external tools to run rust-analyzer as a library. Don't remove any public API from this crate.

---

## Primary Functions

### load_workspace_at

Load a workspace by discovering it from a path.

```rust
pub fn load_workspace_at(
    root: &Path,
    cargo_config: &CargoConfig,
    load_config: &LoadCargoConfig,
    progress: &(dyn Fn(String) + Sync),
) -> anyhow::Result<(RootDatabase, vfs::Vfs, Option<ProcMacroClient>)>
```

**Steps:**
1. Discovers `ProjectManifest` from the given path
2. Loads `ProjectWorkspace`
3. Optionally runs build scripts (`load_out_dirs_from_check`)
4. Calls `load_workspace`

### load_workspace

Load a pre-constructed workspace.

```rust
pub fn load_workspace(
    ws: ProjectWorkspace,
    extra_env: &FxHashMap<String, Option<String>>,
    load_config: &LoadCargoConfig,
) -> anyhow::Result<(RootDatabase, vfs::Vfs, Option<ProcMacroClient>)>
```

Creates a new `RootDatabase` and loads the workspace into it.

### load_workspace_into_db

Load workspace into an existing database.

```rust
pub fn load_workspace_into_db(
    ws: ProjectWorkspace,
    extra_env: &FxHashMap<String, Option<String>>,
    load_config: &LoadCargoConfig,
    db: &mut RootDatabase,
) -> anyhow::Result<(vfs::Vfs, Option<ProcMacroClient>)>
```

Useful for extending foreign databases now that salsa supports it.

**Process:**
1. Spawns VFS loader (`vfs_notify::NotifyHandle`)
2. Optionally starts proc-macro server
3. Converts workspace to `CrateGraph` via `ws.to_crate_graph()`
4. Loads proc macros if server is available
5. Sets up source roots and file sets
6. Waits for VFS to load all files
7. Applies changes to database
8. Optionally primes caches

---

## Configuration Types

### LoadCargoConfig

Configuration for the loading process.

```rust
#[derive(Debug)]
pub struct LoadCargoConfig {
    /// Run `cargo check` to get OUT_DIR and proc-macro paths
    pub load_out_dirs_from_check: bool,

    /// How to start the proc-macro server
    pub with_proc_macro_server: ProcMacroServerChoice,

    /// Pre-fill analysis caches after loading
    pub prefill_caches: bool,
}
```

### ProcMacroServerChoice

How to obtain the proc-macro server.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcMacroServerChoice {
    /// Use proc-macro-srv from the workspace's sysroot
    Sysroot,

    /// Use an explicit path to proc-macro-srv
    Explicit(AbsPathBuf),

    /// Don't use proc macros
    None,
}
```

---

## ProjectFolders

Manages the file system layout for VFS loading.

```rust
#[derive(Default)]
pub struct ProjectFolders {
    /// Entries to load
    pub load: Vec<vfs::loader::Entry>,

    /// Indices of entries to watch for changes
    pub watch: Vec<usize>,

    /// Source root configuration
    pub source_root_config: SourceRootConfig,
}
```

### Construction

```rust
impl ProjectFolders {
    pub fn new(
        workspaces: &[ProjectWorkspace],
        global_excludes: &[AbsPathBuf],
        user_config_dir_path: Option<&AbsPath>,
    ) -> ProjectFolders;
}
```

**Processing:**
1. Collects all package roots from workspaces
2. Deduplicates overlapping source roots (handles rustc workspace edge cases)
3. Separates local vs library filesets
4. Adds buildfile entries for watching
5. Optionally adds user config directory (`rust-analyzer.toml`)

---

## SourceRootConfig

Configuration for partitioning files into source roots.

```rust
#[derive(Default, Debug)]
pub struct SourceRootConfig {
    /// File set configuration
    pub fsc: FileSetConfig,

    /// Indices of local (non-library) filesets
    pub local_filesets: Vec<u64>,
}
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `partition` | `fn(&self, vfs: &Vfs) -> Vec<SourceRoot>` | Create source roots from VFS |
| `source_root_parent_map` | `fn(&self) -> FxHashMap<SourceRootId, SourceRootId>` | Map children to parents |

**Source Root Parent Map:**
- Maps local source roots to their parent source roots
- Uses disjoint-set (union-find) to prevent cycles
- Roots without parents are considered top-level

---

## Proc Macro Loading

### load_proc_macro

Load proc macros from a dylib.

```rust
pub fn load_proc_macro(
    server: &ProcMacroClient,
    path: &AbsPath,
    ignored_macros: &[Box<str>],
) -> ProcMacroLoadResult
```

**Parameters:**
- `server` - The proc-macro client/server connection
- `path` - Path to the proc-macro dylib
- `ignored_macros` - Names of macros to disable

**Returns:** `Result<Vec<ProcMacro>, ProcMacroLoadingError>`

---

## Internal Process

### Crate Graph Loading

The internal `load_crate_graph_into_db` function:

1. Creates `ChangeWithProcMacros` for batch updates
2. Enables proc-macro attribute expansion
3. Waits for VFS to finish loading all files
4. Processes VFS changes into file contents
5. Partitions files into source roots
6. Sets the crate graph and proc macros
7. Applies all changes atomically to the database

### Proc Macro Expansion

The `Expander` struct implements `ProcMacroExpander`:

```rust
impl ProcMacroExpander for Expander {
    fn expand(
        &self,
        db: &dyn ExpandDatabase,
        subtree: &tt::TopSubtree,
        attrs: Option<&tt::TopSubtree>,
        env: &Env,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
        current_dir: String,
    ) -> Result<tt::TopSubtree, ProcMacroExpansionError>;
}
```

Handles sub-requests from proc macros:
- `LocalFilePath` - Get local path for a file ID
- `SourceText` - Get source text for a span
- `LineColumn` - Get line/column for an offset
- `FilePath` - Get full path for a file ID
- `ByteRange` - Get byte range for a span

---

## Return Values

All main loading functions return a tuple:

```rust
(RootDatabase, vfs::Vfs, Option<ProcMacroClient>)
```

| Component | Description |
|-----------|-------------|
| `RootDatabase` | The analysis database with all crates loaded |
| `vfs::Vfs` | Virtual file system with source contents |
| `Option<ProcMacroClient>` | Proc-macro server if enabled and successful |

---

## Error Handling

Loading can fail for various reasons:
- No project manifest found
- Cargo metadata failure
- Build script errors (logged but don't fail loading)
- Proc-macro server startup failure (logged, continues without proc macros)

Build script errors are logged but don't prevent loading - rust-analyzer continues with reduced functionality.

---

## Usage in tarjanize

tarjanize-extract uses ra_ap_load_cargo for:

1. **`load_workspace_at`** - Main entry point to load a Cargo workspace
2. **`LoadCargoConfig`** - Configure loading behavior:
   - `load_out_dirs_from_check: true` - Get OUT_DIR for build.rs analysis
   - `with_proc_macro_server: Sysroot` - Enable proc-macro expansion
   - `prefill_caches: false` - Don't pre-fill (we do targeted queries)

The returned `RootDatabase` provides full semantic analysis through the `hir` API.

---

## Example

```rust
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use project_model::CargoConfig;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let cargo_config = CargoConfig::default();
    let load_config = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: false,
    };

    let (db, vfs, _proc_macro_client) = load_workspace_at(
        Path::new("."),
        &cargo_config,
        &load_config,
        &|msg| eprintln!("{}", msg),
    )?;

    // Now use `db` for semantic analysis via hir
    let all_crates = db.all_crates();
    println!("Loaded {} crates", all_crates.len());

    Ok(())
}
```
