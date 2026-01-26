# ra_ap_project_model

Handles "real world" project models for rust-analyzer, bridging the gap between build systems (Cargo, custom JSON) and rust-analyzer's abstract semantic model (`CrateGraph`).

## Overview

This crate is concerned with:
1. **Project discovery** - Finding the relevant Cargo.toml for a directory
2. **Custom build steps** - build.rs code generation and proc-macro compilation
3. **Lowering** - Converting concrete project models to `base_db::CrateGraph`

rust-analyzer maintains a strict separation between:
- **Pure abstract model**: `base_db::CrateGraph`
- **Concrete build-system models**: `CargoWorkspace`, `ProjectJson`

## Re-exports

```rust
pub use cargo_metadata::Metadata;
pub use crate::{
    build_dependencies::{ProcMacroDylibPath, WorkspaceBuildScripts},
    cargo_workspace::{
        CargoConfig, CargoFeatures, CargoMetadataConfig, CargoWorkspace, Package, PackageData,
        PackageDependency, RustLibSource, Target, TargetData, TargetDirectoryConfig, TargetKind,
    },
    manifest_path::ManifestPath,
    project_json::{ProjectJson, ProjectJsonData},
    sysroot::Sysroot,
    workspace::{FileLoader, PackageRoot, ProjectWorkspace, ProjectWorkspaceKind},
};
```

---

## ProjectManifest

Represents a discovered project manifest file.

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ProjectManifest {
    ProjectJson(ManifestPath),  // rust-project.json
    CargoToml(ManifestPath),    // Cargo.toml
    CargoScript(ManifestPath),  // *.rs cargo scripts
}
```

### Construction & Discovery

| Method | Description |
|--------|-------------|
| `from_manifest_file(path)` | Create from explicit path |
| `discover_single(path)` | Find exactly one project (error if none or multiple) |
| `discover(path)` | Find all projects in parent directories |
| `discover_all(paths)` | Find projects in multiple paths, deduplicated |

### Discovery Logic

1. Looks for `rust-project.json` or `.rust-project.json` in parent directories
2. If not found, looks for `Cargo.toml` in parent directories
3. If nothing in parents, checks immediate child directories for `Cargo.toml`

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `manifest_path` | `fn(&self) -> &ManifestPath` | Get the manifest path |

---

## ManifestPath

A newtype wrapper for manifest file paths (Cargo.toml, rust-project.json, or .rs scripts).

```rust
// From manifest_path.rs
pub struct ManifestPath(AbsPathBuf);
```

Ensures the path is absolute and has a valid manifest filename.

---

## CargoWorkspace

Represents the logical structure of a Cargo workspace. Closely mirrors `cargo metadata` output.

```rust
pub struct CargoWorkspace {
    packages: Arena<PackageData>,
    targets: Arena<TargetData>,
    workspace_root: AbsPathBuf,
    target_directory: AbsPathBuf,
    manifest_path: ManifestPath,
    is_virtual_workspace: bool,
    is_sysroot: bool,
    env: Env,
    requires_rustc_private: bool,
}
```

### Construction

```rust
impl CargoWorkspace {
    pub fn new(
        meta: cargo_metadata::Metadata,
        ws_manifest_path: ManifestPath,
        cargo_env: Env,
        is_sysroot: bool,
    ) -> CargoWorkspace;
}
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `packages` | `fn(&self) -> impl Iterator<Item = Package>` | Iterate all packages |
| `target_by_root` | `fn(&self, root: &AbsPath) -> Option<Target>` | Find target by source root |
| `workspace_root` | `fn(&self) -> &AbsPath` | Get workspace root directory |
| `manifest_path` | `fn(&self) -> &ManifestPath` | Get workspace manifest |
| `target_directory` | `fn(&self) -> &AbsPath` | Get target directory |
| `package_flag` | `fn(&self, package: &PackageData) -> String` | Get cargo package flag |
| `parent_manifests` | `fn(&self, path: &ManifestPath) -> Option<Vec<ManifestPath>>` | Find parent manifests |
| `workspace_features` | `fn(&self) -> FxHashSet<String>` | Get all workspace features |
| `is_virtual_workspace` | `fn(&self) -> bool` | True if workspace has no root package |
| `env` | `fn(&self) -> &Env` | Get environment variables |
| `is_sysroot` | `fn(&self) -> bool` | True if this is the sysroot workspace |
| `requires_rustc_private` | `fn(&self) -> bool` | True if any package needs rustc_private |

### Indexing

```rust
impl ops::Index<Package> for CargoWorkspace {
    type Output = PackageData;
}

impl ops::Index<Target> for CargoWorkspace {
    type Output = TargetData;
}
```

---

## Package and PackageData

A handle to a package and its data.

```rust
pub type Package = Idx<PackageData>;

pub struct PackageData {
    pub version: semver::Version,
    pub name: String,
    pub repository: Option<String>,
    pub manifest: ManifestPath,
    pub targets: Vec<Target>,
    pub is_local: bool,           // From local filesystem (editable)
    pub is_member: bool,          // Member of the workspace
    pub dependencies: Vec<PackageDependency>,
    pub edition: Edition,
    pub features: FxHashMap<String, Vec<String>>,
    pub active_features: Vec<String>,
    pub id: Arc<PackageId>,
    pub authors: Vec<String>,
    pub description: Option<String>,
    pub homepage: Option<String>,
    pub license: Option<String>,
    pub license_file: Option<Utf8PathBuf>,
    pub readme: Option<Utf8PathBuf>,
    pub rust_version: Option<semver::Version>,
    pub metadata: RustAnalyzerPackageMetaData,
    pub all_member_deps: Option<FxHashSet<Package>>,  // Transitive member deps
}
```

**Note**: `is_local` is true for packages without a source (workspace members and path dependencies). `is_member` is true only for explicit workspace members.

---

## Target and TargetData

A handle to a build target and its data.

```rust
pub type Target = Idx<TargetData>;

pub struct TargetData {
    pub package: Package,
    pub name: String,
    pub root: AbsPathBuf,      // Main source file
    pub kind: TargetKind,
    pub required_features: Vec<String>,
}
```

---

## TargetKind

```rust
pub enum TargetKind {
    Bin,
    Lib { is_proc_macro: bool },
    Example,
    Test,
    Bench,
    BuildScript,  // Cargo calls this "custom-build"
    Other,
}
```

### Methods

| Method | Description |
|--------|-------------|
| `new(kinds)` | Create from cargo_metadata kinds |
| `is_executable()` | True for Bin and Example |
| `is_proc_macro()` | True for proc-macro libs |
| `as_cargo_target()` | Get cargo CLI target name |

---

## PackageDependency

```rust
pub struct PackageDependency {
    pub pkg: Package,
    pub name: String,
    pub kind: DepKind,
}
```

---

## DepKind

```rust
pub enum DepKind {
    Normal,  // Library, binary, dev targets (not build script)
    Dev,     // Test/bench targets only
    Build,   // Build script only
}
```

---

## CargoConfig

Configuration for cargo operations.

```rust
pub struct CargoConfig {
    pub all_targets: bool,
    pub features: CargoFeatures,
    pub target: Option<String>,                 // rustc target triple
    pub sysroot: Option<RustLibSource>,
    pub sysroot_src: Option<AbsPathBuf>,
    pub rustc_source: Option<RustLibSource>,    // rustc private crates
    pub extra_includes: Vec<AbsPathBuf>,
    pub cfg_overrides: CfgOverrides,
    pub wrap_rustc_in_build_scripts: bool,
    pub run_build_script_command: Option<Vec<String>>,
    pub extra_args: Vec<String>,
    pub extra_env: FxHashMap<String, Option<String>>,
    pub invocation_strategy: InvocationStrategy,
    pub target_dir_config: TargetDirectoryConfig,
    pub set_test: bool,
    pub no_deps: bool,
}
```

---

## CargoFeatures

```rust
pub enum CargoFeatures {
    All,
    Selected {
        features: Vec<String>,
        no_default_features: bool,
    },
}
```

---

## RustLibSource

How to find rustc source (for sysroot or rustc_private).

```rust
pub enum RustLibSource {
    Path(AbsPathBuf),  // Explicit path
    Discover,          // Auto-detect
}
```

---

## CfgOverrides

Per-crate cfg overrides.

```rust
pub struct CfgOverrides {
    pub global: cfg::CfgDiff,                          // Applied to all crates
    pub selective: FxHashMap<String, cfg::CfgDiff>,    // Per-crate overrides
}
```

### Methods

| Method | Description |
|--------|-------------|
| `len()` | Total number of overrides |
| `apply(cfg_options, name)` | Apply overrides to cfg options |

---

## InvocationStrategy

When to invoke build scripts.

```rust
pub enum InvocationStrategy {
    Once,           // Invoke once for whole workspace
    PerWorkspace,   // Invoke per workspace (default)
}
```

---

## Sysroot

Represents the Rust sysroot with standard library sources.

```rust
// From sysroot.rs
pub struct Sysroot { ... }
```

Used to:
- Locate standard library crates (std, core, alloc, etc.)
- Find rustc toolchain binaries
- Provide proc-macro server

---

## ProjectJson

Alternative to Cargo for custom build systems (Buck, Bazel, etc.).

```rust
// From project_json.rs
pub struct ProjectJson { ... }
pub struct ProjectJsonData { ... }
```

Uses `rust-project.json` format for defining crates and dependencies.

---

## ProjectWorkspace

The unified workspace type combining cargo or JSON projects.

```rust
// From workspace.rs
pub struct ProjectWorkspace { ... }

pub enum ProjectWorkspaceKind {
    Cargo { ... },
    Json { ... },
}
```

This is what gets converted to `CrateGraph`.

---

## Toolchain Info

Submodule for querying toolchain information:

```rust
pub mod toolchain_info {
    pub mod rustc_cfg;      // Query rustc cfg options
    pub mod target_data;    // Target data layout
    pub mod target_tuple;   // Target triple parsing
    pub mod version;        // Toolchain version

    pub enum QueryConfig<'a> {
        Rustc(&'a Sysroot, &'a Path),
        Cargo(&'a Sysroot, &'a ManifestPath, &'a Option<CargoConfigFile>),
    }
}
```

---

## Module Structure

| Module | Description |
|--------|-------------|
| `cargo_workspace.rs` | CargoWorkspace and related types |
| `project_json.rs` | rust-project.json support |
| `workspace.rs` | Unified ProjectWorkspace |
| `sysroot.rs` | Sysroot discovery and loading |
| `manifest_path.rs` | ManifestPath newtype |
| `build_dependencies.rs` | Build script and proc-macro handling |
| `cargo_config_file.rs` | .cargo/config parsing |
| `env.rs` | Environment variable handling |
| `toolchain_info/` | Toolchain querying utilities |

---

## Usage in tarjanize

tarjanize-extract uses ra_ap_project_model for:

1. **CargoWorkspace** - To understand workspace structure
2. **Package/Target** - To identify crates and their source files
3. **PackageData::is_local** / **is_member** - To filter to workspace members
4. **TargetKind** - To identify library vs binary targets

The workspace loading happens through `ra_ap_load_cargo` which uses these types internally.
