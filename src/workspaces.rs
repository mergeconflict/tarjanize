//! Workspace loading for tarjanize.
//!
//! This module initializes rust-analyzer's analysis database for a Cargo
//! workspace. The configuration is carefully tuned for tarjanize's needs:
//!
//! - **Proc macro expansion**: Enabled to capture dependencies from derive macros
//! - **Build script execution**: Enabled to analyze generated code from build.rs
//! - **Lazy caching**: Disabled prefilling to avoid analyzing unused dependencies
//!
//! The returned database and VFS provide the foundation for all subsequent
//! analysis. The VFS (Virtual File System) is needed to map internal FileIds
//! back to filesystem paths.

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use ra_ap_ide_db::RootDatabase;
use ra_ap_load_cargo::{
    LoadCargoConfig, ProcMacroServerChoice, load_workspace_at,
};
use ra_ap_project_model::CargoConfig;
use ra_ap_vfs::Vfs;
use tracing::debug;

/// Load a Rust workspace into rust-analyzer's analysis database.
///
/// Returns the database and VFS. The VFS is needed to convert FileIds to paths.
///
/// # Errors
///
/// Returns [`anyhow::Error`] if:
/// - The path doesn't exist or can't be canonicalized
/// - The path contains non-UTF-8 characters
/// - The workspace can't be loaded (invalid Cargo.toml, missing dependencies, etc.)
///
/// # Example
///
/// ```ignore
/// let (db, vfs) = load_workspace("path/to/workspace")?;
/// let graph = extract_symbol_graph(&db, &vfs, "my_workspace");
/// ```
pub fn load_workspace(path: &str) -> Result<(RootDatabase, Vfs)> {
    // rust-analyzer requires absolute paths for file identification and
    // workspace discovery. canonicalize() also resolves symlinks to ensure
    // we work with the real filesystem location.
    let workspace_path = std::fs::canonicalize(path)
        .context("Failed to canonicalize workspace path")?;

    // rust-analyzer's APIs expect UTF-8 paths internally. Utf8PathBuf makes
    // this contract explicit and provides better error messages if someone
    // uses a non-UTF-8 path (rare).
    let workspace_path = Utf8PathBuf::from_path_buf(workspace_path)
        .map_err(|_| anyhow::anyhow!("Path contains invalid UTF-8"))?;

    // CargoConfig controls how Cargo projects are interpreted (e.g., which
    // features to enable, target platform). Default settings work for most
    // cases.
    let cargo_config = CargoConfig::default();

    let load_config = LoadCargoConfig {
        // Run `cargo check` to execute build scripts and capture their
        // outputs. Critical for tarjanize because build.rs files generate
        // code at compile time (e.g., bindings, lookup tables), and we need
        // to see dependencies in that generated code. Without this, we'd
        // miss edges in our dependency graph.
        load_out_dirs_from_check: true,

        // Use the system's rustc to expand procedural macros. Critical for
        // tarjanize because proc macros like #[derive(Serialize)] generate
        // impl blocks that create dependencies (e.g., impl Serialize depends
        // on serde::Serialize). Without expansion, we'd miss these edges and
        // our SCC computation would be incomplete.
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,

        // Don't precompute queries for all crates upfront. With 169 crates in
        // the graph but only a few workspace members, prefilling would waste
        // time analyzing external dependencies we'll never inspect. Lazy
        // evaluation ensures we only pay for crates we actually query.
        prefill_caches: false,
    };

    // Load the workspace into rust-analyzer's analysis database. This:
    // - Parses all Cargo.toml files to discover crates and dependencies
    // - Parses Rust source files into syntax trees
    // - Builds the semantic model (name resolution, type inference, trait
    //   solving)
    let (db, vfs, _proc_macro_server) = load_workspace_at(
        workspace_path.as_std_path(),
        &cargo_config,
        &load_config,
        &|msg| {
            debug!(message = %msg, "workspace.progress");
        },
    )?;

    Ok((db, vfs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_path() {
        let result = load_workspace("/nonexistent/path/that/does/not/exist");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("canonicalize"),
            "Error should mention canonicalize: {err}"
        );
    }

    /// Integration test for the production workspace loading codepath.
    ///
    /// This exercises load_workspace() and extract_symbol_graph() with a real
    /// Cargo project to ensure the full pipeline works end-to-end.
    #[test]
    fn test_load_workspace_and_extract() {
        use crate::extract::extract_symbol_graph;

        let (db, vfs) =
            load_workspace("tests/fixtures/minimal_crate").expect("load workspace");
        let graph = extract_symbol_graph(&db, &vfs, "minimal_crate");

        // Basic sanity checks
        assert_eq!(graph.workspace_name, "minimal_crate");
        assert_eq!(graph.crates.len(), 1);

        let root = &graph.crates[0];
        assert_eq!(root.name, "minimal_crate");

        // Should have Foo struct and bar function
        let symbol_names: Vec<_> =
            root.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(symbol_names.contains(&"Foo"), "Should have Foo struct");
        assert!(symbol_names.contains(&"bar"), "Should have bar function");

        // bar() -> Foo creates a dependency edge
        let has_edge = graph
            .edges
            .iter()
            .any(|e| e.from.contains("bar") && e.to.contains("Foo"));
        assert!(has_edge, "bar should depend on Foo");
    }
}
