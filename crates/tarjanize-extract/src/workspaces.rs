//! Workspace loading for tarjanize.
//!
//! This module initializes rust-analyzer's analysis database for a Cargo
//! workspace. The configuration is carefully tuned for tarjanize's needs:
//!
//! - **Proc macro expansion**: Enabled to capture dependencies from derive macros
//! - **Build script execution**: Enabled to analyze generated code from build.rs
//! - **Lazy caching**: Disabled prefilling to avoid analyzing unused dependencies

use std::path::Path;

use camino::Utf8PathBuf;
use ra_ap_ide_db::RootDatabase;
use ra_ap_load_cargo::{
    LoadCargoConfig, ProcMacroServerChoice, load_workspace_at,
};
use ra_ap_project_model::CargoConfig;
use tracing::{info, instrument};

use crate::ExtractError;

/// Load a Rust workspace into rust-analyzer's analysis database.
#[instrument]
pub(crate) fn load_workspace(
    path: impl AsRef<Path> + std::fmt::Debug,
) -> Result<RootDatabase, ExtractError> {
    let path = path.as_ref();

    // rust-analyzer requires absolute paths for file identification and
    // workspace discovery. canonicalize() also resolves symlinks to ensure
    // we work with the real filesystem location.
    let canonical = path.canonicalize().map_err(|e| {
        ExtractError::workspace_load(format!(
            "Failed to canonicalize path '{}': {e}",
            path.display()
        ))
    })?;

    // rust-analyzer's APIs expect UTF-8 paths internally.
    let workspace_path =
        Utf8PathBuf::from_path_buf(canonical).map_err(|p| {
            ExtractError::workspace_load(format!(
                "Path contains invalid UTF-8: {}",
                p.display()
            ))
        })?;

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
    let (db, _vfs, _proc_macro_server) = load_workspace_at(
        workspace_path.as_std_path(),
        &cargo_config,
        &load_config,
        &|msg| {
            info!(message = %msg, "workspace.progress");
        },
    )
    .map_err(ExtractError::workspace_load)?;

    Ok(db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_path() {
        let result = load_workspace("/nonexistent/path");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.is_workspace_load(),
            "Error should be workspace_load variant"
        );
    }

    /// Integration test for the production workspace loading codepath.
    ///
    /// This exercises load_workspace() and extract_symbol_graph() with a real
    /// Cargo project to ensure the full pipeline works end-to-end.
    #[test]
    fn test_load_workspace_and_extract() {
        use crate::extract_symbol_graph;

        let db = load_workspace("tests/fixtures/minimal_crate")
            .expect("load workspace");
        let graph = extract_symbol_graph(db);

        // Basic sanity checks
        assert_eq!(graph.crates.len(), 1);
        assert!(graph.crates.contains_key("minimal_crate"));

        let root = &graph.crates["minimal_crate"];

        // Should have Foo struct and bar function
        assert!(root.symbols.contains_key("Foo"), "Should have Foo struct");
        assert!(root.symbols.contains_key("bar"), "Should have bar function");

        // bar() -> Foo creates a dependency edge
        let has_edge = graph
            .edges
            .iter()
            .any(|e| e.from.contains("bar") && e.to.contains("Foo"));
        assert!(has_edge, "bar should depend on Foo");

        // Verify file paths are resolved for real workspaces (not empty like in
        // test fixtures which use virtual paths).
        for (name, symbol) in &root.symbols {
            assert!(
                !symbol.file.is_empty(),
                "Symbol '{}' should have a file path in real workspace",
                name
            );
            assert!(
                symbol.file.ends_with(".rs"),
                "Symbol '{}' file path should be a .rs file, got: {}",
                name,
                symbol.file
            );
        }
    }
}
