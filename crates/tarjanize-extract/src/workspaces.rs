//! Workspace loading for tarjanize.
//!
//! This module initializes rust-analyzer's analysis database for a Cargo
//! workspace.

use std::path::Path;

use anyhow::{Context, anyhow};
use camino::Utf8PathBuf;
use ra_ap_ide_db::RootDatabase;
use ra_ap_load_cargo::{
    LoadCargoConfig, ProcMacroServerChoice, load_workspace_at,
};
use ra_ap_project_model::CargoConfig;
use tracing::{debug_span, info};

/// Load a Rust workspace into rust-analyzer's analysis database.
pub(crate) fn load_workspace(
    path: impl AsRef<Path>,
) -> anyhow::Result<RootDatabase> {
    let path = path.as_ref();
    let _span = debug_span!("load_workspace", path = %path.display()).entered();

    // rust-analyzer requires absolute paths for file identification and
    // workspace discovery. canonicalize() also resolves symlinks to ensure
    // we work with the real filesystem location.
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize path '{}'", path.display()))?;

    // rust-analyzer's APIs expect UTF-8 paths internally.
    let workspace_path = Utf8PathBuf::from_path_buf(canonical.clone())
        .map_err(|_| anyhow!("path contains invalid UTF-8: {}", canonical.display()))?;

    // CargoConfig controls how Cargo projects are interpreted (e.g., which
    // features to enable, target platform). Default settings work for most
    // cases.
    let cargo_config = CargoConfig::default();

    let load_config = LoadCargoConfig {
        // Run `cargo check` to execute build scripts and capture their
        // outputs. Critical for tarjanize because build.rs files generate
        // code at compile time (e.g., bindings, lookup tables), and we need
        // to see dependencies in that generated code.
        load_out_dirs_from_check: true,

        // Use the system's rustc to expand procedural macros. Critical for
        // tarjanize because proc macros like #[derive(Serialize)] generate
        // impl blocks that create dependencies (e.g., impl Serialize depends
        // on serde::Serialize).
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,

        // Fill caches lazily so we don't analyze crates we don't care about.
        prefill_caches: false,

        // Parallelize proc macro expansion across available CPU cores.
        proc_macro_processes: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    };

    // Load the workspace into rust-analyzer's analysis database. This:
    // - Parses all Cargo.toml files to discover crates and dependencies
    // - Parses Rust source files into syntax trees
    // - Builds the semantic model (name resolution, type inference, trait
    //   solving)
    let (db, _, _) = load_workspace_at(
        workspace_path.as_std_path(),
        &cargo_config,
        &load_config,
        &|msg| {
            info!(message = %msg, "workspace.progress");
        },
    )
    .context("failed to load workspace")?;

    Ok(db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_workspace_succeeds() {
        load_workspace("tests/fixtures/minimal")
            .expect("should load valid workspace");
    }

    #[test]
    fn test_load_workspace_fails_for_nonexistent_path() {
        let err = load_workspace("/nonexistent/path")
            .expect_err("should fail for nonexistent path");
        assert!(err.to_string().contains("failed to canonicalize path"));
    }
}
