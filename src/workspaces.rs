use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use ra_ap_ide_db::RootDatabase;
use ra_ap_load_cargo::{
    LoadCargoConfig, ProcMacroServerChoice, load_workspace_at,
};
use ra_ap_project_model::CargoConfig;

/// Load a Rust workspace into rust-analyzer's analysis database.
pub fn load_workspace(path: &str) -> Result<RootDatabase> {
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
    let (db, _vfs, _proc_macro_server) = load_workspace_at(
        workspace_path.as_std_path(),
        &cargo_config,
        &load_config,
        &|msg| {
            println!("Progress: {}", msg);
        },
    )?;

    Ok(db)
}
