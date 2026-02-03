//! Orchestrator mode: Runs cargo and aggregates extraction results.
//!
//! The orchestrator:
//! 1. Reads workspace metadata to get the list of workspace crates
//! 2. Creates temp directories for intermediate JSON files and profiling
//! 3. Runs `cargo build --all-targets` with `RUSTC_WRAPPER` pointing to this binary
//! 4. Reads the per-crate JSON files written by the driver (already include costs)
//! 5. Merges them into a single `SymbolGraph` and outputs JSON
//!
//! Cost application (frontend, backend, overhead) happens in the driver
//! immediately after each crate compiles. This allows the driver to delete
//! raw profile files early, avoiding disk space exhaustion on large workspaces.

use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, ExitCode};
use std::{env, fs};

use anyhow::{Context, Result};
use cargo_metadata::MetadataCommand;
use tarjanize_schemas::{Crate, Module, SymbolGraph};
use tempfile::TempDir;
use tracing::{debug, info, warn};

use crate::driver::CrateResult;
use crate::{Cli, ENV_VERBOSITY};

/// Environment variable that tells the driver where to write output files.
pub const ENV_OUTPUT_DIR: &str = "TARJANIZE_OUTPUT_DIR";

/// Environment variable containing comma-separated workspace crate names.
/// The driver uses this to determine whether to extract symbols or just
/// pass through to rustc.
pub const ENV_WORKSPACE_CRATES: &str = "TARJANIZE_WORKSPACE_CRATES";

/// Environment variable that tells the driver where to write self-profile data.
/// When set, the driver adds `-Zself-profile` flags to rustc invocations.
pub const ENV_PROFILE_DIR: &str = "TARJANIZE_PROFILE_DIR";

/// Run the orchestrator: coordinate cargo check and aggregate results.
pub fn run(cli: &Cli) -> ExitCode {
    match run_inner(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_inner(cli: &Cli) -> Result<()> {
    // Determine the manifest path from CLI or use current directory.
    let manifest_path = find_manifest_path(cli)?;
    let manifest_dir = manifest_path
        .parent()
        .context("manifest path has no parent")?;

    // Get workspace metadata.
    let metadata = MetadataCommand::new()
        .manifest_path(&manifest_path)
        .exec()
        .context("failed to run cargo metadata")?;

    // Collect workspace member crate names.
    // If -p/--package is specified, only analyze those packages.
    // Otherwise, analyze all workspace members.
    let all_workspace_crates: Vec<String> = metadata
        .workspace_packages()
        .iter()
        .map(|pkg| pkg.name.to_string())
        .collect();

    let workspace_crates: Vec<String> = if cli.package.is_empty() {
        all_workspace_crates
    } else {
        // Validate that all specified packages exist in the workspace.
        for pkg in &cli.package {
            if !all_workspace_crates.contains(pkg) {
                anyhow::bail!(
                    "package `{pkg}` not found in workspace. \
                     Available packages: {}",
                    all_workspace_crates.join(", ")
                );
            }
        }
        cli.package.clone()
    };

    if workspace_crates.is_empty() {
        anyhow::bail!("no workspace members found");
    }

    info!(
        "analyzing {} workspace crate(s): {}",
        workspace_crates.len(),
        workspace_crates.join(", ")
    );

    // Create temp directory for intermediate output files.
    // Each crate/target combination writes a separate JSON file here.
    let output_dir =
        TempDir::new().context("failed to create temp directory")?;

    // Get path to this binary for RUSTC_WRAPPER.
    let self_exe =
        env::current_exe().context("failed to get current executable path")?;

    // Pass verbosity level to driver via environment variable.
    let verbosity_level = cli.verbose.tracing_level_filter().to_string();
    debug!(verbosity = %verbosity_level, "passing verbosity to driver");

    // Create a separate target directory for tarjanize.
    // This ensures cargo always invokes rustc (our wrapper) rather than using
    // cached artifacts from the normal target directory. This is the same
    // approach clippy uses.
    let target_dir =
        TempDir::new().context("failed to create target directory")?;

    // Create a directory for self-profile data.
    // The driver adds `-Zself-profile` flags to collect compilation timing.
    let profile_dir =
        TempDir::new().context("failed to create profile directory")?;

    // Run cargo build with our wrapper.
    // We use `cargo build` instead of `cargo check` because:
    // 1. Backend (LLVM) timing requires actual codegen, not just type checking
    // 2. Mono-items are only printed during codegen, not during check
    // 3. Without codegen, we can't measure backend costs at all
    debug!(manifest = %manifest_path.display(), "running cargo build");
    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("--all-targets")
        .arg("--manifest-path")
        .arg(&manifest_path);

    // Add -p flags for each specified package.
    // If no packages specified, cargo checks all workspace members.
    for pkg in &cli.package {
        cmd.arg("-p").arg(pkg);
    }

    // Disable incremental compilation to get consistent CGU naming.
    // Incremental builds use hash-based CGU names that don't match between
    // mono-items output and self-profile data.
    cmd.env(ENV_OUTPUT_DIR, output_dir.path())
        .env(ENV_WORKSPACE_CRATES, workspace_crates.join(","))
        .env(ENV_VERBOSITY, &verbosity_level)
        .env("RUSTC_WRAPPER", &self_exe)
        .env("CARGO_TARGET_DIR", target_dir.path())
        .env(ENV_PROFILE_DIR, profile_dir.path())
        .env("CARGO_INCREMENTAL", "0")
        .current_dir(manifest_dir);

    let status = cmd.status().context("failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("cargo build failed with status: {status}");
    }

    // Aggregate results from all JSON files in the output directory.
    // The driver has already applied costs and deleted profile files,
    // so we just need to merge the Crate structures.
    let graph = aggregate_results(output_dir.path(), &workspace_crates)?;

    // Output the combined symbol graph as JSON to the specified file.
    let file = fs::File::create(&cli.output).with_context(|| {
        format!("failed to create output file {}", cli.output.display())
    })?;
    serde_json::to_writer_pretty(file, &graph).with_context(|| {
        format!("failed to write output to {}", cli.output.display())
    })?;

    Ok(())
}

/// Find the Cargo.toml manifest path from CLI arguments.
/// Uses --manifest-path if provided, otherwise looks in current directory.
/// Returns an absolute path to avoid issues when changing working directory.
fn find_manifest_path(cli: &Cli) -> Result<std::path::PathBuf> {
    if let Some(path) = &cli.manifest_path {
        // Canonicalize to get an absolute path.
        // This is important because we later change the working directory
        // to the manifest's parent, which would break relative paths.
        let path = std::path::Path::new(path);
        return path.canonicalize().with_context(|| {
            format!("manifest path does not exist: {}", path.display())
        });
    }

    // Default to Cargo.toml in current directory.
    let manifest = env::current_dir()
        .context("failed to get current directory")?
        .join("Cargo.toml");

    if manifest.exists() {
        Ok(manifest)
    } else {
        anyhow::bail!("could not find Cargo.toml in current directory")
    }
}

/// Read all JSON files from the output directory and merge them into a `SymbolGraph`.
///
/// The driver has already applied costs to each crate, so we just merge.
fn aggregate_results(
    output_dir: &Path,
    workspace_crates: &[String],
) -> Result<SymbolGraph> {
    let mut crates: HashMap<String, Crate> = HashMap::new();

    // Read all .json files in the output directory.
    for entry in
        fs::read_dir(output_dir).context("failed to read output directory")?
    {
        let entry = entry?;
        let path = entry.path();

        // Skip mono-items files (they're .txt, not .json).
        if path.extension().is_some_and(|ext| ext == "json") {
            let content = fs::read_to_string(&path).with_context(|| {
                format!("failed to read {}", path.display())
            })?;

            let result: CrateResult = serde_json::from_str(&content)
                .with_context(|| {
                    format!("failed to parse {}", path.display())
                })?;

            // Merge this crate's data into the result.
            merge_crate(&mut crates, result.crate_name, result.crate_data);
        }
    }

    // Verify we got results for all workspace crates.
    for pkg_name in workspace_crates {
        let crate_name = pkg_name.replace('-', "_");
        if !crates.contains_key(&crate_name) {
            warn!(
                pkg_name,
                crate_name,
                "no extraction results (may be platform-specific or have no lib/bin targets)"
            );
        }
    }

    info!(crate_count = crates.len(), "aggregated crate results");

    Ok(SymbolGraph { crates })
}

/// Merge a Crate into the crates map.
///
/// If the crate already exists, merge the modules and take max of overhead.
/// Multiple targets (lib, test, bin) may produce separate Crate results
/// that need to be combined.
fn merge_crate(
    crates: &mut HashMap<String, Crate>,
    crate_name: String,
    crate_data: Crate,
) {
    use std::collections::hash_map::Entry;

    match crates.entry(crate_name) {
        Entry::Vacant(e) => {
            e.insert(crate_data);
        }
        Entry::Occupied(mut e) => {
            let existing = e.get_mut();

            // Merge root modules.
            merge_module(&mut existing.root, crate_data.root);

            // Take max of overhead (different targets may see different costs).
            existing.linking_ms = existing.linking_ms.max(crate_data.linking_ms);
            existing.metadata_ms =
                existing.metadata_ms.max(crate_data.metadata_ms);
        }
    }
}

/// Merge a module's contents into an existing module.
fn merge_module(existing: &mut Module, module: Module) {
    use std::collections::hash_map::Entry;

    // Merge symbols. For overlapping symbols (same code compiled in both lib
    // and test targets), take max costs as a conservative estimate.
    for (name, symbol) in module.symbols {
        match existing.symbols.entry(name) {
            Entry::Vacant(e) => {
                e.insert(symbol);
            }
            Entry::Occupied(mut e) => {
                // Symbol exists in both - take max of costs.
                let existing_sym = e.get_mut();
                existing_sym.frontend_cost_ms =
                    existing_sym.frontend_cost_ms.max(symbol.frontend_cost_ms);
                existing_sym.backend_cost_ms =
                    existing_sym.backend_cost_ms.max(symbol.backend_cost_ms);
                // Keep existing dependencies/kind/etc - they should be the same.
            }
        }
    }

    // Recursively merge submodules.
    for (name, submodule) in module.submodules {
        merge_submodule(&mut existing.submodules, name, submodule);
    }
}

/// Recursively merge a submodule.
fn merge_submodule(
    submodules: &mut HashMap<String, Module>,
    name: String,
    module: Module,
) {
    use std::collections::hash_map::Entry;

    match submodules.entry(name) {
        Entry::Vacant(e) => {
            e.insert(module);
        }
        Entry::Occupied(mut e) => {
            merge_module(e.get_mut(), module);
        }
    }
}
