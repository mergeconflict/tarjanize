//! Orchestrator mode: Runs cargo and aggregates extraction results.
//!
//! The orchestrator:
//! 1. Reads workspace metadata to get the list of workspace crates
//! 2. Creates a temp directory for intermediate JSON files
//! 3. Optionally runs a profiled build to collect timing data
//! 4. Runs `cargo check --all-targets` with `RUSTC_WRAPPER` pointing to this binary
//! 5. Reads the per-crate JSON files written by the driver
//! 6. Merges them into a single `SymbolGraph` and outputs JSON

use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, ExitCode};
use std::{env, fs};

use anyhow::{Context, Result};
use cargo_metadata::MetadataCommand;
use tarjanize_schemas::{Module, SymbolGraph};
use tempfile::TempDir;
use tracing::{debug, info, warn};

use crate::profile::ProfileData;
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
        .map(|pkg| pkg.name.clone())
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

    // Run cargo check with our wrapper.
    debug!(manifest = %manifest_path.display(), "running cargo check");
    let mut cmd = Command::new("cargo");
    cmd.arg("check")
        .arg("--all-targets")
        .arg("--manifest-path")
        .arg(&manifest_path);

    // Add -p flags for each specified package.
    // If no packages specified, cargo checks all workspace members.
    for pkg in &cli.package {
        cmd.arg("-p").arg(pkg);
    }

    cmd.env(ENV_OUTPUT_DIR, output_dir.path())
        .env(ENV_WORKSPACE_CRATES, workspace_crates.join(","))
        .env(ENV_VERBOSITY, &verbosity_level)
        .env("RUSTC_WRAPPER", &self_exe)
        .env("CARGO_TARGET_DIR", target_dir.path())
        .env(ENV_PROFILE_DIR, profile_dir.path())
        .current_dir(manifest_dir);

    let status = cmd.status().context("failed to run cargo check")?;

    if !status.success() {
        anyhow::bail!("cargo check failed with status: {status}");
    }

    // Load profile data from the self-profile output.
    let profile_data = ProfileData::load_from_dir(profile_dir.path());

    // Aggregate results from all JSON files in the output directory.
    let graph =
        aggregate_results(output_dir.path(), &workspace_crates, &profile_data)?;

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
/// If profile data is provided, applies timing data to symbol costs.
fn aggregate_results(
    output_dir: &Path,
    workspace_crates: &[String],
    profile_data: &ProfileData,
) -> Result<SymbolGraph> {
    let mut crates: HashMap<String, Module> = HashMap::new();

    // Read all .json files in the output directory.
    for entry in
        fs::read_dir(output_dir).context("failed to read output directory")?
    {
        let entry = entry?;
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "json") {
            let content = fs::read_to_string(&path).with_context(|| {
                format!("failed to read {}", path.display())
            })?;

            let partial: PartialResult = serde_json::from_str(&content)
                .with_context(|| {
                    format!("failed to parse {}", path.display())
                })?;

            // Merge this crate's module into the result.
            // If we already have this crate (e.g., from lib target), merge the modules.
            // If this is a new target (e.g., test), we need to merge symbols.
            merge_module(&mut crates, partial.crate_name, partial.module);
        }
    }

    // Verify we got results for all workspace crates.
    // Some crates might not have been compiled (e.g., platform-specific).
    // Note: Cargo package names use hyphens but rustc crate names use underscores.
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

    // Apply profile timing data to symbol costs.
    // Symbol keys use DefPath format, matching profile data keys directly.
    for (crate_name, module) in &mut crates {
        apply_profile_costs(module, crate_name, profile_data);
    }
    info!(
        profile_paths = profile_data.len(),
        "applied profile timing data to symbols"
    );

    Ok(SymbolGraph { crates })
}

/// Merge a module into the crates map.
/// If the crate already exists, merge the new module's contents into it.
fn merge_module(
    crates: &mut HashMap<String, Module>,
    crate_name: String,
    module: Module,
) {
    use std::collections::hash_map::Entry;

    match crates.entry(crate_name) {
        Entry::Vacant(e) => {
            e.insert(module);
        }
        Entry::Occupied(mut e) => {
            // Merge symbols and submodules.
            let existing = e.get_mut();

            // Merge symbols (test targets may add test-only symbols).
            for (name, symbol) in module.symbols {
                existing.symbols.entry(name).or_insert(symbol);
            }

            // Recursively merge submodules.
            for (name, submodule) in module.submodules {
                merge_submodule(&mut existing.submodules, name, submodule);
            }
        }
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
            let existing = e.get_mut();
            for (sym_name, symbol) in module.symbols {
                existing.symbols.entry(sym_name).or_insert(symbol);
            }
            for (sub_name, sub) in module.submodules {
                merge_submodule(&mut existing.submodules, sub_name, sub);
            }
        }
    }
}

/// Intermediate result written by the driver for each crate/target.
#[derive(serde::Deserialize)]
struct PartialResult {
    crate_name: String,
    module: Module,
}

/// Recursively apply profile timing data to symbols in a module.
///
/// For each symbol, constructs its full path and looks up timing data directly.
/// Symbol keys use rustc's `DefPath` format (e.g., `crate::mod::{{impl}}[1]`), which
/// matches the `-Zself-profile` output format, enabling direct lookup.
///
/// # Panics
///
/// Panics if no timing data exists for a symbol. This strict behavior ensures we
/// catch profile extraction bugs early.
fn apply_profile_costs(
    module: &mut Module,
    path_prefix: &str,
    profile_data: &ProfileData,
) {
    for (name, symbol) in &mut module.symbols {
        let full_path = format!("{path_prefix}::{name}");

        // Look up timing data directly - symbol keys are DefPath format.
        let cost = profile_data.get_cost_ms(&full_path).unwrap_or_else(|| {
            panic!(
                "no timing data for symbol: {full_path}\n\
                 hint: check that -Zself-profile captured this symbol"
            )
        });

        symbol.cost = cost;
    }

    for (submod_name, submodule) in &mut module.submodules {
        let submod_path = format!("{path_prefix}::{submod_name}");
        apply_profile_costs(submodule, &submod_path, profile_data);
    }
}
