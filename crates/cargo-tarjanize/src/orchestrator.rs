//! Orchestrator mode: Runs cargo and aggregates extraction results.
//!
//! The orchestrator:
//! 1. Reads workspace metadata to get the list of workspace crates
//! 2. Creates temp directories for intermediate JSON files and profiling
//! 3. Runs `cargo build --all-targets` with `RUSTC_WRAPPER` pointing to this binary
//! 4. Reads mono-items output files written by the driver for backend cost distribution
//! 5. Reads the per-crate JSON files written by the driver
//! 6. Applies frontend/backend costs from profile data
//! 7. Merges them into a single `SymbolGraph` and outputs JSON

use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, ExitCode};
use std::{env, fs};

use anyhow::{Context, Result};
use cargo_metadata::MetadataCommand;
use tarjanize_schemas::{Crate, Module, SymbolGraph};
use tempfile::TempDir;
use tracing::{debug, info, warn};

use crate::mono_items::MonoItemsMap;
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

    // Run cargo build with our wrapper, capturing stderr for mono-items.
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

    let output = cmd.output().context("failed to run cargo build")?;

    if !output.status.success() {
        // Print stderr on failure for debugging.
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("{stderr}");
        anyhow::bail!("cargo build failed with status: {}", output.status);
    }

    // Parse mono-items from files written by the driver.
    // The driver redirects rustc's stdout to these files.
    let mono_items_by_crate =
        load_mono_items_from_dir(output_dir.path(), &workspace_crates);

    // Load profile data from the self-profile output.
    let profile_data = ProfileData::load_from_dir(profile_dir.path());

    // Aggregate results from all JSON files in the output directory.
    let graph = aggregate_results(
        output_dir.path(),
        &workspace_crates,
        &profile_data,
        &mono_items_by_crate,
    )?;

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

/// Load mono-items from files written by the driver.
///
/// The driver redirects rustc's `-Zprint-mono-items=yes` output to files
/// in the output directory, named `{crate_name}_mono_items.txt`.
fn load_mono_items_from_dir(
    output_dir: &Path,
    workspace_crates: &[String],
) -> HashMap<String, MonoItemsMap> {
    let mut mono_items_by_crate = HashMap::new();

    for pkg_name in workspace_crates {
        let crate_name = pkg_name.replace('-', "_");
        let mono_items_path = output_dir.join(format!("{crate_name}_mono_items.txt"));

        if !mono_items_path.exists() {
            debug!(crate_name, "no mono-items file found");
            continue;
        }

        let content = match fs::read_to_string(&mono_items_path) {
            Ok(c) => c,
            Err(e) => {
                warn!(
                    path = %mono_items_path.display(),
                    error = %e,
                    "failed to read mono-items file"
                );
                continue;
            }
        };

        let map = MonoItemsMap::parse(content.as_bytes(), &crate_name);

        let total_items: usize = map.cgu_to_items.values().map(Vec::len).sum();
        debug!(
            crate_name,
            cgu_count = map.cgu_to_items.len(),
            total_items,
            "loaded mono-items from file"
        );

        if !map.is_empty() {
            mono_items_by_crate.insert(crate_name, map);
        }
    }

    mono_items_by_crate
}

/// Read all JSON files from the output directory and merge them into a `SymbolGraph`.
///
/// Applies profile timing data to symbols and crate overhead.
fn aggregate_results(
    output_dir: &Path,
    workspace_crates: &[String],
    profile_data: &ProfileData,
    mono_items_by_crate: &HashMap<String, MonoItemsMap>,
) -> Result<SymbolGraph> {
    // First pass: collect all modules.
    let mut modules: HashMap<String, Module> = HashMap::new();

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
            merge_module(&mut modules, partial.crate_name, partial.module);
        }
    }

    // Verify we got results for all workspace crates.
    for pkg_name in workspace_crates {
        let crate_name = pkg_name.replace('-', "_");
        if !modules.contains_key(&crate_name) {
            warn!(
                pkg_name,
                crate_name,
                "no extraction results (may be platform-specific or have no lib/bin targets)"
            );
        }
    }

    // Distribute backend costs from CGU timing to symbols.
    let backend_costs = distribute_backend_costs(profile_data, mono_items_by_crate);
    debug!(
        symbol_count = backend_costs.len(),
        "distributed backend costs to symbols"
    );

    // Build final crate map with costs applied.
    let mut crates: HashMap<String, Crate> = HashMap::new();

    for (crate_name, mut module) in modules {
        // Apply frontend costs from profile data.
        apply_frontend_costs(&mut module, &crate_name, profile_data);

        // Apply backend costs from CGU distribution.
        apply_backend_costs(&mut module, &crate_name, &backend_costs);

        // Get crate overhead.
        let overhead = profile_data
            .get_crate_overhead(&crate_name)
            .cloned()
            .unwrap_or_default();

        crates.insert(
            crate_name,
            Crate {
                linking_ms: overhead.linking_ms,
                metadata_ms: overhead.metadata_ms,
                root: module,
            },
        );
    }

    info!(
        frontend_paths = profile_data.frontend_count(),
        cgu_count = profile_data.cgu_count(),
        backend_symbols = backend_costs.len(),
        "applied profile timing data"
    );

    Ok(SymbolGraph { crates })
}

/// Distribute backend costs from CGU timing to individual symbols.
///
/// For each CGU, we have:
/// - Total CGU cost from profile data
/// - List of symbols in that CGU from mono-items
///
/// We distribute the CGU cost equally among its symbols. If a symbol appears
/// in multiple CGUs (due to inlining), it gets a share from each.
fn distribute_backend_costs(
    profile_data: &ProfileData,
    mono_items_by_crate: &HashMap<String, MonoItemsMap>,
) -> HashMap<String, f64> {
    let mut costs: HashMap<String, f64> = HashMap::new();

    let cgu_costs = profile_data.cgu_costs();

    for mono_items in mono_items_by_crate.values() {
        for (cgu_name, items) in &mono_items.cgu_to_items {
            // Look up the CGU cost.
            let Some(cgu_duration) = cgu_costs.get(cgu_name) else {
                // CGU not found in profile data - might be a different crate.
                debug!(cgu_name, "CGU from mono-items not found in profile data");
                continue;
            };

            if items.is_empty() {
                continue;
            }

            // Distribute cost equally among items in this CGU.
            #[expect(
                clippy::cast_precision_loss,
                reason = "CGU item counts are small, precision loss is negligible"
            )]
            let cost_per_item =
                cgu_duration.as_millis_f64() / items.len() as f64;

            for item in items {
                *costs.entry(item.clone()).or_default() += cost_per_item;
            }
        }
    }

    costs
}

/// Merge a module into the modules map.
/// If the crate already exists, merge the new module's contents into it.
fn merge_module(
    modules: &mut HashMap<String, Module>,
    crate_name: String,
    module: Module,
) {
    use std::collections::hash_map::Entry;

    match modules.entry(crate_name) {
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

/// Recursively apply frontend costs from profile data to symbols.
///
/// For each symbol, constructs its full path and looks up timing data.
/// Symbol keys use rustc's `DefPath` format, which matches the profile output.
fn apply_frontend_costs(
    module: &mut Module,
    path_prefix: &str,
    profile_data: &ProfileData,
) {
    for (name, symbol) in &mut module.symbols {
        let full_path = format!("{path_prefix}::{name}");

        // Look up frontend timing data.
        // Use 0.0 if not found (some symbols may not have frontend cost).
        let cost = profile_data.get_frontend_cost_ms(&full_path).unwrap_or(0.0);
        symbol.frontend_cost_ms = cost;
    }

    for (submod_name, submodule) in &mut module.submodules {
        let submod_path = format!("{path_prefix}::{submod_name}");
        apply_frontend_costs(submodule, &submod_path, profile_data);
    }
}

/// Recursively apply backend costs from CGU distribution to symbols.
///
/// For impl blocks, we sum costs from all paths that start with the impl's
/// anchor path. This handles both trait impl methods (which normalize to
/// `Type::{{impl}}`) and inherent methods (which appear as `Type::method`).
fn apply_backend_costs(
    module: &mut Module,
    path_prefix: &str,
    backend_costs: &HashMap<String, f64>,
) {
    use tarjanize_schemas::SymbolKind;

    for (name, symbol) in &mut module.symbols {
        let full_path = format!("{path_prefix}::{name}");

        // Look up backend cost from distribution.
        let mut cost = backend_costs.get(&full_path).copied().unwrap_or(0.0);

        // For impl blocks, sum all costs from paths starting with any anchor.
        // Mono-items for inherent methods appear as `Type::method_name`,
        // while trait impl methods are normalized to `Type::{{impl}}`.
        // By summing all paths with matching prefix, we capture both.
        if let SymbolKind::Impl { anchors, .. } = &symbol.kind {
            for anchor in anchors {
                // Sum costs from all paths that start with this anchor.
                // This includes both `Type::method` and `Type::{{impl}}` paths.
                let prefix = format!("{anchor}::");
                for (path, &path_cost) in backend_costs {
                    if path.starts_with(&prefix) {
                        cost += path_cost;
                    }
                }
            }
        }

        symbol.backend_cost_ms = cost;
    }

    for (submod_name, submodule) in &mut module.submodules {
        let submod_path = format!("{path_prefix}::{submod_name}");
        apply_backend_costs(submodule, &submod_path, backend_costs);
    }
}
