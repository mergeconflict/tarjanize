//! Orchestrator mode: Runs cargo and aggregates extraction results.
//!
//! The build process has two steps:
//!
//! 1. **Profile build** (optional, skipped with `--no-profile`):
//!    Run `cargo +nightly build` with `-Zself-profile` but NO `RUSTC_WRAPPER`.
//!    Produces clean profile data without extraction overhead.
//!
//! 2. **Extraction build**:
//!    Run `cargo build` with `RUSTC_WRAPPER` to extract symbols and mono-items.
//!    Profile data (if collected) is merged with extracted symbols.
//!
//! Separating profiling from extraction ensures profile data only contains
//! real compilation work, not overhead from our extraction callbacks.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::{env, fs};

use anyhow::{Context, Result};
use cargo_metadata::{DependencyKind, Metadata, MetadataCommand};
use tarjanize_schemas::{Package, SymbolGraph, SymbolKind};
use tempfile::TempDir;
use tracing::{debug, info, warn};

use crate::driver::CrateResult;
use crate::{Cli, ENV_VERBOSITY};

/// Configuration for running cargo builds.
/// Bundles all the parameters needed by `run_profile_build` and `run_extraction_build`.
struct BuildConfig<'a> {
    manifest_path: &'a Path,
    manifest_dir: &'a Path,
    workspace_crates: &'a [String],
    /// Maps package name to its manifest directory path.
    /// Used to identify which package integration tests belong to.
    workspace_paths: &'a HashMap<String, PathBuf>,
    self_exe: &'a Path,
    verbosity_level: &'a str,
    target_dir: &'a Path,
    packages: &'a [String],
}

/// Environment variable that tells the driver where to write output files.
pub const ENV_OUTPUT_DIR: &str = "TARJANIZE_OUTPUT_DIR";

/// Environment variable containing comma-separated workspace crate names.
/// The driver uses this to determine whether to extract symbols or just
/// pass through to rustc.
pub const ENV_WORKSPACE_CRATES: &str = "TARJANIZE_WORKSPACE_CRATES";

/// Environment variable that tells the driver where to write self-profile data.
/// When set, the driver adds `-Zself-profile` flags to rustc invocations.
pub const ENV_PROFILE_DIR: &str = "TARJANIZE_PROFILE_DIR";

/// Environment variable that tells the driver to skip profiling.
/// When set to "1", the driver does not add `-Zself-profile` flags
/// and does not apply costs. Used with --no-profile flag.
pub const ENV_SKIP_PROFILE: &str = "TARJANIZE_SKIP_PROFILE";

/// Environment variable containing workspace member paths.
/// Format: `pkg1=/path/to/pkg1,pkg2=/path/to/pkg2`
///
/// Used to identify which package an integration test belongs to.
/// Integration tests have crate names like `sync_mpsc` that don't match
/// package names like `tokio`, but their source files are within the
/// package directory, so we can match by path.
pub const ENV_WORKSPACE_PATHS: &str = "TARJANIZE_WORKSPACE_PATHS";

/// Filename for the crate mapping file within the output directory.
/// Maps crate names (underscores) to package names (may have hyphens).
const CRATE_MAPPING_FILENAME: &str = "crate_mapping.json";

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

    // Collect workspace member crate names and their paths.
    // If -p/--package is specified, only analyze those packages.
    // Otherwise, analyze all workspace members.
    //
    // The paths are used to identify which package integration tests belong to,
    // since integration tests have crate names (e.g., `sync_mpsc`) that differ
    // from their package name (e.g., `tokio`).
    let mut all_workspace_crates: Vec<String> = Vec::new();
    let mut workspace_paths: HashMap<String, PathBuf> = HashMap::new();
    for pkg in metadata.workspace_packages() {
        all_workspace_crates.push(pkg.name.to_string());
        if let Some(manifest_dir) = pkg.manifest_path.parent() {
            workspace_paths.insert(pkg.name.to_string(), manifest_dir.into());
        }
    }

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

    // Get path to this binary for RUSTC_WRAPPER.
    let self_exe =
        env::current_exe().context("failed to get current executable path")?;

    // Pass verbosity level to driver via environment variable.
    let verbosity_level = cli.verbose.tracing_level_filter().to_string();
    debug!(verbosity = %verbosity_level, "passing verbosity to driver");

    // Create temp directories. The target directory is shared between builds
    // so external dependencies compiled in the profile build are reused.
    let target_dir =
        TempDir::new().context("failed to create target directory")?;
    let output_dir =
        TempDir::new().context("failed to create output directory")?;

    // Write crate mapping file for use during result aggregation.
    write_crate_mapping(&metadata, output_dir.path())?;

    // Create a directory for self-profile data.
    // Each target gets its own subdirectory to enable cleanup after processing.
    let profile_dir =
        TempDir::new().context("failed to create profile directory")?;

    let config = BuildConfig {
        manifest_path: &manifest_path,
        manifest_dir,
        workspace_crates: &workspace_crates,
        workspace_paths: &workspace_paths,
        self_exe: &self_exe,
        verbosity_level: &verbosity_level,
        target_dir: target_dir.path(),
        packages: &cli.package,
    };

    // Single cargo build pass: driver does profiling, extraction, cost
    // application, and cleanup for each crate as it compiles.
    run_build(
        &config,
        output_dir.path(),
        profile_dir.path(),
        cli.no_profile,
    )?;

    // Aggregate results from all JSON files.
    // Costs are already applied by the driver.
    let graph =
        aggregate_results(output_dir.path(), &workspace_crates, &metadata)?;

    // Output the combined symbol graph as JSON to the specified file.
    let file = fs::File::create(&cli.output).with_context(|| {
        format!("failed to create output file {}", cli.output.display())
    })?;
    serde_json::to_writer_pretty(file, &graph).with_context(|| {
        format!("failed to write output to {}", cli.output.display())
    })?;

    Ok(())
}

/// Run a single cargo build that does profiling, extraction, and cleanup.
///
/// Uses `RUSTC_WRAPPER` to run our custom driver. For workspace crates, the
/// driver:
/// 1. Adds `-Zself-profile` flags (unless `no_profile` is true)
/// 2. Extracts symbols via `after_analysis` callback
/// 3. Applies costs from profile data
/// 4. Deletes profile files immediately to avoid filling /tmp
///
/// External dependencies compile normally without profiling or extraction.
fn run_build(
    config: &BuildConfig<'_>,
    output_dir: &Path,
    profile_dir: &Path,
    no_profile: bool,
) -> Result<()> {
    info!("running build");

    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("--all-targets")
        .arg("--manifest-path")
        .arg(config.manifest_path);

    for pkg in config.packages {
        cmd.arg("-p").arg(pkg);
    }

    // Format workspace paths as "pkg1=/path1,pkg2=/path2" for the driver.
    let workspace_paths_str: String = config
        .workspace_paths
        .iter()
        .map(|(name, path)| format!("{}={}", name, path.display()))
        .collect::<Vec<_>>()
        .join(",");

    cmd.env(ENV_OUTPUT_DIR, output_dir)
        .env(ENV_WORKSPACE_CRATES, config.workspace_crates.join(","))
        .env(ENV_WORKSPACE_PATHS, &workspace_paths_str)
        .env(ENV_PROFILE_DIR, profile_dir)
        .env(ENV_VERBOSITY, config.verbosity_level)
        .env("RUSTC_WRAPPER", config.self_exe)
        .env("CARGO_TARGET_DIR", config.target_dir)
        .env("CARGO_INCREMENTAL", "0")
        .current_dir(config.manifest_dir);

    // Skip profiling if requested.
    if no_profile {
        cmd.env(ENV_SKIP_PROFILE, "1");
    }

    debug!(manifest = %config.manifest_path.display(), no_profile, "running cargo build");
    let status = cmd.status().context("failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("cargo build failed with status: {status}");
    }

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

/// Read all JSON files from the output directory and build a `SymbolGraph`.
///
/// Costs are already applied by the driver during compilation.
fn aggregate_results(
    output_dir: &Path,
    workspace_crates: &[String],
    metadata: &Metadata,
) -> Result<SymbolGraph> {
    // Load the crate mapping that was written before cargo build.
    let crate_mapping = load_crate_mapping(output_dir)?;

    let mut packages: HashMap<String, Package> = HashMap::new();

    // Read all .json files in the output directory (excluding crate_mapping.json).
    for entry in
        fs::read_dir(output_dir).context("failed to read output directory")?
    {
        let entry = entry?;
        let path = entry.path();

        // Skip non-JSON files and the crate mapping file.
        let is_json = path.extension().is_some_and(|ext| ext == "json");
        let is_mapping = path
            .file_name()
            .is_some_and(|n| n == CRATE_MAPPING_FILENAME);
        if !is_json || is_mapping {
            continue;
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;

        let result: CrateResult = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse {}", path.display()))?;

        // Parse filename to extract target key.
        let target_key =
            parse_target_key_from_filename(&path, &result.crate_name);

        // Look up the package name from the crate mapping.
        let package_name = crate_mapping
            .get(&result.crate_name)
            .cloned()
            .unwrap_or_else(|| result.crate_name.replace('_', "-"));

        // Insert into the package's targets map.
        let package = packages.entry(package_name).or_default();
        package.targets.insert(target_key, result.crate_data);
    }

    // Verify we got results for all workspace packages.
    for pkg_name in workspace_crates {
        if !packages.contains_key(pkg_name) {
            warn!(
                pkg_name,
                "no extraction results (may be platform-specific or have no lib/bin targets)"
            );
        }
    }

    // Populate target-level dependencies from cargo metadata.
    populate_dependencies(&mut packages, metadata);

    // Transform symbol paths from crate-name format to package/target format.
    transform_symbol_paths(&mut packages, &crate_mapping);

    info!(package_count = packages.len(), "aggregated package results");

    Ok(SymbolGraph { packages })
}

/// Parse target key from output filename.
///
/// Filename format: `{crate_name}_{target_key}.json`
/// where `target_key` has `/` replaced with `_` (e.g., `bin_foo` for `bin/foo`).
fn parse_target_key_from_filename(path: &Path, crate_name: &str) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let prefix = format!("{crate_name}_");
    if let Some(suffix) = stem.strip_prefix(&prefix) {
        // Convert target_name back to target/name for composite targets.
        // Filenames use underscore (e.g., `bin_foo.json`) because `/` isn't
        // valid in filenames, but target keys use slash (e.g., `bin/foo`).
        if let Some(rest) = suffix.strip_prefix("bin_") {
            format!("bin/{rest}")
        } else if let Some(rest) = suffix.strip_prefix("test_") {
            // Integration tests: test_foo → test/foo
            // Unit tests remain as just "test" (no underscore suffix).
            format!("test/{rest}")
        } else if let Some(rest) = suffix.strip_prefix("example_") {
            format!("example/{rest}")
        } else if let Some(rest) = suffix.strip_prefix("bench_") {
            format!("bench/{rest}")
        } else {
            suffix.to_string()
        }
    } else {
        "lib".to_string()
    }
}

/// Populate target dependencies from cargo metadata.
///
/// For each target in the symbol graph, extract its dependencies from the
/// corresponding package in cargo metadata. Dependencies are formatted as
/// `"{package}/{target}"` (e.g., `"serde/lib"`, `"my-package/lib"`).
fn populate_dependencies(
    packages: &mut HashMap<String, Package>,
    metadata: &Metadata,
) {
    for cargo_pkg in metadata.workspace_packages() {
        let Some(pkg) = packages.get_mut(cargo_pkg.name.as_str()) else {
            // Package exists in metadata but wasn't extracted (platform-specific, etc.)
            continue;
        };

        // Collect dependencies by kind from cargo metadata.
        let mut normal_deps: HashSet<String> = HashSet::new();
        let mut dev_deps: HashSet<String> = HashSet::new();

        for dep in &cargo_pkg.dependencies {
            // Format as package/lib (assuming lib target for dependencies).
            let dep_ref = format!("{}/lib", dep.name);

            match dep.kind {
                DependencyKind::Normal => {
                    normal_deps.insert(dep_ref);
                }
                DependencyKind::Development => {
                    dev_deps.insert(dep_ref);
                }
                DependencyKind::Build | DependencyKind::Unknown => {
                    // Build dependencies don't affect runtime compilation order.
                    // Unknown kinds are ignored.
                }
            }
        }

        // Populate dependencies for each target.
        for (target_key, crate_data) in &mut pkg.targets {
            if target_key == "lib" {
                // Lib target: only normal dependencies.
                crate_data.dependencies.clone_from(&normal_deps);
            } else if target_key == "test" || target_key.starts_with("test/") {
                // Test targets: normal + dev deps + own lib.
                // "test" = unit tests (lib compiled with --test)
                // "test/{name}" = integration tests (tests/*.rs files)
                let mut deps = normal_deps.clone();
                deps.extend(dev_deps.clone());
                deps.insert(format!("{}/lib", cargo_pkg.name));
                crate_data.dependencies = deps;
            } else if target_key.starts_with("bin/") {
                // Bin target: normal deps + own lib.
                let mut deps = normal_deps.clone();
                deps.insert(format!("{}/lib", cargo_pkg.name));
                crate_data.dependencies = deps;
            } else {
                // Other targets (example, bench): normal deps + own lib.
                let mut deps = normal_deps.clone();
                deps.insert(format!("{}/lib", cargo_pkg.name));
                crate_data.dependencies = deps;
            }
        }
    }
}

// Note: The merge_crate/merge_module functions were removed as part of the
// target separation refactor. With separate lib/test/bin targets, we no longer
// merge results - each target gets its own entry in Package.targets.

/// Write a crate-to-package mapping file for the orchestrator to use later.
///
/// This maps crate names (as rustc sees them, with underscores) to package
/// names (as Cargo defines them, may have hyphens). The mapping is used when
/// aggregating results to correctly associate driver output with packages.
///
/// The mapping file is written to `{output_dir}/crate_mapping.json`.
fn write_crate_mapping(
    metadata: &Metadata,
    output_dir: &Path,
) -> Result<HashMap<String, String>> {
    let mut mapping = HashMap::new();

    for pkg in metadata.workspace_packages() {
        // For each lib target, map the crate name to the package name.
        // The crate name for a lib target is either explicitly set in
        // Cargo.toml [lib] name, or defaults to the package name with
        // hyphens replaced by underscores.
        for target in &pkg.targets {
            // Check if this is a lib target by looking for Lib kind.
            let is_lib = target
                .kind
                .iter()
                .any(|k| matches!(k, cargo_metadata::TargetKind::Lib));
            if is_lib {
                // target.name is the crate name (underscores).
                mapping.insert(target.name.clone(), pkg.name.to_string());
            }
        }
        // Also add a mapping for the default crate name (package name with
        // hyphens → underscores). This handles cases where there's no explicit
        // lib target or it uses the default name.
        let default_crate_name = pkg.name.replace('-', "_");
        mapping
            .entry(default_crate_name)
            .or_insert_with(|| pkg.name.to_string());
    }

    let mapping_path = output_dir.join(CRATE_MAPPING_FILENAME);
    let file = fs::File::create(&mapping_path).with_context(|| {
        format!("failed to create mapping file {}", mapping_path.display())
    })?;
    serde_json::to_writer(file, &mapping).with_context(|| {
        format!("failed to write mapping file {}", mapping_path.display())
    })?;

    debug!(
        path = %mapping_path.display(),
        entries = mapping.len(),
        "wrote crate mapping file"
    );

    Ok(mapping)
}

/// Load the crate mapping from a file.
fn load_crate_mapping(output_dir: &Path) -> Result<HashMap<String, String>> {
    let mapping_path = output_dir.join(CRATE_MAPPING_FILENAME);
    let content = fs::read_to_string(&mapping_path).with_context(|| {
        format!("failed to read mapping file {}", mapping_path.display())
    })?;
    let mapping: HashMap<String, String> = serde_json::from_str(&content)
        .with_context(|| {
            format!("failed to parse mapping file {}", mapping_path.display())
        })?;
    Ok(mapping)
}

/// Transform symbol paths from crate-name format to package/target format.
///
/// Symbol dependencies and impl anchors use paths like `crate_name::module::symbol`.
/// This function transforms them to `[package-name/target]::module::symbol`.
///
/// For same-package references, we use the current target (a binary referencing
/// its own symbols stays within that binary). For cross-package references, we
/// use `/lib` since that's what rustc resolves for `use other_crate::foo`.
fn transform_symbol_paths(
    packages: &mut HashMap<String, Package>,
    crate_mapping: &HashMap<String, String>,
) {
    for (package_name, package) in &mut *packages {
        for (target_key, crate_data) in &mut package.targets {
            transform_module_paths(
                &mut crate_data.root,
                crate_mapping,
                package_name,
                target_key,
            );
        }
    }
}

/// Transform paths in a module and its submodules recursively.
fn transform_module_paths(
    module: &mut tarjanize_schemas::Module,
    crate_mapping: &HashMap<String, String>,
    current_package: &str,
    current_target: &str,
) {
    for symbol in module.symbols.values_mut() {
        // Transform dependencies.
        symbol.dependencies = symbol
            .dependencies
            .iter()
            .map(|dep| {
                transform_path(
                    dep,
                    crate_mapping,
                    current_package,
                    current_target,
                )
            })
            .collect();

        // Transform impl anchors if this is an impl block.
        if let SymbolKind::Impl { anchors, .. } = &mut symbol.kind {
            *anchors = anchors
                .iter()
                .map(|anchor| {
                    transform_path(
                        anchor,
                        crate_mapping,
                        current_package,
                        current_target,
                    )
                })
                .collect();
        }
    }

    for submodule in module.submodules.values_mut() {
        transform_module_paths(
            submodule,
            crate_mapping,
            current_package,
            current_target,
        );
    }
}

/// Transform a single symbol path from crate-name to package/target format.
///
/// Input:  `crate_name::module::symbol`
/// Output: `[package-name/target]::module::symbol`
///
/// For same-package references, uses the current target. For cross-package
/// references, uses `/lib`. If the crate name is not in the mapping (external
/// crate), the path is returned unchanged.
fn transform_path(
    path: &str,
    crate_mapping: &HashMap<String, String>,
    current_package: &str,
    current_target: &str,
) -> String {
    // Parse the crate name from the path (everything before the first `::`)
    let Some((crate_name, rest)) = path.split_once("::") else {
        // No `::` in path - return unchanged.
        return path.to_string();
    };

    // Look up the package name. If not found, this is an external crate.
    let Some(package_name) = crate_mapping.get(crate_name) else {
        // External crate - return unchanged.
        return path.to_string();
    };

    // Determine the target: same package uses current target, cross-package uses lib.
    let target = if package_name == current_package {
        current_target
    } else {
        "lib"
    };

    format!("[{package_name}/{target}]::{rest}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a crate mapping for tests.
    fn make_mapping(entries: &[(&str, &str)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_transform_path_same_package_uses_current_target() {
        // When a symbol in a binary references another symbol in the same package,
        // it should use the current target (bin/foo), not lib.
        let mapping = make_mapping(&[("my_bin", "my-package")]);

        let result = transform_path(
            "my_bin::SomeStruct",
            &mapping,
            "my-package", // current package
            "bin/my_bin", // current target
        );

        assert_eq!(result, "[my-package/bin/my_bin]::SomeStruct");
    }

    #[test]
    fn test_transform_path_cross_package_uses_lib() {
        // When a symbol references another package, it should use lib target.
        let mapping = make_mapping(&[
            ("my_bin", "my-package"),
            ("other_crate", "other-package"),
        ]);

        let result = transform_path(
            "other_crate::SomeStruct",
            &mapping,
            "my-package", // current package
            "bin/my_bin", // current target
        );

        assert_eq!(result, "[other-package/lib]::SomeStruct");
    }

    #[test]
    fn test_transform_path_external_crate_unchanged() {
        // External crates (not in mapping) should be returned unchanged.
        let mapping = make_mapping(&[("my_crate", "my-package")]);

        let result =
            transform_path("serde::Serialize", &mapping, "my-package", "lib");

        assert_eq!(result, "serde::Serialize");
    }

    #[test]
    fn test_transform_path_no_colons_unchanged() {
        // Paths without `::` (just crate name) should be returned unchanged.
        let mapping = make_mapping(&[("my_crate", "my-package")]);

        let result = transform_path("std", &mapping, "my-package", "lib");

        assert_eq!(result, "std");
    }

    #[test]
    fn test_transform_path_test_target() {
        // Test target referencing its own symbols should use test target.
        let mapping = make_mapping(&[("my_crate", "my-package")]);

        let result = transform_path(
            "my_crate::tests::helper",
            &mapping,
            "my-package",
            "test",
        );

        assert_eq!(result, "[my-package/test]::tests::helper");
    }

    #[test]
    fn test_transform_path_nested_module() {
        // Nested module paths should be preserved.
        let mapping = make_mapping(&[("my_crate", "my-package")]);

        let result = transform_path(
            "my_crate::foo::bar::Baz",
            &mapping,
            "my-package",
            "lib",
        );

        assert_eq!(result, "[my-package/lib]::foo::bar::Baz");
    }

    #[test]
    fn test_transform_path_hyphenated_package_name() {
        // Package names with hyphens should be preserved in output.
        let mapping = make_mapping(&[("my_crate", "my-hyphenated-package")]);

        let result = transform_path(
            "my_crate::Item",
            &mapping,
            "my-hyphenated-package",
            "lib",
        );

        assert_eq!(result, "[my-hyphenated-package/lib]::Item");
    }
}
