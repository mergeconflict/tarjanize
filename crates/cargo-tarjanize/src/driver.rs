//! Driver mode: Custom rustc driver for symbol extraction.
//!
//! When invoked as `RUSTC_WRAPPER`, this module:
//! 1. Checks if the crate being compiled is a workspace member
//! 2. For workspace crates: runs rustc with our callbacks to extract symbols
//! 3. For external crates: compiles without extraction
//!
//! All crates are compiled via `rustc_driver` to ensure consistent compiler
//! version. The extraction happens in the `after_analysis` callback, which
//! gives us access to the fully type-checked HIR and THIR.
//!
//! After rustc completes (including codegen), the driver processes profile
//! data and mono-items to compute costs, assembles a complete `Crate`, and
//! deletes the raw profile files to avoid disk space exhaustion.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::{env, fs};

use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use tarjanize_schemas::{Crate, Module, SymbolKind};
use tracing::{debug, info, trace, warn};

use crate::extract;
use crate::mono_items::MonoItemsMap;
use crate::orchestrator::{
    ENV_OUTPUT_DIR, ENV_PROFILE_DIR, ENV_SKIP_PROFILE, ENV_WORKSPACE_CRATES,
    ENV_WORKSPACE_PATHS,
};
use crate::profile::ProfileData;

/// Run the driver: act as a rustc wrapper.
///
/// Arguments should be `["/path/to/rustc", ...actual_rustc_args]`.
pub fn run(args: &[String]) -> ExitCode {
    // Parse environment to determine if we should extract.
    // Environment not set up means just run rustc normally.
    // This can happen during initial cargo probing.
    let Some(config) = DriverConfig::from_env() else {
        trace!("no tarjanize config, passing through to rustc");
        return run_via_rustc_driver(args, &mut NoOpCallbacks);
    };

    // Determine the crate name from rustc arguments.
    // Can't determine crate name means pass through to rustc.
    let Some(crate_name) = find_crate_name(args) else {
        trace!("no crate name in args, passing through to rustc");
        return run_via_rustc_driver(args, &mut NoOpCallbacks);
    };

    // Check if this is a workspace crate we should analyze.
    // There are two cases:
    // 1. Direct match: crate_name matches a workspace package (e.g., "tokio")
    // 2. Integration test: crate_name is the test file name (e.g., "sync_mpsc")
    //    but the source file is within a workspace package directory
    //
    // Cargo package names use hyphens (e.g., "my-crate") but rustc's --crate-name
    // uses underscores (e.g., "my_crate"). Normalize for comparison.
    let direct_match = config
        .workspace_crates
        .iter()
        .find(|pkg_name| pkg_name.replace('-', "_") == crate_name)
        .cloned();

    // If no direct match, check if this is an integration test by looking at
    // the source file path. Integration tests have crate names like "sync_mpsc"
    // but their source is in `tokio/tests/sync_mpsc.rs`.
    let (package_name, is_integration_test) = if let Some(pkg) = direct_match {
        (pkg, false)
    } else if let Some(source_file) = find_source_file(args) {
        if let Some(pkg) =
            find_package_for_source(&config.workspace_paths, &source_file)
        {
            (pkg, true)
        } else {
            // Source file not in any workspace package - external crate.
            trace!(
                crate_name,
                source_file = %source_file.display(),
                "source not in workspace, compiling without extraction"
            );
            return run_via_rustc_driver(args, &mut NoOpCallbacks);
        }
    } else {
        // No direct match and can't find source file - external crate.
        trace!(
            crate_name,
            "not a workspace crate, compiling without extraction"
        );
        return run_via_rustc_driver(args, &mut NoOpCallbacks);
    };

    // Determine target kind from rustc args.
    let is_test = args.iter().any(|a| a == "--test");
    let target_key = determine_target_key(
        args,
        &crate_name,
        &package_name,
        is_integration_test,
    );

    debug!(
        crate_name,
        package_name,
        target_key,
        is_test,
        is_integration_test,
        "extracting symbols from workspace crate"
    );

    // This is a workspace crate - run with our extraction callbacks.
    let mut callbacks = TarjanizeCallbacks {
        config,
        crate_name,
        package_name,
        target_key: target_key.clone(),
        is_test,
        extracted_module: Mutex::new(None),
    };

    // Inject `-Z no-steal-thir` to preserve THIR for body analysis.
    // By default, THIR is "stolen" (deallocated) when MIR is built to save memory.
    // Since `after_analysis` runs after MIR is built, we need this flag to keep
    // THIR available for our extraction.
    let mut compiler_args: Vec<String> = args.to_vec();
    compiler_args.push("-Zno-steal-thir".to_string());

    // Only add self-profiling flags when profiling is enabled.
    // In two-pass mode or --no-profile mode (skip_profile=true), profiling is
    // either done in Pass 1 without RUSTC_WRAPPER, or skipped entirely.
    let target_profile_dir = if callbacks.config.skip_profile {
        None
    } else {
        // Add self-profiling flags to collect compilation timing data.
        // We need `default` for core events, `llvm` for backend costs per CGU,
        // and `args` for DefPath associations.
        //
        // Each target (lib, test, bin) gets its own profile subdirectory so they can
        // clean up independently without interfering with concurrent targets.
        // Use underscore-separated key for filesystem safety (bin/foo → bin_foo).
        let dir = callbacks.config.profile_dir.join(format!(
            "{}_{}",
            callbacks.package_name,
            callbacks.target_key.replace('/', "_")
        ));
        fs::create_dir_all(&dir)
            .expect("failed to create profile subdirectory");
        compiler_args.push(format!("-Zself-profile={}", dir.display()));
        compiler_args
            .push("-Zself-profile-events=default,llvm,args".to_string());
        Some(dir)
    };

    // Print mono-items for backend cost distribution.
    // This tells us which symbols are codegen'd into which CGU.
    // We redirect stdout to a file because cargo swallows rustc's stdout.
    compiler_args.push("-Zprint-mono-items=yes".to_string());

    // Capture mono-items output by redirecting stdout to a file.
    // Rustc's `-Zprint-mono-items` outputs to stdout, which cargo swallows.
    // We redirect to a file that the orchestrator reads later.
    // Use append mode since multiple targets (lib, bin, test) may compile.
    // Use package_name so all targets for a package share the same file.
    let mono_items_path = callbacks
        .config
        .output_dir
        .join(format!("{}_mono_items.txt", callbacks.package_name));

    // Redirect stdout to the mono-items file during compilation.
    // The gag crate provides safe file descriptor redirection.
    // When `_redirect` is dropped, stdout is automatically restored.
    let mono_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&mono_items_path)
        .expect("failed to open mono-items file");
    let _redirect =
        gag::Redirect::stdout(mono_file).expect("failed to redirect stdout");

    let exit_code = run_via_rustc_driver(&compiler_args, &mut callbacks);

    // After rustc completes (including codegen), write the results.
    // In single-pass mode: apply costs from profile data, then delete profile files.
    // In two-pass mode: just write the raw module (orchestrator applies costs later).
    if let Some(module) = callbacks.extracted_module.lock().unwrap().take() {
        if callbacks.config.skip_profile {
            // Two-pass mode: write module without costs.
            // Orchestrator will apply costs from Pass 1's profile data.
            write_crate_without_costs(
                &callbacks.config,
                &callbacks.package_name,
                module,
                &callbacks.target_key,
            );
        } else {
            // Single-pass mode: apply costs and delete profile files.
            process_and_write_crate(
                &callbacks.config,
                &callbacks.crate_name,
                &callbacks.package_name,
                module,
                &callbacks.target_key,
                target_profile_dir
                    .as_ref()
                    .expect("profile dir required in single-pass mode"),
            );
        }
    }

    exit_code
}

/// Run the compiler via `rustc_driver` with the given callbacks.
///
/// This ensures all crates are compiled with the same compiler version,
/// avoiding "compiled by incompatible rustc" errors that occur when mixing
/// different compiler versions.
fn run_via_rustc_driver(
    args: &[String],
    callbacks: &mut (dyn Callbacks + Send),
) -> ExitCode {
    // rustc_driver::catch_fatal_errors handles ICEs gracefully.
    let result = rustc_driver::catch_fatal_errors(|| {
        run_compiler(args, callbacks);
    });

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => ExitCode::FAILURE,
    }
}

/// No-op callbacks for compiling without extraction.
struct NoOpCallbacks;

impl Callbacks for NoOpCallbacks {}

/// Configuration parsed from environment variables.
struct DriverConfig {
    output_dir: PathBuf,
    workspace_crates: Vec<String>,
    /// Maps package name to its manifest directory path.
    /// Used to identify which package integration tests belong to.
    workspace_paths: HashMap<String, PathBuf>,
    /// Directory for self-profile output.
    /// Only used when `skip_profile` is false.
    profile_dir: PathBuf,
    /// Whether to skip profiling (--no-profile mode).
    /// When true, the driver does not add `-Zself-profile` flags
    /// and does not apply costs. Useful for faster iteration when
    /// profiling data is not needed.
    skip_profile: bool,
}

impl DriverConfig {
    fn from_env() -> Option<Self> {
        let output_dir: PathBuf = env::var(ENV_OUTPUT_DIR).ok()?.into();
        let workspace_crates: Vec<String> = env::var(ENV_WORKSPACE_CRATES)
            .ok()?
            .split(',')
            .map(String::from)
            .collect();

        // Parse workspace paths: "pkg1=/path1,pkg2=/path2"
        let workspace_paths: HashMap<String, PathBuf> =
            env::var(ENV_WORKSPACE_PATHS)
                .unwrap_or_default()
                .split(',')
                .filter_map(|entry| {
                    let (name, path) = entry.split_once('=')?;
                    Some((name.to_string(), PathBuf::from(path)))
                })
                .collect();

        // Profile dir is optional when skip_profile is true.
        let profile_dir: PathBuf = env::var(ENV_PROFILE_DIR)
            .map(PathBuf::from)
            .unwrap_or_default();
        let skip_profile = env::var(ENV_SKIP_PROFILE).is_ok_and(|v| v == "1");

        trace!(
            output_dir = %output_dir.display(),
            workspace_crates = ?workspace_crates,
            workspace_paths_count = workspace_paths.len(),
            profile_dir = %profile_dir.display(),
            skip_profile,
            "loaded driver config from env"
        );

        Some(Self {
            output_dir,
            workspace_crates,
            workspace_paths,
            profile_dir,
            skip_profile,
        })
    }
}

/// Find the crate name from rustc arguments.
/// Looks for `--crate-name <name>`.
fn find_crate_name(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--crate-name" {
            return iter.next().cloned();
        }
    }
    None
}

/// Find the source file path from rustc arguments.
///
/// The source file is typically the last `.rs` argument that isn't preceded
/// by a flag that takes a path argument.
fn find_source_file(args: &[String]) -> Option<PathBuf> {
    // Look for the source file - it's usually a .rs file not preceded by a flag.
    // Skip known flags that take path arguments.
    let path_flags = ["--out-dir", "-L", "--extern", "--emit", "-o"];

    let mut skip_next = false;
    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if path_flags.iter().any(|f| arg == *f) {
            skip_next = true;
            continue;
        }
        if std::path::Path::new(arg)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
            && !arg.starts_with('-')
        {
            return Some(PathBuf::from(arg));
        }
    }
    None
}

/// Find which workspace package a source file belongs to.
///
/// Returns the package name if the source file is within one of the workspace
/// member directories.
fn find_package_for_source(
    workspace_paths: &HashMap<String, PathBuf>,
    source_file: &Path,
) -> Option<String> {
    // Canonicalize the source file path for reliable comparison.
    let source_canonical = source_file.canonicalize().ok()?;

    for (pkg_name, pkg_path) in workspace_paths {
        if let Ok(pkg_canonical) = pkg_path.canonicalize()
            && source_canonical.starts_with(&pkg_canonical)
        {
            return Some(pkg_name.clone());
        }
    }
    None
}

/// Determine the target key from rustc arguments.
///
/// Returns a target key like:
/// - "lib" for the main library
/// - "test" for lib compiled with --test (unit tests)
/// - "test/{name}" for integration tests (tests/*.rs files)
/// - "bin/{name}" for binaries
/// - "example/{name}" for examples
/// - "bench/{name}" for benchmarks
///
/// The `is_integration_test` flag indicates the `crate_name` doesn't match the
/// package name, meaning this is an integration test, example, or bench.
fn determine_target_key(
    args: &[String],
    crate_name: &str,
    package_name: &str,
    is_integration_test: bool,
) -> String {
    let is_test = args.iter().any(|a| a == "--test");
    let is_bin = args.iter().any(|a| a == "--crate-type=bin")
        || args
            .windows(2)
            .any(|w| w[0] == "--crate-type" && w[1] == "bin");

    // For integration tests/examples/benches, we need to distinguish them.
    // Look at the source file path to determine the type.
    if is_integration_test {
        if let Some(source) = find_source_file(args) {
            let source_str = source.to_string_lossy();
            trace!(
                crate_name,
                source = %source_str,
                "determining target key from source path"
            );
            // Check for standard target directories. Handle both:
            // - Absolute/workspace-relative paths: tokio/tests/foo.rs (contains /tests/)
            // - Package-relative paths: tests/foo.rs (starts with tests/)
            if source_str.contains("/tests/")
                || source_str.contains("\\tests\\")
                || source_str.starts_with("tests/")
            {
                return format!("test/{crate_name}");
            } else if source_str.contains("/examples/")
                || source_str.contains("\\examples\\")
                || source_str.starts_with("examples/")
            {
                return format!("example/{crate_name}");
            } else if source_str.contains("/benches/")
                || source_str.contains("\\benches\\")
                || source_str.starts_with("benches/")
            {
                return format!("bench/{crate_name}");
            }
        } else {
            trace!(crate_name, "no source file found in args");
        }
        // Fallback: if it's a test target, call it a test
        if is_test {
            return format!("test/{crate_name}");
        }
    }

    // Standard targets where crate_name matches package_name
    if is_test {
        // This is the lib compiled with --test for unit tests
        "test".to_string()
    } else if is_bin {
        // Check if this is the "main" binary (same name as package) or a named binary
        let pkg_crate_name = package_name.replace('-', "_");
        if crate_name == pkg_crate_name {
            "bin".to_string()
        } else {
            format!("bin/{crate_name}")
        }
    } else {
        "lib".to_string()
    }
}

/// Callbacks implementation for symbol extraction.
///
/// The extracted module is stored in `extracted_module` for post-compilation
/// processing. We use a Mutex because `rustc_driver` requires `Callbacks + Send`.
struct TarjanizeCallbacks {
    config: DriverConfig,
    /// The rustc crate name (e.g., `sync_mpsc` for an integration test).
    crate_name: String,
    /// The cargo package name this target belongs to (e.g., "tokio").
    /// For lib/bin targets, this equals `crate_name` (with underscores).
    /// For integration tests, this is the parent package.
    package_name: String,
    /// The target key (e.g., "lib", "test", `test/sync_mpsc`).
    target_key: String,
    /// Whether this is a test target (compiled with `--test`).
    /// When true, only `#[cfg(test)]` items are extracted to avoid
    /// duplicating symbols from the lib target.
    is_test: bool,
    /// The extracted module, populated by `after_analysis`.
    /// Stored here so we can apply costs after rustc completes.
    extracted_module: Mutex<Option<Module>>,
}

impl Callbacks for TarjanizeCallbacks {
    /// Called after type checking is complete.
    ///
    /// Extracts symbols and stores them for post-compilation processing.
    /// We can't apply costs here because codegen hasn't happened yet -
    /// profile data for backend costs isn't available until after codegen.
    fn after_analysis(
        &mut self,
        _compiler: &Compiler,
        tcx: TyCtxt<'_>,
    ) -> Compilation {
        // Extract symbols from this crate.
        // For test targets (is_test=true), only extract #[cfg(test)] items
        // to avoid duplicating symbols from the lib target.
        info!(crate_name = %self.crate_name, is_test = self.is_test, "extracting symbols");
        let extraction = extract::extract_crate(
            tcx,
            &self.crate_name,
            &self.config.workspace_crates,
            self.is_test,
        );

        let symbol_count = count_symbols(&extraction.module);
        debug!(crate_name = %self.crate_name, symbol_count, "extraction complete");

        // Store the module for post-compilation processing.
        // We'll apply costs and write the final Crate JSON after rustc
        // completes (including codegen).
        *self.extracted_module.lock().unwrap() = Some(extraction.module);

        // Continue compilation so cargo can use the artifacts.
        Compilation::Continue
    }
}

/// Count total symbols in a module tree (for logging).
fn count_symbols(module: &Module) -> usize {
    let direct = module.symbols.len();
    let nested: usize = module.submodules.values().map(count_symbols).sum();
    direct + nested
}

/// Write extracted module without applying costs (two-pass mode).
///
/// In two-pass mode, the orchestrator applies costs after both passes complete.
/// This function just writes the raw extraction results.
///
/// The `package_name` is the cargo package this target belongs to (e.g., "tokio").
/// For integration tests, this differs from the rustc crate name (e.g., `sync_mpsc`).
fn write_crate_without_costs(
    config: &DriverConfig,
    package_name: &str,
    module: Module,
    target_key: &str,
) {
    // Build the Crate without costs.
    let crate_data = Crate {
        root: module,
        ..Default::default()
    };

    // Write to JSON file.
    // Filename uses package_name so all targets for a package are grouped together.
    let safe_target_key = target_key.replace('/', "_");
    let filename = format!("{package_name}_{safe_target_key}.json");
    let output_path = config.output_dir.join(filename);

    let result = CrateResult {
        crate_name: package_name.to_string(),
        crate_data,
    };

    let json =
        serde_json::to_string_pretty(&result).expect("failed to serialize");
    fs::write(&output_path, &json).expect("failed to write output file");

    debug!(
        path = %output_path.display(),
        bytes = json.len(),
        "wrote crate (costs deferred to orchestrator)"
    );
}

/// Process profile data, apply costs to the module, and write the final Crate.
///
/// This runs after rustc completes (including codegen) so we have access to:
/// - Mono-items output (which symbols are in which CGU)
/// - Self-profile data (timing for frontend, backend, and overhead)
///
/// After writing the Crate JSON, deletes the profile directory for this target
/// to avoid accumulating huge amounts of temp data on large workspaces.
///
/// The `crate_name` is the rustc crate name (e.g., `sync_mpsc` for integration tests).
/// The `package_name` is the cargo package this target belongs to (e.g., "tokio").
/// For lib targets, these are the same (modulo hyphen/underscore normalization).
/// For integration tests/benches/examples, they differ.
fn process_and_write_crate(
    config: &DriverConfig,
    crate_name: &str,
    package_name: &str,
    mut module: Module,
    target_key: &str,
    profile_dir: &Path,
) {
    let _span = tracing::info_span!(
        "process_and_write_crate",
        crate_name,
        package_name,
        target_key,
        profile_dir = %profile_dir.display(),
    )
    .entered();

    // Load mono-items for backend cost distribution.
    // Uses package_name because mono-items are written per-package.
    let mono_items_path = config
        .output_dir
        .join(format!("{package_name}_mono_items.txt"));
    let mono_items = if mono_items_path.exists() {
        match fs::read_to_string(&mono_items_path) {
            Ok(content) => {
                let map = MonoItemsMap::parse(content.as_bytes(), package_name);
                debug!(
                    package_name,
                    cgu_count = map.cgu_to_items.len(),
                    "loaded mono-items"
                );
                Some(map)
            }
            Err(e) => {
                warn!(package_name, error = %e, "failed to read mono-items");
                None
            }
        }
    } else {
        debug!(package_name, "no mono-items file");
        None
    };

    // Load profile data from this target's dedicated profile directory.
    let profile_data = {
        let _span = tracing::info_span!(
            "load_profile",
            package = %package_name,
            target = %target_key,
            crate_name = %crate_name,
        )
        .entered();
        ProfileData::load_from_dir(profile_dir)
    };

    // Distribute backend costs from CGU timing to individual symbols.
    let backend_costs = if let Some(ref mono) = mono_items {
        distribute_backend_costs(&profile_data, mono)
    } else {
        HashMap::new()
    };

    // Apply costs to the module.
    // Use crate_name for profile lookups since profile data is keyed by rustc
    // crate name (e.g., "sync_mpsc"), not cargo package name (e.g., "tokio").
    apply_frontend_costs(&mut module, crate_name, &profile_data);
    apply_backend_costs(&mut module, crate_name, &backend_costs);

    // Get crate overhead.
    // Also uses crate_name since profile overhead is keyed by rustc crate name.
    let overhead = profile_data
        .get_crate_overhead(crate_name)
        .cloned()
        .unwrap_or_default();

    // Build the complete Crate.
    // Dependencies are populated later by the orchestrator from cargo metadata.
    let crate_data = Crate {
        metadata_ms: overhead.metadata_ms,
        root: module,
        ..Default::default()
    };

    // Write to JSON file.
    // Replace / with _ for filesystem safety (bin/foo → bin_foo).
    // Filename uses package_name so all targets for a package are grouped together.
    let safe_target_key = target_key.replace('/', "_");
    let filename = format!("{package_name}_{safe_target_key}.json");
    let output_path = config.output_dir.join(filename);

    let result = CrateResult {
        crate_name: package_name.to_string(),
        crate_data,
    };

    let json =
        serde_json::to_string_pretty(&result).expect("failed to serialize");
    fs::write(&output_path, &json).expect("failed to write output file");

    debug!(
        path = %output_path.display(),
        bytes = json.len(),
        backend_symbols = backend_costs.len(),
        "wrote crate with costs"
    );

    // Delete this target's profile directory to free disk space.
    // Each target has its own directory, so this won't affect other targets.
    if let Err(e) = fs::remove_dir_all(profile_dir) {
        warn!(
            path = %profile_dir.display(),
            error = %e,
            "failed to delete profile directory"
        );
    } else {
        debug!(path = %profile_dir.display(), "deleted profile directory");
    }
}

/// Distribute backend costs from CGU timing to individual symbols.
///
/// For each CGU, we have:
/// - Total CGU cost from profile data
/// - List of symbols in that CGU from mono-items
///
/// We distribute the CGU cost equally among its symbols.
fn distribute_backend_costs(
    profile_data: &ProfileData,
    mono_items: &MonoItemsMap,
) -> HashMap<String, f64> {
    let mut costs: HashMap<String, f64> = HashMap::new();

    let cgu_costs = profile_data.cgu_costs();

    for (cgu_name, items) in &mono_items.cgu_to_items {
        let Some(cgu_duration) = cgu_costs.get(cgu_name) else {
            debug!(cgu_name, "CGU not found in profile data");
            continue;
        };

        if items.is_empty() {
            continue;
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "CGU item counts are small, precision loss is negligible"
        )]
        let cost_per_item = cgu_duration.as_millis_f64() / items.len() as f64;

        for item in items {
            *costs.entry(item.clone()).or_default() += cost_per_item;
        }
    }

    costs
}

/// Apply frontend costs from profile data to symbols.
fn apply_frontend_costs(
    module: &mut Module,
    path_prefix: &str,
    profile_data: &ProfileData,
) {
    for (name, symbol) in &mut module.symbols {
        let full_path = format!("{path_prefix}::{name}");
        let cost = profile_data.get_frontend_cost_ms(&full_path).unwrap_or(0.0);
        symbol.frontend_cost_ms = cost;
    }

    for (submod_name, submodule) in &mut module.submodules {
        let submod_path = format!("{path_prefix}::{submod_name}");
        apply_frontend_costs(submodule, &submod_path, profile_data);
    }
}

/// Apply backend costs from CGU distribution to symbols.
fn apply_backend_costs(
    module: &mut Module,
    path_prefix: &str,
    backend_costs: &HashMap<String, f64>,
) {
    for (name, symbol) in &mut module.symbols {
        let full_path = format!("{path_prefix}::{name}");
        let mut cost = backend_costs.get(&full_path).copied().unwrap_or(0.0);

        // For impl blocks, sum all costs from paths starting with any anchor.
        if let SymbolKind::Impl { anchors, .. } = &symbol.kind {
            for anchor in anchors {
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

/// Result written by the driver for each crate/target.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CrateResult {
    pub crate_name: String,
    #[serde(flatten)]
    pub crate_data: Crate,
}
