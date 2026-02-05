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
    ENV_OUTPUT_DIR, ENV_PROFILE_DIR, ENV_WORKSPACE_CRATES,
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
    // Cargo package names use hyphens (e.g., "my-crate") but rustc's --crate-name
    // uses underscores (e.g., "my_crate"). Normalize for comparison.
    let is_workspace_crate = config
        .workspace_crates
        .iter()
        .any(|pkg_name| pkg_name.replace('-', "_") == crate_name);

    if !is_workspace_crate {
        // External dependency - compile without extraction.
        // IMPORTANT: We use rustc_driver instead of calling the external rustc binary
        // to ensure all crates are compiled with the same compiler version.
        // Mixing compiler versions causes "compiled by incompatible rustc" errors.
        trace!(
            crate_name,
            "not a workspace crate, compiling without extraction"
        );
        return run_via_rustc_driver(args, &mut NoOpCallbacks);
    }

    // Determine target kind from rustc args.
    let target_key = determine_target_key(args, &crate_name);
    let is_test = args.iter().any(|a| a == "--test");

    debug!(
        crate_name,
        target_key, is_test, "extracting symbols from workspace crate"
    );

    // This is a workspace crate - run with our extraction callbacks.
    let mut callbacks = TarjanizeCallbacks {
        config,
        crate_name,
        is_test,
        extracted_module: Mutex::new(None),
    };

    // Inject `-Z no-steal-thir` to preserve THIR for body analysis.
    // By default, THIR is "stolen" (deallocated) when MIR is built to save memory.
    // Since `after_analysis` runs after MIR is built, we need this flag to keep
    // THIR available for our extraction.
    let mut compiler_args: Vec<String> = args.to_vec();
    compiler_args.push("-Zno-steal-thir".to_string());

    // Add self-profiling flags to collect compilation timing data.
    // We need `default` for core events, `llvm` for backend costs per CGU,
    // and `args` for DefPath associations.
    //
    // Each target (lib, test, bin) gets its own profile subdirectory so they can
    // clean up independently without interfering with concurrent targets.
    // Use underscore-separated key for filesystem safety (bin/foo → bin_foo).
    let target_profile_dir = callbacks.config.profile_dir.join(format!(
        "{}_{}",
        callbacks.crate_name,
        target_key.replace('/', "_")
    ));
    fs::create_dir_all(&target_profile_dir)
        .expect("failed to create profile subdirectory");
    compiler_args
        .push(format!("-Zself-profile={}", target_profile_dir.display()));
    compiler_args.push("-Zself-profile-events=default,llvm,args".to_string());

    // Print mono-items for backend cost distribution.
    // This tells us which symbols are codegen'd into which CGU.
    // We redirect stdout to a file because cargo swallows rustc's stdout.
    compiler_args.push("-Zprint-mono-items=yes".to_string());

    // Capture mono-items output by redirecting stdout to a file.
    // Rustc's `-Zprint-mono-items` outputs to stdout, which cargo swallows.
    // We redirect to a file that the orchestrator reads later.
    // Use append mode since multiple targets (lib, bin, test) may compile.
    let mono_items_path = callbacks
        .config
        .output_dir
        .join(format!("{}_mono_items.txt", callbacks.crate_name));

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

    // After rustc completes (including codegen), process profile data to apply
    // costs to the extracted module, write the final Crate JSON, and delete
    // the profile directory for this target.
    if let Some(module) = callbacks.extracted_module.lock().unwrap().take() {
        process_and_write_crate(
            &callbacks.config,
            &callbacks.crate_name,
            module,
            &target_key,
            &target_profile_dir,
        );
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
    /// Directory for self-profile output.
    profile_dir: PathBuf,
}

impl DriverConfig {
    fn from_env() -> Option<Self> {
        let output_dir: PathBuf = env::var(ENV_OUTPUT_DIR).ok()?.into();
        let workspace_crates: Vec<String> = env::var(ENV_WORKSPACE_CRATES)
            .ok()?
            .split(',')
            .map(String::from)
            .collect();
        let profile_dir: PathBuf = env::var(ENV_PROFILE_DIR).ok()?.into();

        trace!(
            output_dir = %output_dir.display(),
            workspace_crates = ?workspace_crates,
            profile_dir = %profile_dir.display(),
            "loaded driver config from env"
        );

        Some(Self {
            output_dir,
            workspace_crates,
            profile_dir,
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

/// Determine the target key from rustc arguments.
///
/// Returns a target key like "lib", "test", "bin/{name}".
/// This is a simplified version - the full implementation will use the
/// mapping file from the orchestrator.
fn determine_target_key(args: &[String], crate_name: &str) -> String {
    let is_test = args.iter().any(|a| a == "--test");
    let is_bin = args.iter().any(|a| a == "--crate-type=bin")
        || args
            .windows(2)
            .any(|w| w[0] == "--crate-type" && w[1] == "bin");

    if is_test {
        "test".to_string()
    } else if is_bin {
        format!("bin/{crate_name}")
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
    crate_name: String,
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

/// Process profile data, apply costs to the module, and write the final Crate.
///
/// This runs after rustc completes (including codegen) so we have access to:
/// - Mono-items output (which symbols are in which CGU)
/// - Self-profile data (timing for frontend, backend, and overhead)
///
/// After writing the Crate JSON, deletes the profile directory for this target
/// to avoid accumulating huge amounts of temp data on large workspaces.
fn process_and_write_crate(
    config: &DriverConfig,
    crate_name: &str,
    mut module: Module,
    target_key: &str,
    profile_dir: &Path,
) {
    // Load mono-items for backend cost distribution.
    let mono_items_path = config
        .output_dir
        .join(format!("{crate_name}_mono_items.txt"));
    let mono_items = if mono_items_path.exists() {
        match fs::read_to_string(&mono_items_path) {
            Ok(content) => {
                let map = MonoItemsMap::parse(content.as_bytes(), crate_name);
                debug!(
                    crate_name,
                    cgu_count = map.cgu_to_items.len(),
                    "loaded mono-items"
                );
                Some(map)
            }
            Err(e) => {
                warn!(crate_name, error = %e, "failed to read mono-items");
                None
            }
        }
    } else {
        debug!(crate_name, "no mono-items file");
        None
    };

    // Load profile data from this target's dedicated profile directory.
    let profile_data = ProfileData::load_from_dir(profile_dir);

    // Distribute backend costs from CGU timing to individual symbols.
    let backend_costs = if let Some(ref mono) = mono_items {
        distribute_backend_costs(&profile_data, mono)
    } else {
        HashMap::new()
    };

    // Apply costs to the module.
    apply_frontend_costs(&mut module, crate_name, &profile_data);
    apply_backend_costs(&mut module, crate_name, &backend_costs);

    // Get crate overhead.
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
    let safe_target_key = target_key.replace('/', "_");
    let filename = format!("{crate_name}_{safe_target_key}.json");
    let output_path = config.output_dir.join(filename);

    let result = CrateResult {
        crate_name: crate_name.to_string(),
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
