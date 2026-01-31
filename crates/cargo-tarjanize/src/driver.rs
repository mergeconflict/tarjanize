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

use std::path::PathBuf;
use std::process::ExitCode;
use std::{env, fs};

use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use tarjanize_schemas::Module;
use tracing::{debug, info, trace};

use crate::extract;
use crate::orchestrator::{
    ENV_OUTPUT_DIR, ENV_PROFILE_DIR, ENV_WORKSPACE_CRATES,
};

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

    // Determine if this is a test target.
    let is_test = args.iter().any(|a| a == "--test");

    debug!(
        crate_name,
        is_test, "extracting symbols from workspace crate"
    );

    // This is a workspace crate - run with our extraction callbacks.
    let mut callbacks = TarjanizeCallbacks {
        config,
        crate_name,
        is_test,
    };

    // Inject `-Z no-steal-thir` to preserve THIR for body analysis.
    // By default, THIR is "stolen" (deallocated) when MIR is built to save memory.
    // Since `after_analysis` runs after MIR is built, we need this flag to keep
    // THIR available for our extraction.
    let mut compiler_args: Vec<String> = args.to_vec();
    compiler_args.push("-Zno-steal-thir".to_string());

    // Add self-profiling flags to collect compilation timing data.
    compiler_args.push(format!(
        "-Zself-profile={}",
        callbacks.config.profile_dir.display()
    ));
    compiler_args.push("-Zself-profile-events=default,args".to_string());

    run_via_rustc_driver(&compiler_args, &mut callbacks)
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

/// Callbacks implementation for symbol extraction.
struct TarjanizeCallbacks {
    config: DriverConfig,
    crate_name: String,
    is_test: bool,
}

impl Callbacks for TarjanizeCallbacks {
    /// Called after type checking is complete.
    /// This is where we have full access to HIR and THIR for extraction.
    fn after_analysis(
        &mut self,
        _compiler: &Compiler,
        tcx: TyCtxt<'_>,
    ) -> Compilation {
        // Extract symbols from this crate.
        info!(crate_name = %self.crate_name, "extracting symbols");
        let extraction = extract::extract_crate(
            tcx,
            &self.crate_name,
            &self.config.workspace_crates,
        );

        let symbol_count = count_symbols(&extraction.module);
        debug!(crate_name = %self.crate_name, symbol_count, "extraction complete");

        // Write the result to a JSON file.
        // Filename includes crate name and target type to avoid collisions.
        let target_suffix = if self.is_test { "test" } else { "lib" };
        let filename = format!("{}_{}.json", self.crate_name, target_suffix);
        let output_path = self.config.output_dir.join(filename);

        let result = PartialResult {
            crate_name: self.crate_name.clone(),
            module: extraction.module,
        };

        let json =
            serde_json::to_string_pretty(&result).expect("failed to serialize");
        fs::write(&output_path, &json).expect("failed to write output file");
        debug!(path = %output_path.display(), bytes = json.len(), "wrote extraction results");

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

/// Intermediate result written for each crate/target.
#[derive(serde::Serialize)]
struct PartialResult {
    crate_name: String,
    module: Module,
}
