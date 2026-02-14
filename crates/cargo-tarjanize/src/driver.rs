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
//! After rustc completes, the driver processes profile data to compute
//! per-event timing breakdowns, assembles a complete `Target`, and deletes
//! the raw profile files to avoid disk space exhaustion.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::Duration;
use std::{env, fs, io};

use anyhow::{Context, Result};
use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use tarjanize_schemas::{Module, Target, TargetId};
use tracing::{debug, trace, warn};

use crate::extract;
use crate::orchestrator::{
    ENV_OUTPUT_DIR, ENV_PROFILE_DIR, ENV_WORKSPACE_CRATES, ENV_WORKSPACE_PATHS,
};
use crate::profile_processing::{
    append_unmatched_with_span, apply_event_times_with_span,
    collect_profile_paths, load_profile_data_with_span,
    roll_up_unmatched_with_span,
};

/// Identifies a compilation target during extraction.
///
/// Bundles the three pieces of identity needed by profile-processing and
/// extraction functions: the rustc crate name, and the package/target
/// coordinate (`TargetId`).
///
/// `crate_name` is the rustc `--crate-name` value (e.g., `sync_mpsc`).
/// It diverges from `package_name` for integration tests, examples, and
/// benches, so it cannot be derived from `TargetId` alone.
///
/// Why: eliminates the 3-parameter data clump (`crate_name`, `package_name`,
/// `target_key`) passed through 8 functions in the extraction pipeline.
pub(crate) struct CrateIdentity {
    crate_name: String,
    target_id: TargetId,
}

impl CrateIdentity {
    /// Returns the rustc crate name.
    pub(crate) fn crate_name(&self) -> &str {
        &self.crate_name
    }

    /// Returns the cargo package name (from the target id).
    pub(crate) fn package_name(&self) -> &str {
        self.target_id.package()
    }

    /// Returns the target key (e.g., "lib", "test", `"test/sync_mpsc"`).
    pub(crate) fn target_key(&self) -> &str {
        self.target_id.target()
    }
}

/// Path in rustc's self-profile format: `crate_name::module::symbol`.
///
/// Used exclusively for matching against `-Zself-profile` output. Differs
/// from `ModulePath` (no crate prefix) and qualified symbol paths (which use
/// `[pkg/target]::` prefix instead of crate name).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ProfilePath(pub(crate) String);

impl ProfilePath {
    /// Creates the root profile path from a crate name.
    pub(crate) fn from_crate(crate_name: &str) -> Self {
        Self(crate_name.to_owned())
    }

    /// Appends a `::segment` to produce a child path.
    #[must_use]
    pub(crate) fn child(&self, segment: &str) -> Self {
        Self(format!("{}::{segment}", self.0))
    }

    /// Returns the inner string slice.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ProfilePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What kind of compilation target this is.
///
/// Determines the target key format in the `package/target` identifier.
/// Replaces the raw `is_integration_test: bool` flag which was misnamed
/// (it also covered examples and benches) and lost information about
/// which kind of named target was being built.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TargetKind {
    /// Library target (`"lib"`).
    Lib,
    /// Binary target (`"bin/{name}"`).
    Bin(String),
    /// Unit test target — lib compiled with `--test` (`"test"`).
    UnitTest,
    /// Integration test (`"test/{name}"`).
    IntegrationTest(String),
    /// Example target (`"example/{name}"`).
    Example(String),
    /// Benchmark target (`"bench/{name}"`).
    Bench(String),
}

impl TargetKind {
    /// Returns the target key string used in `TargetId`.
    fn target_key(&self) -> String {
        match self {
            Self::Lib => "lib".to_string(),
            Self::Bin(name) => format!("bin/{name}"),
            Self::UnitTest => "test".to_string(),
            Self::IntegrationTest(name) => format!("test/{name}"),
            Self::Example(name) => format!("example/{name}"),
            Self::Bench(name) => format!("bench/{name}"),
        }
    }

    /// Whether this is a named target (integration test, example, or bench).
    fn is_named_target(&self) -> bool {
        matches!(
            self,
            Self::IntegrationTest(_) | Self::Example(_) | Self::Bench(_)
        )
    }
}

/// Whether rustc is compiling in test mode (`--test` flag).
///
/// When enabled, only `#[cfg(test)]` symbols are extracted to avoid
/// duplicating the lib symbol set. Orthogonal to `TargetKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TestMode(bool);

impl TestMode {
    /// Whether test mode is active.
    fn is_test(self) -> bool {
        self.0
    }
}

/// Context for tagging rustc invocations in tracing spans.
///
/// Why: keeps log metadata consistent without threading many parameters.
struct RustcSpanContext<'a> {
    crate_name: Option<&'a str>,
    package_name: Option<&'a str>,
    target_key: Option<&'a str>,
    emit: Option<&'a str>,
    is_test: Option<bool>,
    is_named_target: Option<bool>,
}

/// Owns invocation strings for shared borrows across callbacks and spans.
///
/// Why: avoids cloning names while keeping them alive for the rustc run.
struct InvocationNames {
    crate_name: String,
    package_name: String,
    target_key: String,
}

/// Bundled inputs for a workspace extraction run.
///
/// Why: keeps the extraction entrypoint small and avoids long parameter lists.
struct WorkspaceInvocation {
    config: DriverConfig,
    names: InvocationNames,
    test_mode: TestMode,
    target_kind: TargetKind,
    emit_str: String,
    target_profile_dir: PathBuf,
}

impl<'a> RustcSpanContext<'a> {
    /// Build a context from optional invocation metadata.
    ///
    /// Why: some pass-through paths only have partial information.
    fn new(
        crate_name: Option<&'a str>,
        package_name: Option<&'a str>,
        target_key: Option<&'a str>,
        emit: Option<&'a str>,
        is_test: Option<bool>,
        is_named_target: Option<bool>,
    ) -> Self {
        Self {
            crate_name,
            package_name,
            target_key,
            emit,
            is_test,
            is_named_target,
        }
    }
}

/// Run the driver: act as a rustc wrapper.
///
/// Arguments should be `["/path/to/rustc", ...actual_rustc_args]`.
///
/// Why: cargo invokes this binary as `RUSTC_WRAPPER` during builds.
pub fn run(args: &[String]) -> ExitCode {
    // Parse environment to determine if we should extract.
    // Environment not set up means just run rustc normally.
    // This can happen during initial cargo probing.
    let Some(config) = DriverConfig::from_env() else {
        trace!("no tarjanize config, passing through to rustc");
        let context = RustcSpanContext::new(None, None, None, None, None, None);
        return run_passthrough(args, &context);
    };

    // Determine the crate name from rustc arguments.
    // Can't determine crate name means pass through to rustc.
    let Some(crate_name) = find_crate_name(args) else {
        trace!("no crate name in args, passing through to rustc");
        let context = RustcSpanContext::new(None, None, None, None, None, None);
        return run_passthrough(args, &context);
    };

    let Some((package_name, is_named_target)) =
        resolve_workspace_package(&crate_name, args, &config)
    else {
        let context = RustcSpanContext::new(
            Some(&crate_name),
            None,
            None,
            None,
            None,
            None,
        );
        return run_passthrough(args, &context);
    };

    // Determine target kind and test mode from rustc args.
    let test_mode = TestMode(args.iter().any(|a| a == "--test"));
    let target_kind = determine_target_kind(args, &crate_name, is_named_target);
    let target_key = target_kind.target_key();

    // Only record profiles from metadata-only invocations to avoid duplicate
    // profile files for targets that build multiple times under cargo check.
    let (emit_str, metadata_only) = parse_emit_metadata_only(args);

    // Bundle crate identity for extraction and profile processing.
    let id = CrateIdentity {
        crate_name: crate_name.clone(),
        target_id: TargetId::new(&package_name, &target_key),
    };

    log_rustc_invocation(args, &id, test_mode, &target_kind);

    let passthrough_context = RustcSpanContext::new(
        Some(&crate_name),
        Some(&package_name),
        Some(&target_key),
        Some(emit_str.as_str()),
        Some(test_mode.is_test()),
        Some(target_kind.is_named_target()),
    );

    if !metadata_only {
        debug!(
            crate_name,
            package_name,
            target_key,
            emit = %emit_str,
            "skipping non-metadata invocation"
        );
        return run_passthrough(args, &passthrough_context);
    }

    let target_profile_dir = build_target_profile_dir(
        &config.profile_dir,
        &package_name,
        &target_key,
    );

    if !claim_profile_dir(&target_profile_dir) {
        debug!(
            crate_name,
            package_name,
            target_key,
            emit = %emit_str,
            dir = %target_profile_dir.display(),
            "skipping duplicate metadata invocation"
        );
        return run_passthrough(args, &passthrough_context);
    }

    let invocation = WorkspaceInvocation {
        config,
        names: InvocationNames {
            crate_name,
            package_name,
            target_key,
        },
        test_mode,
        target_kind,
        emit_str,
        target_profile_dir,
    };

    run_workspace_extraction(args, invocation)
}

/// Build the profile directory path for a target.
///
/// Why: each target needs its own directory to avoid concurrent collisions.
fn build_target_profile_dir(
    profile_root: &Path,
    package_name: &str,
    target_key: &str,
) -> PathBuf {
    profile_root.join(format!(
        "{}_{}",
        package_name,
        target_key.replace('/', "_")
    ))
}

/// Run extraction for a workspace crate and write its symbol graph.
///
/// Why: keeps the main driver flow short and focused on routing.
fn run_workspace_extraction(
    args: &[String],
    invocation: WorkspaceInvocation,
) -> ExitCode {
    let WorkspaceInvocation {
        config,
        names,
        test_mode,
        target_kind,
        emit_str,
        target_profile_dir,
    } = invocation;

    debug!(
        crate_name = %names.crate_name,
        package_name = %names.package_name,
        target_key = %names.target_key,
        is_test = test_mode.is_test(),
        target_kind = ?target_kind,
        "extracting symbols from workspace crate"
    );

    // This is a workspace crate - run with our extraction callbacks.
    let mut callbacks = TarjanizeCallbacks {
        config,
        crate_name: names.crate_name.as_str(),
        package_name: names.package_name.as_str(),
        target_key: names.target_key.as_str(),
        test_mode,
        extracted_module: Mutex::new(None),
    };

    let context = RustcSpanContext::new(
        Some(names.crate_name.as_str()),
        Some(names.package_name.as_str()),
        Some(names.target_key.as_str()),
        Some(emit_str.as_str()),
        Some(test_mode.is_test()),
        Some(target_kind.is_named_target()),
    );

    // Prepare profile directory and profiling flags; abort if profiling
    // cannot be set up because extraction would be incomplete.
    let compiler_args = match build_compiler_args(args, &target_profile_dir) {
        Ok(args) => args,
        Err(err) => {
            warn!(
                crate_name = %names.crate_name,
                package_name = %names.package_name,
                target_key = %names.target_key,
                dir = %target_profile_dir.display(),
                error = %err,
                "failed to prepare profile directory"
            );
            return ExitCode::FAILURE;
        }
    };

    let exit_code =
        run_rustc_with_span(&compiler_args, &mut callbacks, &context);

    // After rustc completes (including codegen), apply costs from profile
    // data, write the results, then delete profile files.
    // Recover from mutex poisoning to avoid aborting after extraction panics.
    let mut extracted_module = match callbacks.extracted_module.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            warn!(
                crate_name = %callbacks.crate_name,
                package_name = %callbacks.package_name,
                target_key = %callbacks.target_key,
                "extracted module mutex poisoned; proceeding with recovered data"
            );
            poisoned.into_inner()
        }
    };
    if let Some(module) = extracted_module.take() {
        let crate_id = CrateIdentity {
            crate_name: callbacks.crate_name.to_owned(),
            target_id: TargetId::new(
                callbacks.package_name,
                callbacks.target_key,
            ),
        };
        if let Err(err) = process_and_write_crate(
            &callbacks.config,
            &crate_id,
            module,
            &target_profile_dir,
        ) {
            warn!(
                crate_name = %callbacks.crate_name,
                package_name = %callbacks.package_name,
                target_key = %callbacks.target_key,
                error = %err,
                "failed to process extracted crate data"
            );
            return ExitCode::FAILURE;
        }
    }

    exit_code
}

/// Resolve the workspace package for a rustc crate, if any.
///
/// Returns `(package_name, is_named_target)` when the crate belongs to
/// the workspace, otherwise `None`.
///
/// Why: only workspace crates should be extracted; externals pass through.
fn resolve_workspace_package(
    crate_name: &str,
    args: &[String],
    config: &DriverConfig,
) -> Option<(String, bool)> {
    // Direct match: crate_name matches a workspace package (e.g., "tokio").
    let direct_match = config
        .workspace_crates
        .iter()
        .find(|pkg_name| pkg_name.replace('-', "_") == crate_name)
        .cloned();

    if let Some(pkg) = direct_match {
        return Some((pkg, false));
    }

    // Fallback: integration test crates derive their name from the file.
    if let Some(source_file) = find_source_file(args) {
        if let Some(pkg) =
            find_package_for_source(&config.workspace_paths, &source_file)
        {
            return Some((pkg, true));
        }
        trace!(
            crate_name,
            source_file = %source_file.display(),
            "source not in workspace, compiling without extraction"
        );
        return None;
    }

    trace!(
        crate_name,
        "not a workspace crate, compiling without extraction"
    );
    None
}

/// Parse `--emit` flags and decide whether this is metadata-only.
///
/// Returns `(emit_str, metadata_only)`.
///
/// Why: we only profile metadata-only invocations to avoid duplicates.
fn parse_emit_metadata_only(args: &[String]) -> (String, bool) {
    let emit = extract_flag_value(args, "--emit");
    let emit_str = emit.unwrap_or_default();
    let emit_set: HashSet<&str> = emit_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let allowed_emits = ["metadata", "dep-info"];
    let metadata_only = emit_set.contains("metadata")
        && emit_set.iter().all(|emit| allowed_emits.contains(emit));
    (emit_str, metadata_only)
}

/// Build rustc arguments with extraction-specific flags.
///
/// # Errors
/// Returns an error if the profile directory cannot be created.
///
/// Why: keeps the THIR and self-profile configuration in one place.
fn build_compiler_args(
    args: &[String],
    target_profile_dir: &Path,
) -> Result<Vec<String>> {
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
    fs::create_dir_all(target_profile_dir).with_context(|| {
        format!(
            "failed to create profile directory {}",
            target_profile_dir.display()
        )
    })?;
    compiler_args
        .push(format!("-Zself-profile={}", target_profile_dir.display()));
    compiler_args.push("-Zself-profile-events=default,args".to_string());

    Ok(compiler_args)
}

/// Log rustc invocation metadata for debugging and analysis.
///
/// Why: traceable build metadata is essential when extraction is skipped.
fn log_rustc_invocation(
    args: &[String],
    id: &CrateIdentity,
    test_mode: TestMode,
    target_kind: &TargetKind,
) {
    let crate_name = id.crate_name();
    let package_name = id.package_name();
    let target_key = id.target_key();
    let emit = extract_flag_value(args, "--emit");
    let mut crate_types = Vec::new();
    collect_flag_values(args, "--crate-type", &mut crate_types);
    let metadata = extract_codegen_value(args, "metadata");
    let extra_filename = extract_codegen_value(args, "extra-filename");
    let target = extract_flag_value(args, "--target");

    let emit_str = emit.as_deref().unwrap_or("");
    let metadata_only = emit_str.contains("metadata")
        && !emit_str.contains("link")
        && !emit_str.contains("obj")
        && !emit_str.contains("assembly");

    debug!(
        crate_name,
        package_name,
        target_key,
        is_test = test_mode.is_test(),
        target_kind = ?target_kind,
        emit = %emit_str,
        crate_types = ?crate_types,
        target = %target.as_deref().unwrap_or(""),
        metadata = %metadata.as_deref().unwrap_or(""),
        extra_filename = %extra_filename.as_deref().unwrap_or(""),
        metadata_only,
        "rustc invocation"
    );
}

/// Run rustc with a tracing span derived from the provided context.
///
/// Why: ensures consistent log fields across normal and extraction runs.
fn run_rustc_with_span<C: Callbacks + Send>(
    args: &[String],
    callbacks: &mut C,
    context: &RustcSpanContext<'_>,
) -> ExitCode {
    let span = tracing::debug_span!(
        "rustc_invocation",
        crate_name = context.crate_name.unwrap_or("<unknown>"),
        package_name = context.package_name.unwrap_or("<unknown>"),
        target_key = context.target_key.unwrap_or("<unknown>"),
        emit = context.emit.unwrap_or(""),
        is_test = context.is_test.unwrap_or(false),
        is_named_target = context.is_named_target.unwrap_or(false),
    );
    let _enter = span.enter();
    run_via_rustc_driver(args, callbacks)
}

/// Run rustc without extraction, preserving span context.
///
/// Why: pass-through paths still need consistent tracing fields.
fn run_passthrough(
    args: &[String],
    context: &RustcSpanContext<'_>,
) -> ExitCode {
    run_rustc_with_span(args, &mut NoOpCallbacks, context)
}

/// Extract the value for a single flag from rustc arguments.
///
/// Why: rustc arguments allow both `--flag value` and `--flag=value`.
fn extract_flag_value(args: &[String], flag: &str) -> Option<String> {
    let flag_eq = format!("{flag}=");
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            if let Some(value) = iter.next() {
                return Some(value.clone());
            }
        } else if let Some(value) = arg.strip_prefix(&flag_eq) {
            return Some(value.to_string());
        }
    }
    None
}

/// Collect repeated flag values from rustc arguments.
///
/// Why: flags like `--crate-type` can appear multiple times.
fn collect_flag_values(args: &[String], flag: &str, out: &mut Vec<String>) {
    let flag_eq = format!("{flag}=");
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            if let Some(value) = iter.next() {
                out.push(value.clone());
            }
        } else if let Some(value) = arg.strip_prefix(&flag_eq) {
            out.push(value.to_string());
        }
    }
}

/// Extract a `-C key=value` codegen option from rustc args.
///
/// Why: used for logging and metadata-only detection.
fn extract_codegen_value(args: &[String], key: &str) -> Option<String> {
    let key_eq = format!("{key}=");
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "-C"
            && let Some(value) = iter.next()
            && let Some(rest) = value.strip_prefix(&key_eq)
        {
            return Some(rest.to_string());
        } else if let Some(rest) = arg.strip_prefix("-C")
            && let Some(value) = rest.strip_prefix(&key_eq)
        {
            return Some(value.to_string());
        }
    }
    None
}

/// Atomically claim a profile directory for extraction.
///
/// Returns `true` if this invocation is the first to claim the directory,
/// `false` if another concurrent invocation already claimed it.
///
/// `cargo check --all-targets` can invoke rustc multiple times for the same
/// crate with different feature sets (different `-C metadata` hashes). Both
/// invocations are metadata-only and map to the same target key. The old
/// `profile_dir_has_profile` check was racy (TOCTOU): it looked for
/// `.mm_profdata` files, but rustc only writes those at process exit, so
/// concurrent invocations would both see an empty directory and proceed.
///
/// This function uses `File::create_new` on a sentinel file, which is
/// atomic on all platforms. The first invocation creates the sentinel and
/// proceeds; subsequent invocations see `AlreadyExists` and skip.
///
/// Why: prevents duplicate extraction and duplicate profile files without
/// a TOCTOU race window.
fn claim_profile_dir(dir: &Path) -> bool {
    // Ensure the directory exists so we can place the sentinel.
    if fs::create_dir_all(dir).is_err() {
        return false;
    }

    let sentinel = dir.join(".claimed");
    match fs::File::create_new(&sentinel) {
        Ok(_) => true,
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => false,
        Err(e) => {
            warn!(
                dir = %dir.display(),
                error = %e,
                "failed to create profile sentinel"
            );
            false
        }
    }
}

/// Run the compiler via `rustc_driver` with the given callbacks.
///
/// This ensures all crates are compiled with the same compiler version,
/// avoiding "compiled by incompatible rustc" errors that occur when mixing
/// different compiler versions.
///
/// Why: `rustc_driver` handles compiler setup and fatal errors uniformly.
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
///
/// Why: external crates should compile normally without extra hooks.
struct NoOpCallbacks;

impl Callbacks for NoOpCallbacks {}

/// Configuration parsed from environment variables.
///
/// Why: the driver must be configured via env when invoked by cargo.
pub(crate) struct DriverConfig {
    pub(crate) output_dir: PathBuf,
    workspace_crates: Vec<String>,
    /// Maps package name to its manifest directory path.
    /// Used to identify which package integration tests belong to.
    ///
    /// Why: integration test crates must be mapped back to their package.
    workspace_paths: HashMap<String, PathBuf>,
    /// Directory for self-profile output.
    ///
    /// Why: per-target profiles need a shared base directory.
    profile_dir: PathBuf,
}

impl DriverConfig {
    /// Load the driver configuration from environment variables.
    ///
    /// Why: cargo invokes the driver without CLI args, so env is the contract.
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

        let profile_dir: PathBuf = env::var(ENV_PROFILE_DIR).ok()?.into();

        trace!(
            output_dir = %output_dir.display(),
            workspace_crates = ?workspace_crates,
            workspace_paths_count = workspace_paths.len(),
            profile_dir = %profile_dir.display(),
            "loaded driver config from env"
        );

        Some(Self {
            output_dir,
            workspace_crates,
            workspace_paths,
            profile_dir,
        })
    }
}

/// Find the crate name from rustc arguments.
/// Looks for `--crate-name <name>`.
///
/// Why: crate name is the primary identity for extraction and mapping.
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
///
/// Why: integration test crates are identified by their source file path.
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
///
/// Why: integration test crate names don't match package names.
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
/// Determines the `TargetKind` from rustc arguments and resolution context.
///
/// The `is_named_target` flag indicates this is an integration test,
/// example, or bench (detected by the caller from crate/package name mismatch).
///
/// Why: target keys must match the orchestrator's mapping scheme.
fn determine_target_kind(
    args: &[String],
    crate_name: &str,
    is_named_target: bool,
) -> TargetKind {
    let is_test = args.iter().any(|a| a == "--test");
    let is_bin = args.iter().any(|a| a == "--crate-type=bin")
        || args
            .windows(2)
            .any(|w| w[0] == "--crate-type" && w[1] == "bin");

    // For named targets (integration tests/examples/benches), inspect the
    // source path to distinguish the exact target directory.
    if is_named_target {
        if let Some(source) = find_source_file(args) {
            let source_str = source.to_string_lossy();
            trace!(
                crate_name,
                source = %source_str,
                "determining target kind from source path"
            );
            // Check for standard target directories. Handle both:
            // - Absolute/workspace-relative paths: tokio/tests/foo.rs (contains /tests/)
            // - Package-relative paths: tests/foo.rs (starts with tests/)
            if source_str.contains("/tests/")
                || source_str.contains("\\tests\\")
                || source_str.starts_with("tests/")
            {
                return TargetKind::IntegrationTest(crate_name.to_string());
            } else if source_str.contains("/examples/")
                || source_str.contains("\\examples\\")
                || source_str.starts_with("examples/")
            {
                return TargetKind::Example(crate_name.to_string());
            } else if source_str.contains("/benches/")
                || source_str.contains("\\benches\\")
                || source_str.starts_with("benches/")
            {
                return TargetKind::Bench(crate_name.to_string());
            }
        } else {
            trace!(crate_name, "no source file found in args");
        }
        // Fallback: if it's a test target, call it a test
        if is_test {
            return TargetKind::IntegrationTest(crate_name.to_string());
        }
    }

    // Standard targets where crate_name matches package_name.
    if is_test {
        TargetKind::UnitTest
    } else if is_bin {
        TargetKind::Bin(crate_name.to_string())
    } else {
        TargetKind::Lib
    }
}

/// Callbacks implementation for symbol extraction.
///
/// The extracted module is stored in `extracted_module` for post-compilation
/// processing. We use a Mutex because `rustc_driver` requires `Callbacks + Send`.
///
/// Why: extraction must happen inside rustc, but cost application happens after.
struct TarjanizeCallbacks<'a> {
    config: DriverConfig,
    /// The rustc crate name (e.g., `sync_mpsc` for an integration test).
    crate_name: &'a str,
    /// The cargo package name this target belongs to (e.g., "tokio").
    /// For lib/bin targets, this equals `crate_name` (with underscores).
    /// For integration tests, this is the parent package.
    package_name: &'a str,
    /// The target key (e.g., "lib", "test", `test/sync_mpsc`).
    target_key: &'a str,
    /// Whether this is a test target (compiled with `--test`).
    /// When true, only `#[cfg(test)]` items are extracted to avoid
    /// duplicating symbols from the lib target.
    test_mode: TestMode,
    /// The extracted module, populated by `after_analysis`.
    /// Stored here so we can apply costs after rustc completes.
    ///
    /// Why: `after_analysis` runs before profiling data is finalized.
    extracted_module: Mutex<Option<Module>>,
}

impl Callbacks for TarjanizeCallbacks<'_> {
    /// Called after type checking is complete.
    ///
    /// Extracts symbols and stores them for post-compilation processing.
    /// Profile data is applied after rustc completes.
    ///
    /// Why: we need fully type-checked HIR/THIR to extract dependencies.
    fn after_analysis(
        &mut self,
        _compiler: &Compiler,
        tcx: TyCtxt<'_>,
    ) -> Compilation {
        // Extract symbols from this crate.
        // For test targets, only extract #[cfg(test)] items to avoid
        // duplicating the entire lib symbol set. The compiler already
        // excludes cfg(test) items from lib targets, so this separation
        // ensures no symbol duplication between lib and test targets.
        debug!(crate_name = %self.crate_name, is_test = self.test_mode.is_test(), "extracting symbols");
        let extraction = extract::extract_crate(
            tcx,
            self.crate_name,
            &self.config.workspace_crates,
            self.test_mode.is_test(),
        );

        let symbol_count = extraction.module.count_symbols();
        debug!(crate_name = %self.crate_name, symbol_count, "extraction complete");

        if symbol_count == 0 {
            if self.test_mode.is_test() {
                // Test targets with no #[cfg(test)] items produce 0 symbols
                // because the test_only filter excludes everything to avoid
                // duplicating lib symbols. Skip writing this target entirely
                // — it adds no useful information for scheduling analysis.
                debug!(
                    crate_name = %self.crate_name,
                    package = %self.package_name,
                    target = %self.target_key,
                    "skipping test target with no #[cfg(test)] items"
                );
                return Compilation::Continue;
            }
            // Facade crates (e.g., gateway-types re-exporting from
            // gateway-types-versions) consist entirely of `pub use` and `mod`
            // items, which we don't extract as symbols. These are real nodes
            // in the dependency graph so we keep them — they just have 0
            // symbols and negligible self-cost.
            debug!(
                crate_name = %self.crate_name,
                package = %self.package_name,
                target = %self.target_key,
                "non-test target produced 0 symbols (likely a facade/re-export crate)"
            );
        }

        // Store the module for post-compilation processing.
        // We'll apply costs and write the final Target JSON after rustc
        // completes (including codegen).
        // If a previous panic poisoned the mutex, recover so we can still
        // surface the extracted module for diagnostics.
        let mut extracted_module = match self.extracted_module.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!(
                    crate_name = %self.crate_name,
                    package_name = %self.package_name,
                    target_key = %self.target_key,
                    "extracted module mutex poisoned; continuing with recovered data"
                );
                poisoned.into_inner()
            }
        };
        *extracted_module = Some(extraction.module);

        // Continue compilation so cargo can use the artifacts.
        Compilation::Continue
    }
}

/// Process profile data, apply per-event timing breakdowns to the module, and
/// write the final Target.
///
/// This runs after rustc completes so we have access to self-profile data
/// for per-event timing attribution. Backend cost tracking was removed because
/// it's unreliable and irrelevant for crate splitting (backend is parallel via
/// CGUs and doesn't meaningfully affect the critical path).
///
/// After writing the Target JSON, deletes the profile directory for this target
/// to avoid accumulating huge amounts of temp data on large workspaces.
///
/// The `crate_name` is the rustc crate name (e.g., `sync_mpsc` for integration tests).
/// The `package_name` is the cargo package this target belongs to (e.g., "tokio").
/// For lib targets, these are the same (modulo hyphen/underscore normalization).
/// For integration tests/benches/examples, they differ.
///
///
/// # Errors
/// Returns an error if the crate output cannot be serialized or written.
///
/// Why: combines extraction and profiling into a single serialized output.
fn process_and_write_crate(
    config: &DriverConfig,
    id: &CrateIdentity,
    mut module: Module,
    profile_dir: &Path,
) -> Result<()> {
    let crate_name = id.crate_name();
    let package_name = id.package_name();
    let target_key = id.target_key();
    let _span = tracing::debug_span!(
        "process_and_write_crate",
        crate_name,
        package_name,
        target_key,
        profile_dir = %profile_dir.display(),
    )
    .entered();

    let (symbol_paths, module_paths) = collect_profile_paths(&module, id);

    // Load profile data from this target's dedicated profile directory.
    let mut profile_data =
        load_profile_data_with_span(profile_dir, id, &symbol_paths);

    // Get target timings (wall-clock time and unattributed event times).
    // Also uses crate_name since profile data is keyed by rustc crate name.
    let mut timings = profile_data
        .get_target_timings(crate_name)
        .cloned()
        .unwrap_or_default();
    let crate_prefix = format!("{crate_name}::");
    let summary = roll_up_unmatched_with_span(
        &mut profile_data,
        &symbol_paths,
        &module_paths,
        &crate_prefix,
        id,
    );

    // Apply per-event timing breakdowns to the module's symbols.
    // Use crate_name for profile lookups since profile data is keyed by rustc
    // crate name (e.g., "sync_mpsc"), not cargo package name (e.g., "tokio").
    apply_event_times_with_span(&mut module, id, &profile_data);
    for (label, ms) in &summary.totals_by_label {
        *timings.event_times_ms.entry(label.clone()).or_default() += *ms;
    }
    if summary.total_unmatched_ms > Duration::ZERO {
        *timings
            .event_times_ms
            .entry("unmatched".to_string())
            .or_default() += summary.total_unmatched_ms;
    }

    append_unmatched_with_span(config, id, &summary);

    // Build the complete Target.
    // Dependencies are populated later by the orchestrator from cargo metadata.
    let crate_data = Target {
        timings,
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

    let json = serde_json::to_string_pretty(&result).with_context(|| {
        format!(
            "failed to serialize crate output for {package_name}/{target_key}"
        )
    })?;
    fs::write(&output_path, &json).with_context(|| {
        format!("failed to write crate output to {}", output_path.display())
    })?;

    debug!(
        path = %output_path.display(),
        bytes = json.len(),
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The primary binary (crate name matches package name after hyphen→underscore
    /// normalization) must include the crate name in the target key, producing
    /// `"bin/{crate_name}"` instead of bare `"bin"`. Without this, impl block
    /// paths like `[pkg/bin]::{{impl}}[N]` won't match anchor paths like
    /// `[pkg/bin/name]::Struct`, causing condense to strand impls as roots.
    #[test]
    fn primary_binary_target_key_includes_name() {
        let args = vec!["--crate-type=bin".to_string()];
        let kind = determine_target_kind(&args, "omicron_dev", false);
        assert_eq!(kind, TargetKind::Bin("omicron_dev".into()));
        assert_eq!(kind.target_key(), "bin/omicron_dev");
    }

    /// Secondary binaries (crate name differs from package name) should also
    /// produce `Bin("{crate_name}")`.
    #[test]
    fn secondary_binary_target_key_includes_name() {
        let args = vec!["--crate-type=bin".to_string()];
        let kind = determine_target_kind(&args, "other_bin", false);
        assert_eq!(kind, TargetKind::Bin("other_bin".into()));
        assert_eq!(kind.target_key(), "bin/other_bin");
    }

    /// Lib targets should produce `TargetKind::Lib`.
    #[test]
    fn lib_target_kind() {
        let args = vec!["--crate-type=lib".to_string()];
        let kind = determine_target_kind(&args, "omicron_dev", false);
        assert_eq!(kind, TargetKind::Lib);
        assert_eq!(kind.target_key(), "lib");
    }

    /// Unit test targets (lib with --test) should produce `UnitTest`.
    #[test]
    fn unit_test_target_kind() {
        let args = vec!["--crate-type=lib".to_string(), "--test".to_string()];
        let kind = determine_target_kind(&args, "omicron_dev", false);
        assert_eq!(kind, TargetKind::UnitTest);
        assert_eq!(kind.target_key(), "test");
    }

    /// Named targets default to `IntegrationTest` when no source file
    /// is found in args.
    #[test]
    fn named_target_defaults_to_integration_test() {
        let args = vec!["--test".to_string()];
        let kind = determine_target_kind(&args, "my_test", true);
        assert_eq!(kind, TargetKind::IntegrationTest("my_test".into()));
        assert_eq!(kind.target_key(), "test/my_test");
    }

    #[test]
    fn profile_path_from_crate() {
        let p = ProfilePath::from_crate("my_crate");
        assert_eq!(p.as_str(), "my_crate");
    }

    #[test]
    fn profile_path_child_chain() {
        let p = ProfilePath::from_crate("my_crate")
            .child("foo")
            .child("Bar");
        assert_eq!(p.as_str(), "my_crate::foo::Bar");
    }

    #[test]
    fn profile_path_display() {
        let p = ProfilePath::from_crate("c").child("m");
        assert_eq!(p.to_string(), "c::m");
    }
}

/// Result written by the driver for each crate/target.
///
/// Why: orchestrator consumes this JSON to assemble the workspace graph.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CrateResult {
    pub crate_name: String,
    #[serde(flatten)]
    pub crate_data: Target,
}
