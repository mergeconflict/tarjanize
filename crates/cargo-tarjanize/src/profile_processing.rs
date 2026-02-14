//! Profile processing: collect paths, load profiles, apply event times.
//!
//! These functions bridge the extraction phase (HIR/THIR symbol extraction)
//! with the profiling phase (`-Zself-profile` data). They handle:
//!
//! - Collecting symbol and module paths for profile matching
//! - Loading profile data from self-profile output
//! - Rolling up unmatched frontend costs to parent modules
//! - Applying per-event timing breakdowns to extracted symbols
//! - Writing unmatched-path diagnostic reports
//!
//! Why a separate module: these functions form a cohesive group that depends
//! on `ProfileData`, `ProfilePath`, `CrateIdentity`, and `Module` but are
//! independent of the rustc driver callbacks and compiler argument handling
//! in `driver.rs`.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{env, fs, io};

use tarjanize_schemas::{Module, duration_to_ms_f64};
use tracing::warn;

use crate::driver::{CrateIdentity, DriverConfig, ProfilePath};
use crate::profile::{ProfileData, RollupSummary};

/// Collect profile-related symbol and module paths with tracing spans.
///
/// Why: path collection is shared between rollup and event attribution.
pub(crate) fn collect_profile_paths(
    module: &Module,
    id: &CrateIdentity,
) -> (HashSet<String>, HashSet<String>) {
    let crate_name = id.crate_name();
    let package_name = id.package_name();
    let target_key = id.target_key();
    let mut symbol_paths = HashSet::new();
    tracing::debug_span!(
        "collect_symbol_paths",
        crate_name,
        package_name,
        target_key
    )
    .in_scope(|| {
        collect_symbol_paths(
            module,
            &ProfilePath::from_crate(crate_name),
            &mut symbol_paths,
        );
    });

    let mut module_paths = HashSet::new();
    tracing::debug_span!(
        "collect_module_paths",
        crate_name,
        package_name,
        target_key
    )
    .in_scope(|| {
        collect_module_paths(
            module,
            &ProfilePath::from_crate(crate_name),
            &mut module_paths,
        );
    });

    (symbol_paths, module_paths)
}

/// Load profile data with a tracing span.
///
/// Why: keeps profiling I/O visible in traces without duplicating span setup.
pub(crate) fn load_profile_data_with_span(
    profile_dir: &Path,
    id: &CrateIdentity,
    symbol_paths: &HashSet<String>,
) -> ProfileData {
    let _span = tracing::debug_span!(
        "load_profile",
        package = %id.package_name(),
        target = %id.target_key(),
        crate_name = %id.crate_name(),
    )
    .entered();
    ProfileData::load_from_dir_with_symbols(profile_dir, Some(symbol_paths))
}

/// Roll up unmatched frontend costs with a tracing span.
///
/// Why: consolidates summary generation and its tracing metadata.
pub(crate) fn roll_up_unmatched_with_span(
    profile_data: &mut ProfileData,
    symbol_paths: &HashSet<String>,
    module_paths: &HashSet<String>,
    crate_prefix: &str,
    id: &CrateIdentity,
) -> RollupSummary {
    let crate_name = id.crate_name();
    let package_name = id.package_name();
    let target_key = id.target_key();
    tracing::debug_span!(
        "roll_up_unmatched_frontend_costs",
        crate_name,
        package_name,
        target_key
    )
    .in_scope(|| {
        profile_data.roll_up_unmatched_frontend_costs(
            symbol_paths,
            module_paths,
            crate_prefix,
        )
    })
}

/// Apply event times to the module with a tracing span.
///
/// Why: keeps per-target attribution visible in traces.
pub(crate) fn apply_event_times_with_span(
    module: &mut Module,
    id: &CrateIdentity,
    profile_data: &ProfileData,
) {
    let crate_name = id.crate_name();
    let package_name = id.package_name();
    let target_key = id.target_key();
    tracing::debug_span!(
        "apply_event_times",
        crate_name,
        package_name,
        target_key
    )
    .in_scope(|| {
        apply_event_times(
            module,
            &ProfilePath::from_crate(crate_name),
            profile_data,
        );
    });
}

/// Append unmatched paths with a tracing span.
///
/// Why: unmatched-path reporting is useful but should not derail extraction.
pub(crate) fn append_unmatched_with_span(
    config: &DriverConfig,
    id: &CrateIdentity,
    summary: &RollupSummary,
) {
    let crate_name = id.crate_name();
    let package_name = id.package_name();
    let target_key = id.target_key();
    tracing::debug_span!(
        "append_unmatched_paths",
        crate_name,
        package_name,
        target_key
    )
    .in_scope(|| {
        if let Err(error) =
            append_unmatched_paths(&unmatched_output_path(config), id, summary)
        {
            warn!(
                error = %error,
                "failed to append unmatched frontend paths"
            );
        }
    });
}

/// Apply per-event timing breakdowns from profile data to symbols.
///
/// For each symbol, looks up its event-level self-time map from the profile
/// data and stores it in `symbol.event_times_ms`. Symbols without profile
/// data keep an empty map.
///
/// Why: per-symbol timings are required for cost modeling and visualization.
fn apply_event_times(
    module: &mut Module,
    path_prefix: &ProfilePath,
    profile_data: &ProfileData,
) {
    for (name, symbol) in &mut module.symbols {
        let full_path = path_prefix.child(name);
        if let Some(event_map) =
            profile_data.get_event_times_ms(full_path.as_str())
        {
            symbol.event_times_ms = event_map;
        }
    }

    for (submod_name, submodule) in &mut module.submodules {
        apply_event_times(
            submodule,
            &path_prefix.child(submod_name),
            profile_data,
        );
    }
}

/// Determine the output path for unmatched-path logs.
///
/// Why: allow overrides while defaulting to the output directory.
fn unmatched_output_path(config: &DriverConfig) -> PathBuf {
    if let Ok(path) = env::var("TARJANIZE_UNMATCHED_PATH") {
        PathBuf::from(path)
    } else {
        config.output_dir.join("unmatched_paths.tsv")
    }
}

/// Append unmatched paths to a TSV report.
///
/// Why: unmatched costs indicate missing attribution and need visibility.
fn append_unmatched_paths(
    output_path: &Path,
    id: &CrateIdentity,
    summary: &RollupSummary,
) -> io::Result<()> {
    if summary.unmatched_paths.is_empty() && summary.module_paths.is_empty() {
        return Ok(());
    }

    let package_name = id.package_name();
    let target_key = id.target_key();
    let crate_name = id.crate_name();

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;
    if file.metadata()?.len() == 0 {
        writeln!(file, "package\ttarget\tcrate\tkind\tpath\ttotal_ms")?;
    }

    for (path, ms) in &summary.unmatched_paths {
        let path = path.replace(['\t', '\n'], " ");
        let ms = duration_to_ms_f64(*ms);
        writeln!(
            file,
            "{package_name}\t{target_key}\t{crate_name}\tunmatched\t{path}\t{ms:.6}"
        )?;
    }
    for (path, ms) in &summary.module_paths {
        let path = path.replace(['\t', '\n'], " ");
        let ms = duration_to_ms_f64(*ms);
        writeln!(
            file,
            "{package_name}\t{target_key}\t{crate_name}\tmodule\t{path}\t{ms:.6}"
        )?;
    }

    Ok(())
}

/// Collect all symbol paths into `symbol_paths`.
///
/// Paths are normalized to underscore crate names to match profile keys.
///
/// Why: profile keys use crate-name normalization; path collection must match.
fn collect_symbol_paths(
    module: &Module,
    path_prefix: &ProfilePath,
    symbol_paths: &mut HashSet<String>,
) {
    for name in module.symbols.keys() {
        let full_path = path_prefix.child(name).as_str().replace('-', "_");
        symbol_paths.insert(full_path);
    }

    for (submod_name, submodule) in &module.submodules {
        collect_symbol_paths(
            submodule,
            &path_prefix.child(submod_name),
            symbol_paths,
        );
    }
}

/// Collect all module paths into `module_paths`.
///
/// These paths correspond to module `DefPaths`, which aren't extracted as symbols.
///
/// Why: unmatched costs may roll up to module scopes rather than symbols.
fn collect_module_paths(
    module: &Module,
    path_prefix: &ProfilePath,
    module_paths: &mut HashSet<String>,
) {
    for (submod_name, submodule) in &module.submodules {
        let submod_path = path_prefix.child(submod_name);
        let normalized = submod_path.as_str().replace('-', "_");
        module_paths.insert(normalized.clone());
        // Recurse with the normalized path so children also use the
        // dash-free prefix (matching rustc profile key format).
        let normalized_prefix = ProfilePath(normalized);
        collect_module_paths(submodule, &normalized_prefix, module_paths);
    }
}
