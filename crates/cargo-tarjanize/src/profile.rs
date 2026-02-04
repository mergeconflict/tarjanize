//! Self-profile data parsing for accurate cost estimation.
//!
//! This module parses rustc's `-Zself-profile` output to get real compilation
//! time per symbol. The approach:
//!
//! 1. Run a profiled build with `-Zself-profile` and `-Zself-profile-events=default,llvm,args`
//! 2. Parse the `.mm_profdata` files using the `analyzeme` crate
//! 3. Categorize events into frontend, backend (CGU), and overhead costs
//! 4. Use this timing data for the cost fields in `Symbol` and `Crate`
//!
//! ## Event Categories
//!
//! - **Frontend**: Serial compilation phases (parsing, type checking, borrow checking)
//! - **Backend**: LLVM codegen, parallelizable across CGUs
//! - **Overhead**: Per-crate fixed costs (linking, metadata generation)

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use analyzeme::ProfilingData;
use tracing::{debug, info, warn};

/// Category of a self-profile event for cost allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventCategory {
    /// Frontend work: parsing, type checking, borrow checking, etc.
    /// These run serially and scale with symbol count.
    Frontend,
    /// Backend work: LLVM codegen, runs in parallel across CGUs.
    Backend,
    /// Crate overhead: linking, metadata generation (fixed per crate).
    Overhead,
    /// Events we don't track (internal compiler events, etc.).
    Skip,
}

/// Overhead costs for a single crate.
#[derive(Debug, Default, Clone)]
pub struct CrateOverhead {
    /// Total linking time in milliseconds.
    pub linking_ms: f64,
    /// Metadata generation time in milliseconds.
    pub metadata_ms: f64,
}

/// Aggregated timing data from self-profile output.
///
/// Separates frontend costs (by `DefPath`) from backend costs (by CGU)
/// and crate-level overhead.
#[derive(Debug, Default)]
pub struct ProfileData {
    /// Frontend costs indexed by normalized `DefPath`.
    frontend_costs: HashMap<String, Duration>,
    /// Backend costs indexed by CGU name.
    cgu_costs: HashMap<String, Duration>,
    /// Crate overhead indexed by crate name.
    crate_overhead: HashMap<String, CrateOverhead>,
}

impl ProfileData {
    /// Load profile data from a directory containing `.mm_profdata` files.
    ///
    /// Parses all profile files in the directory and categorizes timing data.
    /// Returns an empty `ProfileData` if the directory doesn't exist
    /// or contains no profile files.
    pub fn load_from_dir(dir: &Path) -> Self {
        let mut data = ProfileData::default();

        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) => {
                warn!(
                    "failed to read profile directory {}: {}",
                    dir.display(),
                    e
                );
                return data;
            }
        };

        let mut file_count = 0;
        let mut event_count = 0;

        for entry in entries.flatten() {
            let path = entry.path();

            // Profile files have the .mm_profdata extension.
            if path.extension().is_some_and(|ext| ext == "mm_profdata") {
                // The stem is the path we pass to ProfilingData::new.
                // e.g., "tarjanize_schemas-0060816.mm_profdata" -> "tarjanize_schemas-0060816"
                if let Some(stem) = path.file_stem() {
                    let stem_str = stem.to_string_lossy();
                    let stem_path = dir.join(stem);

                    match ProfilingData::new(&stem_path) {
                        Ok(profile) => {
                            // Extract crate name from profile filename.
                            // Format: "crate_name-XXXXXXX" where X is hex digits.
                            let crate_name = extract_crate_name(&stem_str);

                            file_count += 1;
                            event_count +=
                                data.aggregate_profile(&profile, &crate_name);
                        }
                        Err(e) => {
                            debug!(
                                "failed to load profile {}: {}",
                                stem_path.display(),
                                e
                            );
                        }
                    }
                }
            }
        }

        info!(
            file_count,
            event_count,
            frontend_paths = data.frontend_costs.len(),
            cgu_count = data.cgu_costs.len(),
            crate_count = data.crate_overhead.len(),
            "loaded self-profile data"
        );

        data
    }

    /// Aggregate timing data from a single profile file.
    ///
    /// Returns the number of events processed.
    fn aggregate_profile(
        &mut self,
        profile: &ProfilingData,
        crate_name: &str,
    ) -> usize {
        let mut count = 0;

        for event in profile.iter_full() {
            let Some(dur) = event.duration() else {
                continue;
            };

            let label = &event.label;
            let category = categorize_event(label);

            match category {
                EventCategory::Frontend => {
                    // Frontend events have DefPath in additional_data[0].
                    if let Some(raw_path) = event.additional_data.first() {
                        let normalized = normalize_frontend_path(raw_path);
                        if !normalized.is_empty() {
                            *self
                                .frontend_costs
                                .entry(normalized)
                                .or_default() += dur;
                            count += 1;
                        }
                    }
                }
                EventCategory::Backend => {
                    // Backend events (LLVM) have CGU name in additional_data[0].
                    if let Some(cgu_name) = event.additional_data.first() {
                        *self
                            .cgu_costs
                            .entry(cgu_name.to_string())
                            .or_default() += dur;
                        count += 1;
                    }
                }
                EventCategory::Overhead => {
                    // Overhead events are per-crate fixed costs.
                    let overhead = self
                        .crate_overhead
                        .entry(crate_name.to_string())
                        .or_default();

                    if is_linking_event(label) {
                        overhead.linking_ms += dur.as_millis_f64();
                    } else if is_metadata_event(label) {
                        overhead.metadata_ms += dur.as_millis_f64();
                    }
                    count += 1;
                }
                EventCategory::Skip => {}
            }
        }

        count
    }

    /// Get the frontend compilation cost for a symbol path.
    ///
    /// Returns the total time in milliseconds, or None if no timing data exists.
    pub fn get_frontend_cost_ms(&self, path: &str) -> Option<f64> {
        self.frontend_costs.get(path).map(Duration::as_millis_f64)
    }

    /// Get all CGU costs for backend cost distribution.
    ///
    /// Returns a reference to the map of CGU name -> total duration.
    pub fn cgu_costs(&self) -> &HashMap<String, Duration> {
        &self.cgu_costs
    }

    /// Get crate overhead costs.
    ///
    /// Returns None if no overhead was recorded for this crate.
    pub fn get_crate_overhead(
        &self,
        crate_name: &str,
    ) -> Option<&CrateOverhead> {
        self.crate_overhead.get(crate_name)
    }

    /// Get the number of unique frontend paths with timing data.
    #[expect(dead_code, reason = "useful for debugging and logging")]
    pub fn frontend_count(&self) -> usize {
        self.frontend_costs.len()
    }

    /// Get the number of unique CGUs with timing data.
    #[expect(dead_code, reason = "useful for debugging and logging")]
    pub fn cgu_count(&self) -> usize {
        self.cgu_costs.len()
    }
}

/// Extract crate name from profile filename.
///
/// Profile files are named `crate_name-XXXXXXX` where X is hex digits.
/// Returns the crate name portion.
fn extract_crate_name(stem: &str) -> String {
    // Find the last hyphen followed by hex digits.
    if let Some(last_hyphen) = stem.rfind('-') {
        let suffix = &stem[last_hyphen + 1..];
        // Check if suffix looks like hex digits (profile ID).
        if suffix.chars().all(|c| c.is_ascii_hexdigit()) {
            return stem[..last_hyphen].to_string();
        }
    }
    // Fallback: use the whole stem.
    stem.to_string()
}

/// Categorize a self-profile event label into frontend, backend, or overhead.
fn categorize_event(label: &str) -> EventCategory {
    // Backend events (LLVM codegen, parallel across CGUs).
    if label.starts_with("LLVM_")
        || label.starts_with("codegen_module")
        || label == "LLVM_lto_optimize"
        || label == "LLVM_thin_lto_import"
    {
        return EventCategory::Backend;
    }

    // Overhead events (per-crate fixed costs).
    if is_linking_event(label) || is_metadata_event(label) {
        return EventCategory::Overhead;
    }

    // Skip internal/bookkeeping events.
    if label.starts_with("incr_comp_")
        || label.starts_with("self_profile")
        || label == "serialize_dep_graph"
        || label == "serialize_work_products"
        || label == "copy_all_cgu_workproducts_to_incr_comp_cache"
    {
        return EventCategory::Skip;
    }

    // Everything else is frontend work (parsing, type checking, etc.).
    EventCategory::Frontend
}

/// Check if an event label is a linking event.
fn is_linking_event(label: &str) -> bool {
    label == "link_crate"
        || label == "link_binary"
        || label == "link_rlib"
        || label.starts_with("link_")
}

/// Check if an event label is a metadata generation event.
fn is_metadata_event(label: &str) -> bool {
    label == "generate_crate_metadata" || label.starts_with("metadata_")
}

/// Normalize a `DefPath` from self-profile output for frontend costs.
///
/// Self-profile paths use internal naming conventions that we normalize:
/// - `_[N]` for anonymous items (derive macro generated code)
/// - `{{impl}}` for impl blocks
/// - `{{impl}}[N]` for multiple impls
/// - `{{closure}}` for closures
///
/// We aggregate timing to match how we collapse dependencies in extraction:
/// - Impl methods → impl block: `crate::Type::{{impl}}::method` → `crate::Type::{{impl}}`
/// - Closures → parent function: `crate::foo::{{closure}}` → `crate::foo`
/// - Nested closures → outermost function: `crate::foo::{{closure}}::{{closure}}` → `crate::foo`
fn normalize_frontend_path(path: &str) -> String {
    // Skip paths that don't look like DefPaths (no :: separator).
    if !path.contains("::") {
        return String::new();
    }

    // Skip internal/generated paths that start with compiler internals.
    if path.starts_with("PseudoCanonicalInput")
        || path.starts_with("LocalModDefId")
        || path.starts_with("core::")
        || path.starts_with("std::")
        || path.starts_with("alloc::")
    {
        return String::new();
    }

    // First, aggregate closures to their parent. Closures are marked with
    // `{{closure}}` and may be nested, so we find the FIRST occurrence and
    // truncate before it.
    // Examples:
    // - `crate::foo::{{closure}}` → `crate::foo`
    // - `crate::foo::{{closure}}[0]` → `crate::foo`
    // - `crate::foo::{{closure}}::{{closure}}` → `crate::foo`
    let path = if let Some(closure_pos) = path.find("::{{closure}}") {
        &path[..closure_pos]
    } else {
        path
    };

    // Aggregate to impl block level: truncate after {{impl}} or {{impl}}[N].
    // This matches our dependency collapsing behavior.
    if let Some(impl_pos) = path.find("{{impl}}") {
        let end = impl_pos + "{{impl}}".len();
        // Check for {{impl}}[N] suffix.
        let rest = &path[end..];
        if rest.starts_with('[')
            && let Some(bracket_end) = rest.find(']')
        {
            return path[..=(end + bracket_end)].to_string();
        }
        return path[..end].to_string();
    }

    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_frontend_path_keeps_crate_paths() {
        assert_eq!(
            normalize_frontend_path("my_crate::foo::bar"),
            "my_crate::foo::bar"
        );
    }

    #[test]
    fn test_normalize_frontend_path_skips_std() {
        assert_eq!(normalize_frontend_path("std::vec::Vec"), "");
        assert_eq!(normalize_frontend_path("core::fmt::Debug"), "");
    }

    #[test]
    fn test_normalize_frontend_path_skips_non_paths() {
        assert_eq!(normalize_frontend_path("()"), "");
        assert_eq!(normalize_frontend_path("<no path>"), "");
    }

    #[test]
    fn test_normalize_frontend_path_aggregates_to_impl() {
        // Method inside impl -> aggregate to impl.
        assert_eq!(
            normalize_frontend_path("my_crate::Type::{{impl}}::method"),
            "my_crate::Type::{{impl}}"
        );

        // Nested items inside impl -> aggregate to impl.
        assert_eq!(
            normalize_frontend_path(
                "my_crate::Type::{{impl}}::deserialize::visit_map"
            ),
            "my_crate::Type::{{impl}}"
        );

        // Derive-generated impl with anonymous block.
        assert_eq!(
            normalize_frontend_path(
                "my_crate::module::_[7]::{{impl}}::deserialize"
            ),
            "my_crate::module::_[7]::{{impl}}"
        );

        // Numbered impl (multiple impls on same type).
        assert_eq!(
            normalize_frontend_path("my_crate::Type::{{impl}}[1]::method"),
            "my_crate::Type::{{impl}}[1]"
        );
    }

    #[test]
    fn test_normalize_frontend_path_keeps_bare_impl() {
        // Bare impl without methods - keep as-is.
        assert_eq!(
            normalize_frontend_path("my_crate::Type::{{impl}}"),
            "my_crate::Type::{{impl}}"
        );
    }

    #[test]
    fn test_normalize_frontend_path_aggregates_closures() {
        // Closure in function -> aggregate to function.
        assert_eq!(
            normalize_frontend_path("my_crate::foo::{{closure}}"),
            "my_crate::foo"
        );

        // Closure with disambiguation index.
        assert_eq!(
            normalize_frontend_path("my_crate::foo::{{closure}}[0]"),
            "my_crate::foo"
        );

        // Nested closures -> aggregate to outermost function.
        assert_eq!(
            normalize_frontend_path("my_crate::foo::{{closure}}::{{closure}}"),
            "my_crate::foo"
        );

        // Closure inside impl method -> aggregate to impl block.
        // First closure is removed, then impl method is aggregated to impl.
        assert_eq!(
            normalize_frontend_path(
                "my_crate::Type::{{impl}}::method::{{closure}}"
            ),
            "my_crate::Type::{{impl}}"
        );
    }

    #[test]
    fn test_categorize_event_frontend() {
        assert_eq!(categorize_event("typeck"), EventCategory::Frontend);
        assert_eq!(categorize_event("borrowck"), EventCategory::Frontend);
        assert_eq!(categorize_event("optimized_mir"), EventCategory::Frontend);
    }

    #[test]
    fn test_categorize_event_backend() {
        assert_eq!(
            categorize_event("LLVM_module_codegen"),
            EventCategory::Backend
        );
        assert_eq!(categorize_event("codegen_module"), EventCategory::Backend);
    }

    #[test]
    fn test_categorize_event_overhead() {
        assert_eq!(categorize_event("link_crate"), EventCategory::Overhead);
        assert_eq!(categorize_event("link_binary"), EventCategory::Overhead);
        assert_eq!(
            categorize_event("generate_crate_metadata"),
            EventCategory::Overhead
        );
    }

    #[test]
    fn test_categorize_event_skip() {
        assert_eq!(
            categorize_event("incr_comp_persist_result"),
            EventCategory::Skip
        );
        assert_eq!(
            categorize_event("self_profile_alloc_query_strings"),
            EventCategory::Skip
        );
    }

    #[test]
    fn test_extract_crate_name() {
        assert_eq!(
            extract_crate_name("tarjanize_schemas-0060816"),
            "tarjanize_schemas"
        );
        assert_eq!(extract_crate_name("my-crate-name-abc123"), "my-crate-name");
        // No hex suffix - use whole name.
        assert_eq!(extract_crate_name("simple_crate"), "simple_crate");
    }
}
