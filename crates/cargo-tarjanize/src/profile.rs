//! Self-profile data parsing for accurate cost estimation.
//!
//! This module parses rustc's `-Zself-profile` output to get real compilation
//! time per symbol. The approach:
//!
//! 1. Run a profiled build with `-Zself-profile` and `-Zself-profile-events=default,args`
//! 2. Parse the `.mm_profdata` files using the `analyzeme` crate
//! 3. Aggregate timing data by `DefPath`
//! 4. Use this timing data for the `cost` field in `Symbol`
//!
//! The self-profile data includes timing for queries like `typeck`, `optimized_mir`,
//! and `codegen_fn`, each annotated with the `DefPath` of the item being processed.
//! By summing all query times for each `DefPath`, we get a good approximation of
//! the total compilation cost for that symbol.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use analyzeme::ProfilingData;
use tracing::{debug, info, warn};

/// Aggregated timing data from self-profile output.
///
/// Maps normalized `DefPath`s to total compilation time.
#[derive(Debug, Default)]
pub struct ProfileData {
    /// Total time spent on each symbol, indexed by normalized path.
    timings: HashMap<String, Duration>,
}

impl ProfileData {
    /// Load profile data from a directory containing `.mm_profdata` files.
    ///
    /// Parses all profile files in the directory and aggregates timing data
    /// by `DefPath`. Returns an empty `ProfileData` if the directory doesn't exist
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

            // Profile files have the .mm_profdata extension
            if path.extension().is_some_and(|ext| ext == "mm_profdata") {
                // The stem is the path we pass to ProfilingData::new
                // e.g., "tarjanize_schemas-0060816.mm_profdata" -> "tarjanize_schemas-0060816"
                if let Some(stem) = path.file_stem() {
                    let stem_path = dir.join(stem);
                    match ProfilingData::new(&stem_path) {
                        Ok(profile) => {
                            file_count += 1;
                            event_count += data.aggregate_profile(&profile);
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
            unique_paths = data.timings.len(),
            "loaded self-profile data"
        );

        data
    }

    /// Aggregate timing data from a single profile file.
    ///
    /// Returns the number of events processed.
    fn aggregate_profile(&mut self, profile: &ProfilingData) -> usize {
        let mut count = 0;

        for event in profile.iter_full() {
            if let Some(dur) = event.duration() {
                // The DefPath is in additional_data[0] for most queries.
                // If there's no path, skip this event.
                if let Some(raw_path) = event.additional_data.first() {
                    let normalized = normalize_path(raw_path);
                    if !normalized.is_empty() {
                        *self.timings.entry(normalized).or_default() += dur;
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Get the compilation cost for a symbol path.
    ///
    /// Returns the total time in milliseconds, or None if no timing data exists.
    pub fn get_cost_ms(&self, path: &str) -> Option<f64> {
        self.timings.get(path).map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Get the number of unique paths with timing data.
    pub fn len(&self) -> usize {
        self.timings.len()
    }
}

/// Normalize a `DefPath` from self-profile output to match our extracted paths.
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
fn normalize_path(path: &str) -> String {
    // Skip paths that don't look like DefPaths (no :: separator)
    if !path.contains("::") {
        return String::new();
    }

    // Skip internal/generated paths that start with compiler internals
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

    // Aggregate to impl block level: truncate after {{impl}} or {{impl}}[N]
    // This matches our dependency collapsing behavior.
    if let Some(impl_pos) = path.find("{{impl}}") {
        let end = impl_pos + "{{impl}}".len();
        // Check for {{impl}}[N] suffix
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
    fn test_normalize_path_keeps_crate_paths() {
        assert_eq!(normalize_path("my_crate::foo::bar"), "my_crate::foo::bar");
    }

    #[test]
    fn test_normalize_path_skips_std() {
        assert_eq!(normalize_path("std::vec::Vec"), "");
        assert_eq!(normalize_path("core::fmt::Debug"), "");
    }

    #[test]
    fn test_normalize_path_skips_non_paths() {
        assert_eq!(normalize_path("()"), "");
        assert_eq!(normalize_path("<no path>"), "");
    }

    #[test]
    fn test_normalize_path_aggregates_to_impl() {
        // Method inside impl -> aggregate to impl
        assert_eq!(
            normalize_path("my_crate::Type::{{impl}}::method"),
            "my_crate::Type::{{impl}}"
        );

        // Nested items inside impl -> aggregate to impl
        assert_eq!(
            normalize_path("my_crate::Type::{{impl}}::deserialize::visit_map"),
            "my_crate::Type::{{impl}}"
        );

        // Derive-generated impl with anonymous block
        assert_eq!(
            normalize_path("my_crate::module::_[7]::{{impl}}::deserialize"),
            "my_crate::module::_[7]::{{impl}}"
        );

        // Numbered impl (multiple impls on same type)
        assert_eq!(
            normalize_path("my_crate::Type::{{impl}}[1]::method"),
            "my_crate::Type::{{impl}}[1]"
        );
    }

    #[test]
    fn test_normalize_path_keeps_bare_impl() {
        // Bare impl without methods - keep as-is
        assert_eq!(
            normalize_path("my_crate::Type::{{impl}}"),
            "my_crate::Type::{{impl}}"
        );
    }

    #[test]
    fn test_normalize_path_aggregates_closures() {
        // Closure in function -> aggregate to function
        assert_eq!(
            normalize_path("my_crate::foo::{{closure}}"),
            "my_crate::foo"
        );

        // Closure with disambiguation index
        assert_eq!(
            normalize_path("my_crate::foo::{{closure}}[0]"),
            "my_crate::foo"
        );

        // Nested closures -> aggregate to outermost function
        assert_eq!(
            normalize_path("my_crate::foo::{{closure}}::{{closure}}"),
            "my_crate::foo"
        );

        // Closure inside impl method -> aggregate to impl block
        // First closure is removed, then impl method is aggregated to impl
        assert_eq!(
            normalize_path("my_crate::Type::{{impl}}::method::{{closure}}"),
            "my_crate::Type::{{impl}}"
        );
    }
}
