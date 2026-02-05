//! Self-profile data parsing for accurate cost estimation.
//!
//! This module parses rustc's `-Zself-profile` output to get real compilation
//! time per symbol. The approach:
//!
//! 1. Run a profiled build with `-Zself-profile` and `-Zself-profile-events=default,llvm,args`
//! 2. Parse the `.mm_profdata` files using the `analyzeme` crate
//! 3. Categorize events into frontend, backend (CGU), and overhead costs
//! 4. Compute **self-time** (excluding nested queries) to avoid double-counting
//! 5. Use this timing data for the cost fields in `Symbol` and `Crate`
//!
//! ## Self-Time Computation
//!
//! Profile events can nest: `typeck(foo)` may trigger `typeck(bar)`. Naively
//! summing durations would double-count `bar`'s time (once for bar, once inside
//! foo). We compute self-time = duration - sum(children) to get the time each
//! symbol directly consumes.
//!
//! The algorithm walks events in reverse order (events are emitted at their end
//! time, so parents come after children). For each event, we subtract its
//! duration from its parent's self-time.
//!
//! ## Event Categories
//!
//! - **Frontend**: Serial compilation phases (parsing, type checking, borrow checking)
//! - **Backend**: LLVM codegen, parallelizable across CGUs
//! - **Overhead**: Per-crate fixed costs (linking, metadata generation)

use std::collections::HashMap;
use std::panic;
use std::path::Path;
use std::time::Duration;

use analyzeme::{Event, EventPayload, ProfilingData, Timestamp};
use tracing::{debug, info, info_span, warn};

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
        let _span = info_span!("load_from_dir", dir = %dir.display()).entered();

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

        // Collect all profile files first for logging.
        let profile_files: Vec<_> = entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "mm_profdata") {
                    path.file_name().map(|n| n.to_string_lossy().into_owned())
                } else {
                    None
                }
            })
            .collect();

        info!(
            dir = %dir.display(),
            file_count = profile_files.len(),
            files = ?profile_files,
            "found profile files"
        );

        let mut file_count = 0;
        let mut event_count = 0;

        // Re-read directory to process files (iterator was consumed).
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) => {
                warn!(
                    "failed to re-read profile directory {}: {}",
                    dir.display(),
                    e
                );
                return data;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();

            // Profile files have the .mm_profdata extension.
            if path.extension().is_some_and(|ext| ext == "mm_profdata") {
                // The stem is the path we pass to ProfilingData::new.
                // e.g., "tarjanize_schemas-0060816.mm_profdata" -> "tarjanize_schemas-0060816"
                if let Some(stem) = path.file_stem() {
                    let stem_str = stem.to_string_lossy();
                    let stem_path = dir.join(stem);

                    // Use catch_unwind because decodeme can panic on corrupted
                    // profile data (e.g., truncated string tables).
                    let stem_path_clone = stem_path.clone();
                    let result = panic::catch_unwind(|| {
                        ProfilingData::new(&stem_path_clone)
                    });

                    match result {
                        Ok(Ok(profile)) => {
                            // Extract crate name from profile filename.
                            // Format: "crate_name-XXXXXXX" where X is hex digits.
                            let crate_name = extract_crate_name(&stem_str);

                            let _span = info_span!(
                                "aggregate_profile",
                                file = %stem_str,
                                crate_name = %crate_name,
                                num_events = profile.num_events()
                            )
                            .entered();

                            file_count += 1;
                            event_count +=
                                data.aggregate_profile(&profile, &crate_name);
                        }
                        Ok(Err(e)) => {
                            debug!(
                                "failed to load profile {}: {}",
                                stem_path.display(),
                                e
                            );
                        }
                        Err(_) => {
                            warn!(
                                "profile data corrupted (parser panic): {}",
                                stem_path.display()
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

    /// Aggregate timing data from a single profile file using self-time.
    ///
    /// Self-time is the time spent in an event excluding time spent in nested
    /// child events. This avoids double-counting when summing costs, since
    /// query events can nest (e.g., `typeck(foo)` triggers `typeck(bar)`).
    ///
    /// The algorithm walks events in reverse order (events are emitted at their
    /// end time, so parents come after children in the stream). For each event,
    /// we subtract its duration from the parent's accumulated self-time.
    ///
    /// Returns the number of events processed.
    fn aggregate_profile(
        &mut self,
        profile: &ProfilingData,
        crate_name: &str,
    ) -> usize {
        // Per-thread state: stack of currently open events and their
        // accumulated self-times. We need to track self-time separately because
        // we modify it as we encounter children.
        struct PerThreadState<'a> {
            /// Stack of open events paired with their accumulated self-time.
            /// Self-time starts at full duration and decreases for children.
            stack: Vec<(Event<'a>, Duration)>,
        }

        let mut threads: HashMap<u32, PerThreadState<'_>> = HashMap::new();
        let mut count = 0;
        let mut raw_duration_sum = Duration::ZERO;
        let mut recorded_self_time_sum = Duration::ZERO;

        // Walk events in reverse order. Events are emitted at their end time,
        // so a parent event appears after all its children in the stream.
        // Walking backwards means we see parents before children.
        for event in profile.iter_full().rev() {
            // Skip non-interval events (instants, integers).
            let EventPayload::Timestamp(Timestamp::Interval { .. }) =
                event.payload
            else {
                continue;
            };

            let Some(duration) = event.duration() else {
                continue;
            };

            // Track raw duration for debugging
            if &*event.event_kind == QUERY_EVENT_KIND
                || &*event.event_kind == GENERIC_ACTIVITY_EVENT_KIND
            {
                raw_duration_sum += duration;
            }

            let thread = threads
                .entry(event.thread_id)
                .or_insert_with(|| PerThreadState { stack: Vec::new() });

            // Pop events from the stack that don't contain the current event.
            // After this loop, the top of the stack (if any) is the parent of
            // the current event.
            while let Some((top_event, top_self_time)) =
                thread.stack.last().cloned()
            {
                if top_event.contains(&event) {
                    // Top event is parent of current event - keep it.
                    break;
                }

                // Top event ended before current event started - finalize it.
                thread.stack.pop();
                if let Some(recorded) =
                    self.record_event(&top_event, top_self_time, crate_name)
                {
                    recorded_self_time_sum += recorded;
                }
                count += 1;
            }

            // Subtract current event's duration from parent's self-time.
            if let Some((_, parent_self_time)) = thread.stack.last_mut() {
                *parent_self_time = parent_self_time.saturating_sub(duration);
            }

            // Push current event with initial self-time = full duration.
            // This will be reduced as we encounter its children.
            thread.stack.push((event, duration));
        }

        // Finalize any remaining events on the stacks.
        for (_, thread) in threads {
            for (event, self_time) in thread.stack {
                if let Some(recorded) =
                    self.record_event(&event, self_time, crate_name)
                {
                    recorded_self_time_sum += recorded;
                }
                count += 1;
            }
        }

        // Log the inflation ratio for debugging
        info!(
            raw_ms = raw_duration_sum.as_millis(),
            self_time_ms = recorded_self_time_sum.as_millis(),
            "self-time sums"
        );

        count
    }

    /// Record an event's self-time into the appropriate cost map.
    ///
    /// Returns the recorded self-time if the event was recorded as Frontend,
    /// otherwise None.
    fn record_event(
        &mut self,
        event: &Event<'_>,
        self_time: Duration,
        crate_name: &str,
    ) -> Option<Duration> {
        let label = &event.label;
        let event_kind = &event.event_kind;
        let category = categorize_event(label, event_kind);

        match category {
            EventCategory::Frontend => {
                // Frontend events have DefPath in additional_data[0].
                if let Some(raw_path) = event.additional_data.first() {
                    let normalized = normalize_frontend_path(raw_path);
                    if !normalized.is_empty() {
                        *self.frontend_costs.entry(normalized).or_default() +=
                            self_time;
                        return Some(self_time);
                    }
                }
                None
            }
            EventCategory::Backend => {
                // Backend events (LLVM) have CGU name in additional_data[0].
                if let Some(cgu_name) = event.additional_data.first() {
                    *self.cgu_costs.entry(cgu_name.to_string()).or_default() +=
                        self_time;
                }
                None
            }
            EventCategory::Overhead => {
                // Overhead events are per-crate fixed costs.
                // We only track metadata generation; linking is too small
                // to matter (< 1% of lib build time) and unpredictable.
                if is_metadata_event(label) {
                    let overhead = self
                        .crate_overhead
                        .entry(crate_name.to_string())
                        .or_default();
                    overhead.metadata_ms += self_time.as_millis_f64();
                }
                None
            }
            EventCategory::Skip => None,
        }
    }

    /// Get the frontend compilation cost for a symbol path.
    ///
    /// Returns the total time in milliseconds, or None if no timing data exists.
    /// Normalizes the path by replacing hyphens with underscores in the crate
    /// name prefix, since Rust crate names use underscores but cargo package
    /// names use hyphens.
    pub fn get_frontend_cost_ms(&self, path: &str) -> Option<f64> {
        // Profile paths use underscores (Rust convention), but our paths use
        // cargo package names with hyphens. Normalize the crate name prefix.
        let normalized = path.replace('-', "_");
        self.frontend_costs
            .get(&normalized)
            .map(Duration::as_millis_f64)
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
    /// Normalizes the crate name by replacing hyphens with underscores,
    /// since Rust crate names use underscores but cargo package names use hyphens.
    pub fn get_crate_overhead(
        &self,
        crate_name: &str,
    ) -> Option<&CrateOverhead> {
        // Profile filenames use underscores (Rust convention), but cargo
        // package names use hyphens. Normalize to match.
        let normalized = crate_name.replace('-', "_");
        self.crate_overhead.get(&normalized)
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

/// Event kinds from rustc's self-profile that we care about.
const QUERY_EVENT_KIND: &str = "Query";
const GENERIC_ACTIVITY_EVENT_KIND: &str = "GenericActivity";

/// Categorize a self-profile event into frontend, backend, or overhead.
///
/// We filter by both `event_kind` and `label`:
/// - `event_kind` tells us the category (Query, etc.)
/// - `label` tells us the specific operation (typeck, etc.)
///
/// We only count Query and generic activity events as frontend work.
/// Other event kinds (incremental hashing, load result) happen INSIDE queries
/// and are already accounted for in the query's time.
fn categorize_event(label: &str, event_kind: &str) -> EventCategory {
    // Backend events (LLVM codegen, parallel across CGUs).
    // These are GenericActivity events with LLVM-related labels.
    if label.starts_with("LLVM_")
        || label.starts_with("codegen_module")
        || label == "LLVM_lto_optimize"
        || label == "LLVM_thin_lto_import"
    {
        return EventCategory::Backend;
    }

    // Overhead events (per-crate fixed costs).
    // Only metadata matters; linking is < 1% of lib build time.
    if is_metadata_event(label) {
        return EventCategory::Overhead;
    }

    // Skip internal/bookkeeping events and linking (too small to matter).
    if label.starts_with("incr_comp_")
        || label.starts_with("self_profile")
        || label.starts_with("link_")
        || label == "serialize_dep_graph"
        || label == "serialize_work_products"
        || label == "copy_all_cgu_workproducts_to_incr_comp_cache"
    {
        return EventCategory::Skip;
    }

    // Only count Query and GenericActivity events as frontend work.
    // Other event kinds (IncrementalResultHashing, IncrementalLoadResult, etc.)
    // happen inside queries and are already accounted for in the parent's time.
    if event_kind != QUERY_EVENT_KIND
        && event_kind != GENERIC_ACTIVITY_EVENT_KIND
    {
        return EventCategory::Skip;
    }

    // Everything else is frontend work (parsing, type checking, etc.).
    EventCategory::Frontend
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
        // Query events with typeck/borrowck labels are frontend.
        assert_eq!(
            categorize_event("typeck", QUERY_EVENT_KIND),
            EventCategory::Frontend
        );
        assert_eq!(
            categorize_event("borrowck", QUERY_EVENT_KIND),
            EventCategory::Frontend
        );
        assert_eq!(
            categorize_event("optimized_mir", QUERY_EVENT_KIND),
            EventCategory::Frontend
        );
        // GenericActivity events are also frontend.
        assert_eq!(
            categorize_event("some_activity", GENERIC_ACTIVITY_EVENT_KIND),
            EventCategory::Frontend
        );
    }

    #[test]
    fn test_categorize_event_backend() {
        assert_eq!(
            categorize_event(
                "LLVM_module_codegen",
                GENERIC_ACTIVITY_EVENT_KIND
            ),
            EventCategory::Backend
        );
        assert_eq!(
            categorize_event("codegen_module", GENERIC_ACTIVITY_EVENT_KIND),
            EventCategory::Backend
        );
    }

    #[test]
    fn test_categorize_event_overhead() {
        // Only metadata events are tracked as overhead; linking is skipped.
        assert_eq!(
            categorize_event(
                "generate_crate_metadata",
                GENERIC_ACTIVITY_EVENT_KIND
            ),
            EventCategory::Overhead
        );
        assert_eq!(
            categorize_event(
                "metadata_decode_entry",
                GENERIC_ACTIVITY_EVENT_KIND
            ),
            EventCategory::Overhead
        );
    }

    #[test]
    fn test_categorize_event_skip() {
        // Incremental compilation events are skipped.
        assert_eq!(
            categorize_event(
                "incr_comp_persist_result",
                GENERIC_ACTIVITY_EVENT_KIND
            ),
            EventCategory::Skip
        );
        assert_eq!(
            categorize_event(
                "self_profile_alloc_query_strings",
                GENERIC_ACTIVITY_EVENT_KIND
            ),
            EventCategory::Skip
        );
        // Linking events are skipped (< 1% of lib build time).
        assert_eq!(
            categorize_event("link_crate", GENERIC_ACTIVITY_EVENT_KIND),
            EventCategory::Skip
        );
        assert_eq!(
            categorize_event("link_binary", GENERIC_ACTIVITY_EVENT_KIND),
            EventCategory::Skip
        );
        // Non-Query/GenericActivity event kinds are skipped.
        assert_eq!(
            categorize_event("typeck", "IncrementalResultHashing"),
            EventCategory::Skip
        );
        assert_eq!(
            categorize_event("typeck", "IncrementalLoadResult"),
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
