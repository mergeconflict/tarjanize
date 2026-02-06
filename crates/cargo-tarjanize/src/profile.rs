//! Self-profile data parsing for accurate cost estimation.
//!
//! This module parses rustc's `-Zself-profile` output to get real compilation
//! time per symbol. The approach:
//!
//! 1. Run `cargo check` with `-Zself-profile` and `-Zself-profile-events=default,args`
//! 2. Parse the `.mm_profdata` files using the `analyzeme` crate
//! 3. Compute **self-time** (excluding nested queries) to avoid double-counting
//! 4. Attribute self-time to symbols when possible, or to `event_times_ms` otherwise
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
//! ## No-Double-Counting Event Accounting
//!
//! Each event's self-time lands in exactly one place — never both:
//! - Events with a usable `DefPath` go to `frontend_costs` (per-symbol,
//!   keyed by normalized `DefPath` then event label).
//! - Events without a usable `DefPath` go to `event_times_ms` (target-level,
//!   keyed by event label).
//!
//! This ensures `sum(target.event_times_ms) + sum(all symbol event_times_ms)`
//! equals total self-time, with no double-counting.
//!
//! Wall-clock `wall_time_ms` is the span of all profiled events
//! (min start to max end). We use `cargo check` so only frontend events
//! are profiled — no LLVM, codegen, or linking.

use std::collections::HashMap;
use std::panic;
use std::path::Path;
use std::time::{Duration, SystemTime};

use analyzeme::{Event, EventPayload, ProfilingData, Timestamp};
use tarjanize_schemas::TargetTimings;
use tracing::{debug, info, info_span, warn};

/// Aggregated timing data from self-profile output.
///
/// Contains frontend costs (by `DefPath` and event label) and target-level
/// timings (wall-clock span and unattributed event times).
#[derive(Debug, Default)]
pub struct ProfileData {
    /// Frontend costs indexed by normalized `DefPath`, then by event label.
    /// Outer key = normalized `DefPath`, inner key = event label, value = self-time.
    /// Only events with a usable `DefPath` are stored here.
    frontend_costs: HashMap<String, HashMap<String, Duration>>,
    /// Target timings indexed by crate name.
    target_timings: HashMap<String, TargetTimings>,
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
            target_count = data.target_timings.len(),
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
    /// Each event's self-time lands in exactly one place:
    /// - Events with a usable `DefPath` go to `frontend_costs` (per-symbol).
    /// - Events without a usable `DefPath` go to `event_times_ms` (target-level).
    ///
    /// Wall-clock `wall_time_ms` is the span of all profiled events
    /// (min start to max end across all events).
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

        // Track min start / max end timestamps for wall-clock.
        // All events contribute to a single wall-clock span. Unattributed
        // events (including backend) go to event_times_ms on the target.
        let mut min_start: Option<SystemTime> = None;
        let mut max_end: Option<SystemTime> = None;

        // Walk events in reverse order. Events are emitted at their end time,
        // so a parent event appears after all its children in the stream.
        // Walking backwards means we see parents before children.
        for event in profile.iter_full().rev() {
            // Skip non-interval events (instants, integers).
            let EventPayload::Timestamp(Timestamp::Interval { start, end }) =
                event.payload
            else {
                continue;
            };

            let Some(duration) = event.duration() else {
                continue;
            };

            // Track wall-clock interval across all events.
            min_start = Some(min_start.map_or(start, |v| v.min(start)));
            max_end = Some(max_end.map_or(end, |v| v.max(end)));

            // Track raw duration for debugging.
            raw_duration_sum += duration;

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

        // Compute wall-clock duration from min/max timestamps.
        let timings = self
            .target_timings
            .entry(crate_name.to_string())
            .or_default();
        timings.wall_time_ms += duration_between(min_start, max_end);

        // Log the inflation ratio for debugging.
        info!(
            raw_ms = raw_duration_sum.as_millis(),
            self_time_ms = recorded_self_time_sum.as_millis(),
            wall_time_ms = timings.wall_time_ms,
            "self-time sums"
        );

        count
    }

    /// Record an event's self-time into exactly one cost map.
    ///
    /// Each event lands in exactly one place — no double-counting:
    /// - Events with a usable `DefPath` go to `frontend_costs` (per-symbol,
    ///   keyed by normalized `DefPath` then event label).
    /// - Events without a usable `DefPath` go to `event_times_ms` (target-level,
    ///   keyed by event label).
    ///
    /// Returns the recorded self-time if the event was attributed to a symbol,
    /// otherwise None.
    fn record_event(
        &mut self,
        event: &Event<'_>,
        self_time: Duration,
        crate_name: &str,
    ) -> Option<Duration> {
        let label = &*event.label;

        // Try to attribute to a symbol via DefPath.
        if let Some(raw_path) = event.additional_data.first() {
            let normalized = normalize_frontend_path(raw_path);
            if !normalized.is_empty() {
                // Attributed event: record ONLY in frontend_costs (per-symbol).
                *self
                    .frontend_costs
                    .entry(normalized)
                    .or_default()
                    .entry(label.to_string())
                    .or_default() += self_time;
                return Some(self_time);
            }
        }

        // Unattributed event: record ONLY in event_times_ms (target-level).
        let timings = self
            .target_timings
            .entry(crate_name.to_string())
            .or_default();
        *timings.event_times_ms.entry(label.to_string()).or_default() +=
            self_time.as_millis_f64();

        None
    }

    /// Get per-event compilation times for a symbol path.
    ///
    /// Returns a map of event label to time in milliseconds, or None if no
    /// timing data exists. Normalizes the path by replacing hyphens with
    /// underscores in the crate name prefix, since Rust crate names use
    /// underscores but cargo package names use hyphens.
    pub fn get_event_times_ms(
        &self,
        path: &str,
    ) -> Option<HashMap<String, f64>> {
        // Profile paths use underscores (Rust convention), but our paths use
        // cargo package names with hyphens. Normalize the crate name prefix.
        let normalized = path.replace('-', "_");
        self.frontend_costs.get(&normalized).map(|inner| {
            inner
                .iter()
                .map(|(label, dur)| (label.clone(), dur.as_millis_f64()))
                .collect()
        })
    }

    /// Get target timings (wall-clock time and unattributed event times).
    ///
    /// Returns None if no timings were recorded for this crate.
    /// Normalizes the crate name by replacing hyphens with underscores,
    /// since Rust crate names use underscores but cargo package names use hyphens.
    pub fn get_target_timings(
        &self,
        crate_name: &str,
    ) -> Option<&TargetTimings> {
        // Profile filenames use underscores (Rust convention), but cargo
        // package names use hyphens. Normalize to match.
        let normalized = crate_name.replace('-', "_");
        self.target_timings.get(&normalized)
    }

    /// Get the number of unique frontend paths with timing data.
    #[expect(dead_code, reason = "useful for debugging and logging")]
    pub fn frontend_count(&self) -> usize {
        self.frontend_costs.len()
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

/// Compute the duration in milliseconds between two optional `SystemTime` values.
///
/// Returns 0.0 if either value is `None` (no events of that category were seen)
/// or if `end` is before `start`.
fn duration_between(
    min_start: Option<SystemTime>,
    max_end: Option<SystemTime>,
) -> f64 {
    match (min_start, max_end) {
        (Some(start), Some(end)) => end
            .duration_since(start)
            .unwrap_or(Duration::ZERO)
            .as_millis_f64(),
        _ => 0.0,
    }
}

/// Normalize a `DefPath` from self-profile output for frontend costs.
///
/// Self-profile paths use internal naming conventions that we normalize to
/// aggregate descendant `DefPaths` up to their nearest ancestor symbol. This
/// is critical for accurate per-symbol cost attribution: per-item events
/// like `predicates_of`, `generics_of`, `type_of`, and `def_span` fire
/// once per `DefId` including all descendants (closures, generic params,
/// anonymous consts, opaque types, constructors, coroutine bodies). Without
/// this aggregation, those events are stored under keys that match nothing
/// in the extracted symbol map and are silently lost.
///
/// The function iteratively peels segments from the end of the path until
/// it reaches a segment that could be a real symbol name:
///
/// - `{{closure}}[N]` → closures (may nest)
/// - `{{opaque}}[N]` → async fn future types
/// - `{{constructor}}` → enum/struct constructors
/// - `{{coroutine}}` → coroutine bodies
/// - `_[N]` or `_` → anonymous consts
/// - Single-char uppercase idents or `'_[N]` → generic/lifetime params
/// - Anything after `{{impl}}[N]` → impl methods aggregate to impl block
fn normalize_frontend_path(path: &str) -> String {
    // Skip paths that don't look like DefPaths (no :: separator).
    if !path.contains("::") {
        return String::new();
    }

    // Skip compiler-internal pseudo-paths that aren't real symbols.
    // These are compiler query keys, not DefPaths we can attribute to symbols.
    if path.starts_with("PseudoCanonicalInput")
        || path.starts_with("LocalModDefId")
    {
        return String::new();
    }

    // Iteratively peel descendant segments from the end. We loop because
    // descendants can nest (e.g., `foo::{{opaque}}::{{closure}}`).
    let mut current = path;
    while let Some(sep_pos) = current.rfind("::") {
        let last_segment = &current[sep_pos + 2..];

        // Check if this is an internal `DefPath` segment that should be peeled.
        if is_descendant_segment(last_segment) {
            current = &current[..sep_pos];
            continue;
        }

        // Special case: if a PARENT segment is `{{impl}}[N]` or `{{impl}}`,
        // this is an impl method — truncate to the impl block.
        // Only check the portion before the last `::` separator to avoid
        // matching when `{{impl}}` IS the last segment itself.
        let prefix = &current[..sep_pos];
        if let Some(impl_pos) = prefix.rfind("::{{impl}}") {
            let impl_start = impl_pos + 2; // skip `::`
            let impl_end = impl_start + "{{impl}}".len();
            let rest_after_impl = &prefix[impl_end..];

            // Check if the rest between {{impl}} and the last segment is
            // either empty (direct method) or starts with [N] (numbered impl).
            if rest_after_impl.is_empty() {
                // `foo::{{impl}}::method` → `foo::{{impl}}`
                return current[..impl_end].to_string();
            }
            if rest_after_impl.starts_with('[') && rest_after_impl.contains(']')
            {
                // `foo::{{impl}}[N]::method` → `foo::{{impl}}[N]`
                let bracket_end = impl_end
                    + rest_after_impl
                        .find(']')
                        .expect("bracket_end must exist")
                    + 1;
                return current[..bracket_end].to_string();
            }
        }

        // Not a descendant segment — we've reached the symbol level.
        break;
    }

    // After peeling, check if we ended on an `{{impl}}[N]` — keep it.
    if let Some(impl_pos) = current.find("{{impl}}") {
        let end = impl_pos + "{{impl}}".len();
        let rest = &current[end..];
        if rest.starts_with('[')
            && let Some(bracket_end) = rest.find(']')
        {
            return current[..=(end + bracket_end)].to_string();
        }
        return current[..end].to_string();
    }

    current.to_string()
}

/// Returns true if a `DefPath` segment is a compiler-internal descendant
/// that should be peeled off to reach the parent symbol.
///
/// These segments represent entities that can't be split independently
/// from their parent and generate per-item events that should roll up.
fn is_descendant_segment(segment: &str) -> bool {
    // Braced compiler-internal segments: {{closure}}, {{closure}}[N],
    // {{opaque}}, {{opaque}}[N], {{constructor}}, {{coroutine}}.
    // Exclude {{impl}} and {{impl}}[N] — those are symbol-level entities.
    if segment.starts_with("{{") && !segment.starts_with("{{impl}}") {
        return true;
    }

    // Anonymous consts: `_` or `_[N]`.
    if segment == "_" || (segment.starts_with("_[") && segment.ends_with(']')) {
        return true;
    }

    // Lifetime parameters: `'_`, `'_[N]`, `'a`, etc.
    if segment.starts_with('\'') {
        return true;
    }

    // Generic type parameters: single uppercase letter like `T`, `U`, `S`.
    // These are generated by `generics_of` and similar per-item queries.
    // We check for single uppercase ASCII characters — real symbol names
    // are almost never single uppercase letters.
    if segment.len() == 1 && segment.as_bytes()[0].is_ascii_uppercase() {
        return true;
    }

    false
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
    fn test_normalize_frontend_path_keeps_std_paths() {
        // std/core/alloc paths are now kept (they produce non-empty
        // normalized paths). Events on external types that can't be
        // attributed to a local symbol get caught by record_event
        // and stored in event_times_ms instead.
        assert_eq!(normalize_frontend_path("std::vec::Vec"), "std::vec::Vec");
        assert_eq!(
            normalize_frontend_path("core::fmt::Debug"),
            "core::fmt::Debug"
        );
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

    // --- New descendant type tests (generic params, anon consts, opaque, etc.) ---

    #[test]
    fn test_normalize_generic_type_param() {
        // Generic type params like `foo::T` should roll up to `foo`.
        assert_eq!(
            normalize_frontend_path("my_crate::foo::T"),
            "my_crate::foo"
        );
        assert_eq!(
            normalize_frontend_path("my_crate::foo::U"),
            "my_crate::foo"
        );
    }

    #[test]
    fn test_normalize_lifetime_param() {
        // Lifetime params: `foo::'_` and `foo::'_[1]`.
        assert_eq!(
            normalize_frontend_path("my_crate::foo::'_"),
            "my_crate::foo"
        );
        assert_eq!(
            normalize_frontend_path("my_crate::foo::'_[1]"),
            "my_crate::foo"
        );
    }

    #[test]
    fn test_normalize_anonymous_const() {
        // Anonymous consts: `foo::_` and `foo::_[7]`.
        assert_eq!(
            normalize_frontend_path("my_crate::foo::_"),
            "my_crate::foo"
        );
        assert_eq!(
            normalize_frontend_path("my_crate::foo::_[7]"),
            "my_crate::foo"
        );
    }

    #[test]
    fn test_normalize_opaque_type() {
        // Opaque types from async fns.
        assert_eq!(
            normalize_frontend_path("my_crate::foo::{{opaque}}"),
            "my_crate::foo"
        );
        assert_eq!(
            normalize_frontend_path("my_crate::foo::{{opaque}}[2]"),
            "my_crate::foo"
        );
    }

    #[test]
    fn test_normalize_constructor() {
        assert_eq!(
            normalize_frontend_path("my_crate::foo::{{constructor}}"),
            "my_crate::foo"
        );
    }

    #[test]
    fn test_normalize_coroutine() {
        assert_eq!(
            normalize_frontend_path("my_crate::foo::{{coroutine}}"),
            "my_crate::foo"
        );
    }

    #[test]
    fn test_normalize_nested_opaque_closure() {
        // Nesting: `foo::{{opaque}}::{{closure}}` → `foo`.
        assert_eq!(
            normalize_frontend_path("my_crate::foo::{{opaque}}::{{closure}}"),
            "my_crate::foo"
        );
    }

    #[test]
    fn test_normalize_impl_method_with_generic_param() {
        // Impl + descendant: `Type::{{impl}}::method::T` → `Type::{{impl}}`.
        assert_eq!(
            normalize_frontend_path("my_crate::Type::{{impl}}::method::T"),
            "my_crate::Type::{{impl}}"
        );
    }

    #[test]
    fn test_normalize_impl_numbered_method_with_closure() {
        // `Type::{{impl}}[1]::method::{{closure}}` → `Type::{{impl}}[1]`.
        assert_eq!(
            normalize_frontend_path(
                "my_crate::Type::{{impl}}[1]::method::{{closure}}"
            ),
            "my_crate::Type::{{impl}}[1]"
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

    /// Build fake profiling data with `ProfilingDataBuilder` and feed it
    /// through `aggregate_profile` to test wall-clock computation.
    #[test]
    fn test_wall_clock_all_events() {
        use analyzeme::ProfilingDataBuilder;

        // Construct synthetic profiling data with both frontend and backend events.
        // All events contribute to a single wall-clock span now.
        //
        // Events:
        //   typeck: [10ms, 30ms]
        //   borrowck: [25ms, 50ms]
        //   LLVM_module_codegen: [60ms, 100ms]
        //   LLVM_module_codegen: [65ms, 90ms]
        //
        // Wall-clock = max(end) - min(start) = 100ms - 10ms = 90ms
        let ms = 1_000_000; // 1ms in nanos
        let mut builder = ProfilingDataBuilder::new();
        builder
            .interval("Query", "typeck", 0, 10 * ms, 30 * ms, |_| {})
            .interval("Query", "borrowck", 0, 25 * ms, 50 * ms, |_| {})
            .interval(
                "GenericActivity",
                "LLVM_module_codegen",
                0,
                60 * ms,
                100 * ms,
                |_| {},
            )
            .interval(
                "GenericActivity",
                "LLVM_module_codegen",
                0,
                65 * ms,
                90 * ms,
                |_| {},
            );

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");

        // Wall-clock spans all events: 100ms - 10ms = 90ms.
        assert!(
            (timings.wall_time_ms - 90.0).abs() < 0.1,
            "Expected wall_time_ms ~90.0, got {}",
            timings.wall_time_ms
        );
    }

    /// Single event produces correct wall-clock span.
    #[test]
    fn test_wall_clock_single_event() {
        use analyzeme::ProfilingDataBuilder;

        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        builder.interval("Query", "typeck", 0, 10 * ms, 50 * ms, |_| {});

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");

        assert!(
            (timings.wall_time_ms - 40.0).abs() < 0.1,
            "Expected wall_time_ms ~40.0, got {}",
            timings.wall_time_ms
        );
    }

    /// Metadata events are now part of the wall-clock span
    /// and recorded in `event_times_ms` with their raw label.
    #[test]
    fn test_metadata_in_frontend_span_and_event_times_ms() {
        use analyzeme::ProfilingDataBuilder;

        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        // Frontend event followed by metadata event.
        builder
            .interval("Query", "typeck", 0, 0, 10 * ms, |_| {})
            .interval(
                "GenericActivity",
                "generate_crate_metadata",
                0,
                10 * ms,
                25 * ms,
                |_| {},
            );

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");

        // Frontend wall-clock includes metadata: 25ms - 0ms = 25ms.
        assert!(
            (timings.wall_time_ms - 25.0).abs() < 0.1,
            "Expected wall_time_ms ~25.0 (includes metadata), got {}",
            timings.wall_time_ms
        );

        // Metadata self-time appears in event_times_ms under its raw label.
        let metadata_cost = timings
            .event_times_ms
            .get("generate_crate_metadata")
            .copied()
            .unwrap_or(0.0);
        assert!(
            (metadata_cost - 15.0).abs() < 0.1,
            "Expected event_times_ms['generate_crate_metadata'] ~15.0, got {metadata_cost}",
        );
    }

    /// Nested events must not inflate wall-clock time.
    ///
    /// Wall-clock = max(end) - min(start) across all events in a category,
    /// so nested events (whose intervals are contained within a parent) don't
    /// extend the span. If someone changed to duration-sum, this would break.
    #[test]
    fn test_wall_clock_nested_events_no_inflation() {
        use analyzeme::ProfilingDataBuilder;

        // Parent: typeck [10ms, 50ms] (40ms span)
        // Child:  typeck [20ms, 30ms] (10ms, fully nested)
        //
        // Wall-clock should be 40ms (50-10), not 50ms (40+10).
        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        builder.interval("Query", "typeck", 0, 10 * ms, 50 * ms, |inner| {
            inner.interval("Query", "typeck", 0, 20 * ms, 30 * ms, |_| {});
        });

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");

        assert!(
            (timings.wall_time_ms - 40.0).abs() < 0.1,
            "Nested events should not inflate wall-clock: expected ~40, got {}",
            timings.wall_time_ms
        );
    }

    /// Events on different threads should merge into a single wall-clock span.
    ///
    /// Wall-clock tracking is global (not per-thread) because we want the total
    /// elapsed time for a compilation, regardless of thread.
    #[test]
    fn test_wall_clock_multi_thread_merged() {
        use analyzeme::ProfilingDataBuilder;

        // Thread 0: LLVM_module_codegen [10ms, 50ms]
        // Thread 1: LLVM_module_codegen [20ms, 80ms]
        //
        // Wall-clock should be 80 - 10 = 70ms (global min/max).
        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        builder
            .interval(
                "GenericActivity",
                "LLVM_module_codegen",
                0,
                10 * ms,
                50 * ms,
                |_| {},
            )
            .interval(
                "GenericActivity",
                "LLVM_module_codegen",
                1,
                20 * ms,
                80 * ms,
                |_| {},
            );

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");

        assert!(
            (timings.wall_time_ms - 70.0).abs() < 0.1,
            "Multi-thread events should merge: expected ~70, got {}",
            timings.wall_time_ms
        );
    }

    /// Metadata events extend the wall-clock span and appear in `event_times_ms`.
    #[test]
    fn test_metadata_extends_wall_clock_span() {
        use analyzeme::ProfilingDataBuilder;

        // All events contribute to a single wall-clock span:
        //   typeck [10ms, 50ms]
        //   generate_crate_metadata [50ms, 70ms]
        //   LLVM_module_codegen [70ms, 100ms]
        //
        // Wall-clock = 100 - 10 = 90ms
        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        builder
            .interval("Query", "typeck", 0, 10 * ms, 50 * ms, |_| {})
            .interval(
                "GenericActivity",
                "generate_crate_metadata",
                0,
                50 * ms,
                70 * ms,
                |_| {},
            )
            .interval(
                "GenericActivity",
                "LLVM_module_codegen",
                0,
                70 * ms,
                100 * ms,
                |_| {},
            );

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");

        // Wall-clock spans all events: 100 - 10 = 90ms.
        assert!(
            (timings.wall_time_ms - 90.0).abs() < 0.1,
            "Wall-clock should span all events: expected ~90, got {}",
            timings.wall_time_ms
        );
        // Metadata self-time in event_times_ms.
        let metadata_cost = timings
            .event_times_ms
            .get("generate_crate_metadata")
            .copied()
            .unwrap_or(0.0);
        assert!(
            (metadata_cost - 20.0).abs() < 0.1,
            "Metadata event_cost should be ~20, got {metadata_cost}",
        );
    }

    /// Backend events are tracked in `event_times_ms` (complete ledger).
    #[test]
    fn test_backend_events_in_event_times_ms() {
        use analyzeme::ProfilingDataBuilder;

        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        builder.interval(
            "GenericActivity",
            "LLVM_module_codegen",
            0,
            10 * ms,
            50 * ms,
            |_| {},
        );

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");

        // Backend self-time should appear in event_times_ms.
        let llvm_cost = timings
            .event_times_ms
            .get("LLVM_module_codegen")
            .copied()
            .unwrap_or(0.0);
        assert!(
            (llvm_cost - 40.0).abs() < 0.1,
            "Expected event_times_ms['LLVM_module_codegen'] ~40, got {llvm_cost}",
        );
    }

    /// Events without a `DefPath` (e.g., `incr_comp`, `self_profile`) are now
    /// tracked in `event_times_ms` rather than being silently dropped.
    #[test]
    fn test_unattributed_events_in_event_times_ms() {
        use analyzeme::ProfilingDataBuilder;

        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        // incr_comp event with no DefPath data — previously skipped entirely.
        builder.interval(
            "GenericActivity",
            "incr_comp_persist_result",
            0,
            10 * ms,
            30 * ms,
            |_| {},
        );

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");

        // incr_comp event should appear in event_times_ms.
        let cost = timings
            .event_times_ms
            .get("incr_comp_persist_result")
            .copied()
            .unwrap_or(0.0);
        assert!(
            (cost - 20.0).abs() < 0.1,
            "Expected event_times_ms['incr_comp_persist_result'] ~20, got {cost}",
        );

        // And it should extend the frontend wall-clock span.
        assert!(
            (timings.wall_time_ms - 20.0).abs() < 0.1,
            "Expected wall_time_ms ~20, got {}",
            timings.wall_time_ms
        );
    }

    /// Symbol-attributed events appear ONLY in per-symbol `frontend_costs`.
    ///
    /// Events with a usable `DefPath` go to `frontend_costs` (exposed via
    /// `get_event_times_ms`), NOT to target-level `event_times_ms`. This
    /// ensures no double-counting: each event's self-time lands in exactly
    /// one place.
    #[test]
    fn test_attributed_events_in_event_times_ms() {
        use std::borrow::Cow;

        let mut data = ProfileData::default();

        // Construct an event with a valid DefPath in additional_data.
        // record_event should attribute it to the symbol AND record in
        // event_times_ms.
        let event = Event {
            event_kind: Cow::Borrowed("Query"),
            label: Cow::Borrowed("typeck"),
            additional_data: vec![Cow::Borrowed("my_crate::foo::bar")],
            thread_id: 0,
            payload: EventPayload::Timestamp(Timestamp::Interval {
                start: SystemTime::UNIX_EPOCH,
                end: SystemTime::UNIX_EPOCH + Duration::from_millis(10),
            }),
        };

        let result =
            data.record_event(&event, Duration::from_millis(10), "my_crate");

        // Should be attributed to a symbol (returns Some).
        assert!(result.is_some(), "event with DefPath should be attributed");

        // Should appear in per-symbol event_times_ms for attribution.
        let event_map = data.get_event_times_ms("my_crate::foo::bar").unwrap();
        let typeck_symbol_cost =
            event_map.get("typeck").copied().unwrap_or(0.0);
        assert!(
            (typeck_symbol_cost - 10.0).abs() < 0.1,
            "Expected symbol event_times_ms['typeck'] ~10, got {typeck_symbol_cost}",
        );

        // Attributed events should NOT appear in target-level
        // event_times_ms — they go only to the per-symbol map. The
        // target-level map should either be absent or contain 0 for
        // "typeck" since this event was fully attributed.
        let timings = data.get_target_timings("my_crate");
        let target_typeck = timings
            .and_then(|t| t.event_times_ms.get("typeck").copied())
            .unwrap_or(0.0);
        assert!(
            target_typeck.abs() < 0.1,
            "Attributed events should not appear in target-level event_times_ms, got {target_typeck}",
        );
    }

    /// Multiple events with the same label accumulate in `event_times_ms`.
    #[test]
    fn test_event_times_ms_accumulate_same_label() {
        use analyzeme::ProfilingDataBuilder;

        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        // Two separate typeck events (non-overlapping).
        builder
            .interval("Query", "typeck", 0, 0, 10 * ms, |_| {})
            .interval("Query", "typeck", 0, 15 * ms, 30 * ms, |_| {});

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");
        let typeck_cost =
            timings.event_times_ms.get("typeck").copied().unwrap_or(0.0);

        // 10ms + 15ms = 25ms (self-time of both events).
        assert!(
            (typeck_cost - 25.0).abs() < 0.1,
            "Expected accumulated typeck ~25, got {typeck_cost}",
        );
    }

    /// Complete accounting: `sum(target.event_times_ms) +
    /// sum(all symbol event_times_ms)` = total self-time.
    ///
    /// Attributed events land in per-symbol maps, unattributed events land
    /// in target-level `event_times_ms`. Together they account for all
    /// profiled self-time with no double-counting.
    #[test]
    fn test_event_times_ms_complete_accounting() {
        use std::borrow::Cow;

        let mut data = ProfileData::default();

        // Attributed event: typeck with DefPath -> goes to frontend_costs.
        let typeck_event = Event {
            event_kind: Cow::Borrowed("Query"),
            label: Cow::Borrowed("typeck"),
            additional_data: vec![Cow::Borrowed("my_crate::foo::bar")],
            thread_id: 0,
            payload: EventPayload::Timestamp(Timestamp::Interval {
                start: SystemTime::UNIX_EPOCH,
                end: SystemTime::UNIX_EPOCH + Duration::from_millis(10),
            }),
        };
        data.record_event(&typeck_event, Duration::from_millis(10), "my_crate");

        // Unattributed event: metadata with no DefPath -> goes to
        // event_times_ms.
        let metadata_event = Event {
            event_kind: Cow::Borrowed("GenericActivity"),
            label: Cow::Borrowed("generate_crate_metadata"),
            additional_data: vec![],
            thread_id: 0,
            payload: EventPayload::Timestamp(Timestamp::Interval {
                start: SystemTime::UNIX_EPOCH + Duration::from_millis(10),
                end: SystemTime::UNIX_EPOCH + Duration::from_millis(25),
            }),
        };
        data.record_event(
            &metadata_event,
            Duration::from_millis(15),
            "my_crate",
        );

        // Unattributed event: LLVM with no DefPath -> goes to
        // event_times_ms.
        let llvm_event = Event {
            event_kind: Cow::Borrowed("GenericActivity"),
            label: Cow::Borrowed("LLVM_module_codegen"),
            additional_data: vec![],
            thread_id: 0,
            payload: EventPayload::Timestamp(Timestamp::Interval {
                start: SystemTime::UNIX_EPOCH + Duration::from_millis(30),
                end: SystemTime::UNIX_EPOCH + Duration::from_millis(60),
            }),
        };
        data.record_event(&llvm_event, Duration::from_millis(30), "my_crate");

        // Sum target-level event_times_ms (unattributed only).
        let target_sum: f64 = data
            .get_target_timings("my_crate")
            .map_or(0.0, |t| t.event_times_ms.values().copied().sum());

        // Sum all per-symbol event_times_ms (attributed only).
        let symbol_sum: f64 = data
            .get_event_times_ms("my_crate::foo::bar")
            .map_or(0.0, |m| m.values().copied().sum());

        // Total: attributed(10) + unattributed(15 + 30) = 55ms.
        let total = target_sum + symbol_sum;
        assert!(
            (total - 55.0).abs() < 0.1,
            "sum(target.event_times_ms) + sum(symbol event_times_ms) should be ~55, got {total}",
        );

        // Verify the split: symbol got 10ms, target got 45ms.
        assert!(
            (symbol_sum - 10.0).abs() < 0.1,
            "Symbol sum should be ~10, got {symbol_sum}",
        );
        assert!(
            (target_sum - 45.0).abs() < 0.1,
            "Target sum should be ~45, got {target_sum}",
        );
    }

    /// Nested events: only self-time (not raw duration) goes into
    /// `event_times_ms`, preventing double-counting.
    #[test]
    fn test_event_times_ms_use_self_time_not_duration() {
        use analyzeme::ProfilingDataBuilder;

        let ms = 1_000_000;
        let mut builder = ProfilingDataBuilder::new();
        // Parent typeck [0, 20ms] contains child typeck [5ms, 15ms].
        // Parent self-time = 20 - 10 = 10ms.
        // Child self-time = 10ms.
        builder.interval("Query", "typeck", 0, 0, 20 * ms, |b| {
            b.interval("Query", "typeck", 0, 5 * ms, 15 * ms, |_| {});
        });

        let profile = builder.into_profiling_data();
        let mut data = ProfileData::default();
        data.aggregate_profile(&profile, "test_crate");

        let timings =
            data.get_target_timings("test_crate").expect("should exist");
        let typeck_cost =
            timings.event_times_ms.get("typeck").copied().unwrap_or(0.0);

        // Self-time: parent(10) + child(10) = 20ms.
        // NOT raw duration: parent(20) + child(10) = 30ms.
        assert!(
            (typeck_cost - 20.0).abs() < 0.1,
            "Expected self-time ~20 (not raw duration ~30), got {typeck_cost}",
        );
    }

    /// Empty `event_times_ms` when no events are profiled.
    #[test]
    fn test_event_times_ms_empty_without_events() {
        let data = ProfileData::default();
        assert!(data.get_target_timings("nonexistent").is_none());
    }
}
