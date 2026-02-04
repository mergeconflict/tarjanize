//! Mono-items parsing for backend cost distribution.
//!
//! This module parses rustc's `-Zprint-mono-items=yes` output to map symbols
//! to their codegen units (CGUs). This mapping is essential for distributing
//! backend (LLVM) costs to individual symbols.
//!
//! ## Output Format
//!
//! The `-Zprint-mono-items=yes` flag prints lines like:
//! ```text
//! MONO_ITEM fn crate_name::module::function @@ cgu_name[Internal]
//! MONO_ITEM static crate_name::STATIC @@ cgu_name[External]
//! ```
//!
//! A single mono-item may appear in multiple CGUs due to inlining.

use std::collections::HashMap;
use std::io::BufRead;

use regex::Regex;
use tracing::{debug, trace};

/// Mapping from CGU names to the symbols they contain.
#[derive(Debug, Default)]
pub struct MonoItemsMap {
    /// Map from CGU name to list of symbol paths in that CGU.
    pub cgu_to_items: HashMap<String, Vec<String>>,
}

impl MonoItemsMap {
    /// Parse mono-items output from a reader.
    ///
    /// Reads lines from stderr output and extracts `MONO_ITEM` entries.
    /// The `crate_name` is used to filter items from the target crate.
    pub fn parse<R: BufRead>(reader: R, crate_name: &str) -> Self {
        // Regex to match: MONO_ITEM (fn|static) <path> @@ <cgu_name>[linkage]
        // The path may contain generic parameters like `<T>` or `<impl Trait>`.
        // CGU names have format like `crate_name.hash-cgu.N`.
        let re =
            Regex::new(r"^MONO_ITEM\s+(fn|static)\s+(.+?)\s+@@\s+([^\[\s]+)\[")
                .expect("valid regex");

        let mut map = MonoItemsMap::default();
        let mut line_count = 0;
        let mut match_count = 0;

        for line in reader.lines().map_while(Result::ok) {
            line_count += 1;

            // Skip lines that don't start with MONO_ITEM.
            if !line.starts_with("MONO_ITEM") {
                continue;
            }

            if let Some(caps) = re.captures(&line) {
                let item_kind = &caps[1];
                let raw_path = &caps[2];
                let cgu_name = &caps[3];

                // Filter by CGU name - only include items from our crate.
                // CGU names have format `crate_name.hash-cgu.N`.
                if !cgu_name.starts_with(crate_name)
                    || !cgu_name[crate_name.len()..].starts_with('.')
                {
                    continue;
                }

                // Normalize the path to match our symbol paths.
                let Some(normalized) =
                    normalize_mono_item(raw_path, crate_name)
                else {
                    trace!(
                        raw_path,
                        "skipped mono-item (normalization failed)"
                    );
                    continue;
                };

                trace!(
                    kind = item_kind,
                    path = normalized,
                    cgu = cgu_name,
                    "parsed mono-item"
                );

                map.cgu_to_items
                    .entry(cgu_name.to_string())
                    .or_default()
                    .push(normalized);

                match_count += 1;
            }
        }

        debug!(
            crate_name,
            line_count,
            match_count,
            cgu_count = map.cgu_to_items.len(),
            "parsed mono-items output"
        );

        map
    }

    /// Check if the map has any items.
    #[expect(dead_code, reason = "useful for debugging")]
    pub fn is_empty(&self) -> bool {
        self.cgu_to_items.is_empty()
    }
}

/// Normalize a mono-item path to match our symbol paths.
///
/// Mono-item paths have several forms that need normalization:
/// - Root module items: `foo` → `crate_name::foo`
/// - Nested items: `crate_name::mod::foo` → kept as-is
/// - Generic parameters: `foo::<T>` → `foo`
/// - Closures: `foo::{closure#0}` → `foo`
/// - Shims: `<crate::Type as Trait>::method` needs special handling
///
/// Returns `None` if the path doesn't belong to the target crate.
fn normalize_mono_item(path: &str, crate_name: &str) -> Option<String> {
    // Skip shim paths (start with `<`).
    // These are compiler-generated and don't correspond to user symbols.
    if path.starts_with('<') {
        return normalize_impl_shim(path, crate_name);
    }

    // Normalize the path to have a crate prefix.
    // The caller has already filtered by CGU name to ensure this item belongs
    // to our crate. Paths may come in several forms:
    // 1. Root-level items: `main` → `crate_name::main`
    // 2. Module items: `driver::run` → `crate_name::driver::run`
    // 3. Already prefixed: `crate_name::foo` → keep as-is
    let full_path = if let Some(after_crate) = path.strip_prefix(crate_name) {
        // Already has our crate prefix. Verify it's exactly our crate.
        if !after_crate.is_empty() && !after_crate.starts_with("::") {
            // Path starts with crate_name but isn't our crate (e.g., crate_name_extra).
            return None;
        }
        path.to_string()
    } else {
        // No crate prefix - add it.
        format!("{crate_name}::{path}")
    };

    // Strip generic parameters at all levels.
    let normalized = strip_generics(&full_path);

    // Aggregate closures to their parent function.
    let normalized = aggregate_closures(&normalized);

    // Aggregate to impl block level if applicable.
    let normalized = aggregate_to_impl(&normalized);

    Some(normalized)
}

/// Normalize an impl shim path like `<crate::module::Type as Trait>::method`.
///
/// Returns the type-level impl path if the type is from our crate.
/// The path format is `crate::module::Type::{{impl}}` which matches
/// the anchor paths stored in our extracted impl symbols.
fn normalize_impl_shim(path: &str, crate_name: &str) -> Option<String> {
    // Format: `<crate::module::Type as Trait>::method`
    // We want to extract the full type path.

    // Find the type portion (after `<` and before ` as` or `>`).
    let inner = path.strip_prefix('<')?;
    let type_end = inner
        .find(" as ")
        .unwrap_or_else(|| inner.find(">::").unwrap_or(inner.len()));
    let type_path = &inner[..type_end];

    // Strip generics and references from the type path.
    let type_path = type_path.trim_start_matches('&');
    let type_path = strip_generics(type_path);

    // Check if this looks like a path from our crate.
    // Type paths from our crate look like `Type` (root-level), `module::Type`
    // (relative), or `crate_name::module::Type` (absolute).
    let full_type_path =
        if let Some(after_crate) = type_path.strip_prefix(crate_name) {
            // Verify it's exactly our crate (not `crate_name_extra`).
            if !after_crate.is_empty() && !after_crate.starts_with("::") {
                return None; // Different crate with similar prefix
            }
            type_path.clone()
        } else {
            // No crate prefix - add it. This handles both:
            // - Single identifiers like `Cli` (root-level types)
            // - Relative paths like `module::Type`
            // CGU filtering already ensures we only process items from our crate.
            format!("{crate_name}::{type_path}")
        };

    // Return as type-level impl path (matches anchor format).
    Some(format!("{full_type_path}::{{{{impl}}}}"))
}

/// Strip generic parameters from a path.
///
/// Handles nested generics like `Foo<Bar<Baz>>` and turbofish like `foo::<T>`.
fn strip_generics(path: &str) -> String {
    let mut result = String::with_capacity(path.len());
    let mut depth: u32 = 0;

    for c in path.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => result.push(c),
            _ => {}
        }
    }

    // Remove trailing `::` that might remain after stripping turbofish like `::<T>`.
    result.trim_end_matches("::").to_string()
}

/// Aggregate closure paths to their parent function.
///
/// `crate::foo::{closure#0}` → `crate::foo`
fn aggregate_closures(path: &str) -> String {
    // Find first occurrence of `{closure` and truncate before the `::`.
    if let Some(pos) = path.find("::{closure") {
        return path[..pos].to_string();
    }
    path.to_string()
}

/// Aggregate to impl block level.
///
/// `crate::Type::{{impl}}::method` → `crate::Type::{{impl}}`
fn aggregate_to_impl(path: &str) -> String {
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
    fn test_strip_generics() {
        assert_eq!(strip_generics("foo"), "foo");
        assert_eq!(strip_generics("foo<T>"), "foo");
        assert_eq!(strip_generics("foo<T, U>"), "foo");
        assert_eq!(strip_generics("foo<Bar<Baz>>"), "foo");
        assert_eq!(strip_generics("crate::foo::bar<T>"), "crate::foo::bar");
    }

    #[test]
    fn test_aggregate_closures() {
        assert_eq!(aggregate_closures("crate::foo"), "crate::foo");
        assert_eq!(aggregate_closures("crate::foo::{closure#0}"), "crate::foo");
        assert_eq!(
            aggregate_closures("crate::foo::{closure#0}::{closure#1}"),
            "crate::foo"
        );
    }

    #[test]
    fn test_normalize_mono_item_basic() {
        assert_eq!(
            normalize_mono_item("my_crate::foo", "my_crate"),
            Some("my_crate::foo".to_string())
        );
        assert_eq!(
            normalize_mono_item("my_crate::foo::bar", "my_crate"),
            Some("my_crate::foo::bar".to_string())
        );
    }

    #[test]
    fn test_normalize_mono_item_adds_crate_prefix() {
        // Module-relative paths get crate prefix added.
        // (Filtering of other crates happens at CGU level, not here.)
        assert_eq!(
            normalize_mono_item("driver::run", "my_crate"),
            Some("my_crate::driver::run".to_string())
        );
        assert_eq!(
            normalize_mono_item("profile::categorize_event", "my_crate"),
            Some("my_crate::profile::categorize_event".to_string())
        );
        // Ensure prefix match doesn't succeed (different crate with similar name).
        assert_eq!(
            normalize_mono_item("my_crate_extra::foo", "my_crate"),
            None
        );
    }

    #[test]
    fn test_normalize_mono_item_strips_generics() {
        assert_eq!(
            normalize_mono_item("my_crate::foo::<T>", "my_crate"),
            Some("my_crate::foo".to_string())
        );
        assert_eq!(
            normalize_mono_item("my_crate::foo::<Bar<Baz>>", "my_crate"),
            Some("my_crate::foo".to_string())
        );
    }

    #[test]
    fn test_normalize_mono_item_aggregates_closures() {
        assert_eq!(
            normalize_mono_item("my_crate::foo::{closure#0}", "my_crate"),
            Some("my_crate::foo".to_string())
        );
    }

    #[test]
    fn test_normalize_impl_shim() {
        // Type at crate root -> type-level impl.
        assert_eq!(
            normalize_impl_shim(
                "<my_crate::Type as Trait>::method",
                "my_crate"
            ),
            Some("my_crate::Type::{{impl}}".to_string())
        );
        // Type in submodule -> type-level impl.
        assert_eq!(
            normalize_impl_shim(
                "<my_crate::module::Type as Trait>::method",
                "my_crate"
            ),
            Some("my_crate::module::Type::{{impl}}".to_string())
        );
        // Relative path (no crate prefix) -> gets crate prefix added.
        // (CGU filtering ensures we only process items from our crate.)
        assert_eq!(
            normalize_impl_shim("<module::Type as Trait>::method", "my_crate"),
            Some("my_crate::module::Type::{{impl}}".to_string())
        );
        // Reference type -> reference stripped.
        assert_eq!(
            normalize_impl_shim("<&module::Type as Trait>::method", "my_crate"),
            Some("my_crate::module::Type::{{impl}}".to_string())
        );
        // Single identifier (root-level type) -> gets crate prefix added.
        // This handles types like `Cli` in `<Cli as clap::Parser>::parse`.
        assert_eq!(
            normalize_impl_shim("<Cli as clap::Parser>::parse", "my_crate"),
            Some("my_crate::Cli::{{impl}}".to_string())
        );
    }

    #[test]
    fn test_parse_mono_items() {
        let input = "\
MONO_ITEM fn my_crate::foo @@ my_crate.abc123-cgu.0[Internal]
MONO_ITEM fn my_crate::bar::<i32> @@ my_crate.abc123-cgu.1[External]
MONO_ITEM static my_crate::STATIC @@ my_crate.abc123-cgu.0[Internal]
MONO_ITEM fn other_crate::baz @@ other_crate.def456-cgu.0[Internal]
some other output line
";
        let map = MonoItemsMap::parse(input.as_bytes(), "my_crate");

        assert_eq!(map.cgu_to_items.len(), 2);
        assert!(
            map.cgu_to_items
                .get("my_crate.abc123-cgu.0")
                .unwrap()
                .contains(&"my_crate::foo".to_string())
        );
        assert!(
            map.cgu_to_items
                .get("my_crate.abc123-cgu.0")
                .unwrap()
                .contains(&"my_crate::STATIC".to_string())
        );
        assert!(
            map.cgu_to_items
                .get("my_crate.abc123-cgu.1")
                .unwrap()
                .contains(&"my_crate::bar".to_string())
        );
    }
}
