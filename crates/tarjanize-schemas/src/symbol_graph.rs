//! Symbol graph schema for representing extracted dependency information.
//!
//! The symbol graph captures all symbols (functions, structs, traits, impl
//! blocks, etc.) in a workspace along with their dependency relationships.
//! This is the output of the extraction phase and an input to subsequent
//! analysis phases.
//!
//! ## Terminology
//!
//! We use Cargo and rustc terminology precisely:
//! - **Package**: A Cargo.toml and its contents. Has a unique name (may have hyphens).
//! - **Target**: A compilation unit within a package (lib, bin, test, etc.). Cargo's term.
//!
//! A Package contains one or more Targets (lib, bin, test, example, bench).

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_with::{DurationMilliSecondsWithFrac, Same, serde_as};
use ts_rs::TS;

/// Returns true if the duration is zero (for serde `skip_serializing_if`).
///
/// Why: keeping zero durations out of JSON keeps payloads small and avoids
/// noise in downstream tooling that treats missing as zero.
fn is_zero_duration(value: &Duration) -> bool {
    value.is_zero()
}

/// Helper to sum all values in a `HashMap<String, Duration>`.
///
/// Used by both `Symbol.event_times_ms` (total attributed cost) and
/// `TargetTimings.event_times_ms` (total unattributed cost).
///
/// Why: centralizes the total calculation so both paths stay consistent and
/// any future changes (e.g., filtering) happen in one place.
pub fn sum_event_times<S: ::std::hash::BuildHasher>(
    map: &HashMap<String, Duration, S>,
) -> Duration {
    map.values().sum()
}

/// Converts a `Duration` to f64 milliseconds.
///
/// Why: the cost model and scheduling primitives operate on floating-point
/// milliseconds for regression and analysis, while compilation timings are
/// stored as `Duration` internally.
pub fn duration_to_ms_f64(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Converts f64 milliseconds back to a `Duration`.
///
/// Why: regression math is float-based, but scheduling uses `Duration`.
pub fn ms_to_duration(ms: f64) -> Duration {
    Duration::from_secs_f64(ms / 1000.0)
}

/// Absolute `::` -separated path to a module within a compilation target.
///
/// Example: `"foo::bar::baz"` for the module `baz` nested inside `foo::bar`.
/// The root module is represented by an empty path (`ModulePath::root()`).
///
/// Used as the `module_path` parameter in recursive module-tree walks,
/// replacing the bare `&str` + `is_empty()` pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModulePath(String);

impl ModulePath {
    /// Creates a module path from a raw string.
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    /// Returns the root (empty) module path.
    pub fn root() -> Self {
        Self(String::new())
    }

    /// Returns a child path by appending `::segment`.
    ///
    /// If this path is the root, returns just the segment (no leading `::`).
    #[must_use]
    pub fn child(&self, segment: &str) -> Self {
        if self.0.is_empty() {
            Self(segment.to_owned())
        } else {
            Self(format!("{}::{segment}", self.0))
        }
    }

    /// Returns the underlying string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns `true` if this is the root module path.
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for ModulePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Fully-qualified symbol path: `[package/target]::module::symbol`.
///
/// The canonical path format after extraction. Wraps the bracketed
/// `[pkg/target]::rest` format used for workspace-global identification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QualifiedSymbolPath(String);

impl QualifiedSymbolPath {
    /// Constructs a path from its components.
    pub fn new(
        target_id: &TargetId,
        module_path: &ModulePath,
        symbol: &str,
    ) -> Self {
        if module_path.is_root() {
            Self(format!("[{target_id}]::{symbol}"))
        } else {
            Self(format!("[{target_id}]::{}::{symbol}", module_path.as_str()))
        }
    }

    /// Splits the bracketed prefix from the rest of the path.
    ///
    /// Returns `(prefix, rest)` where `prefix` is `"[package/target]"` and
    /// `rest` is the remaining `"module::symbol"` portion.
    ///
    /// Returns `None` if the path doesn't start with `[`.
    pub fn parse_prefix(path: &str) -> Option<(&str, &str)> {
        if !path.starts_with('[') {
            return None;
        }
        let bracket_end = path.find(']')?;
        let prefix = &path[..=bracket_end];
        let rest = path.get(bracket_end + 1..)?.strip_prefix("::")?;
        Some((prefix, rest))
    }

    /// Returns the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for QualifiedSymbolPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identifies a compilation target as `"package/target"` (e.g. `"tokio/lib"`).
///
/// Wraps the `{package}/{target}` string format used throughout the pipeline
/// to reference compilation units. Provides typed accessors for the package
/// and target components, eliminating repeated `split_once('/')` calls.
///
/// Serializes transparently as a plain string for JSON compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TargetId(String);

impl TargetId {
    /// Constructs a `TargetId` from separate package and target components.
    ///
    /// The resulting string is `"{package}/{target}"`.
    pub fn new(package: &str, target: &str) -> Self {
        Self(format!("{package}/{target}"))
    }

    /// Parses a `"package/target"` string, returning `None` if no `/` is found.
    pub fn parse(s: &str) -> Option<Self> {
        // Validate that at least one '/' exists so accessors never panic.
        s.find('/').map(|_| Self(s.to_owned()))
    }

    /// Returns the package portion (everything before the first `/`).
    pub fn package(&self) -> &str {
        self.0
            .split_once('/')
            .expect("TargetId always contains '/'")
            .0
    }

    /// Returns the target portion (everything after the first `/`).
    pub fn target(&self) -> &str {
        self.0
            .split_once('/')
            .expect("TargetId always contains '/'")
            .1
    }

    /// Returns the underlying string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TargetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for TargetId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<str> for TargetId {
    /// Enables `HashMap<TargetId, _>::get("pkg/lib")` lookups.
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// Root structure representing the entire symbol graph of a workspace.
///
/// The symbol graph captures all symbols and their dependencies within a
/// Rust workspace. Each package contains targets (lib, bin, test, etc.),
/// and each target compiles to a crate with its own module tree.
/// Dependencies are stored directly on each symbol rather than as a
/// separate edge list.
///
/// Why: downstream phases need a stable, serializable representation that
/// survives process boundaries without rustc internals.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
pub struct SymbolGraph {
    /// All packages in the workspace, keyed by package name (hyphens preserved).
    pub packages: HashMap<String, Package>,
}

/// A package in the workspace containing one or more compilation targets.
///
/// Each package corresponds to a Cargo.toml and contains multiple targets:
/// - `lib`: The library target (if present)
/// - `test`: Unit tests (`#[cfg(test)]` code in lib)
/// - `bin/{name}`: Binary targets
/// - `example/{name}`: Example targets
/// - `bench/{name}`: Benchmark targets
///
/// Integration tests (files in `tests/`) are separate packages from Cargo's
/// perspective and get their own entry in the top-level `packages` map.
///
/// Why: we model packages explicitly to preserve Cargo's build graph
/// boundaries, which drive dependency and scheduling behavior.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
pub struct Package {
    /// Compilation targets for this package, keyed by target identifier.
    /// Each target compiles to a separate crate (rustc compilation unit).
    ///
    /// Target keys use the format:
    /// - `"lib"` for library targets
    /// - `"test"` for unit tests
    /// - `"bin/{name}"` for binary targets
    /// - `"example/{name}"` for example targets
    /// - `"bench/{name}"` for benchmark targets
    pub targets: HashMap<String, Target>,
}

/// Wall-clock timing for a compilation target.
///
/// Captures the profiler's wall-clock time and unattributed event self-times
/// per compilation target. For crate splitting, only frontend time matters —
/// it's serial, determines rmeta readiness, and gates downstream crates.
///
/// `event_times_ms` contains ONLY unattributed event self-times — events
/// that could NOT be attributed to a specific symbol via `DefPath`. Together
/// with `Symbol.event_times_ms`, the two maps account for all profiled
/// self-time with no double-counting: every millisecond of profiled
/// self-time lands in exactly one place.
///
/// These values come directly from `-Zself-profile` wall-clock intervals when
/// profiling is enabled. For synthetic crates (from condense), they are
/// estimated from per-symbol costs and the fitted regression model.
///
/// Why: target-level timings are the unit of critical-path scheduling and
/// cost modeling, so they must be stored independently of symbols.
#[serde_as]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
pub struct TargetTimings {
    /// Wall-clock elapsed time of all profiled events.
    ///
    /// Measured as max(end) - min(start) across all events in the
    /// profile. Serialized as f64 milliseconds for JSON consumers.
    #[serde(
        rename = "wall_time_ms",
        default,
        skip_serializing_if = "is_zero_duration"
    )]
    #[serde_as(as = "DurationMilliSecondsWithFrac")]
    #[ts(as = "f64")]
    pub wall_time: Duration,

    /// Unattributed self-time breakdown by event label (ms).
    ///
    /// Keys are the original self-profile event labels (e.g.
    /// `"metadata_decode_entry_generics_of"`, `"LLVM_module_codegen"`).
    /// Values are accumulated self-time for events that were NOT attributed
    /// to a specific symbol via `DefPath`. Together with `Symbol.event_times_ms`,
    /// these two maps account for all profiled self-time.
    ///
    /// The metadata decode cost for the regression model is derived by summing
    /// all `metadata_decode_*` entries at query time.
    ///
    /// Serialized as floating-point milliseconds for JSON consumers.
    ///
    /// Why: `Duration` keeps internal arithmetic exact while preserving
    /// the existing on-disk schema.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    #[serde_as(as = "HashMap<Same, DurationMilliSecondsWithFrac>")]
    #[ts(type = "Record<string, number>")]
    pub event_times_ms: HashMap<String, Duration>,
}

/// A compilation target (rustc compilation unit) within a package.
///
/// Each target represents a separate compilation unit with its own:
/// - Module tree containing symbols
/// - Timing data (wall-clock frontend + event cost breakdown)
/// - Dependencies on other targets
///
/// This separation allows accurate modeling of:
/// - Dev-dependency relationships (tests depend on libs, not vice versa)
/// - Per-target compilation costs
/// - Critical path through the actual build graph
///
/// Why: targets are the granularity at which rustc schedules work and
/// produces artifacts, so analysis must preserve them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
pub struct Target {
    /// Wall-clock timing breakdown for this target.
    pub timings: TargetTimings,

    /// Dependencies on other targets.
    ///
    /// Each entry is a target reference in the format `"{package}/{target}"`:
    /// - `"other-package/lib"` - depends on another package's library
    /// - `"my-package/lib"` - test/bin target depends on own library
    ///
    /// For lib targets: contains normal dependencies (`{dep-package}/lib`)
    /// For test targets: normal + dev deps + own lib (`{self-package}/lib`)
    /// For bin targets: normal deps + own lib (`{self-package}/lib`)
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub dependencies: HashSet<String>,

    /// The root module containing all symbols and submodules for this target.
    pub root: Module,
}

/// A module (or crate root) containing symbols and submodules.
///
/// Modules form a tree structure where each module can contain:
/// - Symbols (functions, structs, enums, traits, impl blocks, etc.)
/// - Child submodules
///
/// The crate's root module (lib.rs or main.rs) is the top of this tree.
/// Both module and symbol names are stored as `HashMap` keys, not in the
/// structs themselves.
///
/// Why: a tree mirrors Rust's module hierarchy and powers both UI display
/// and structural transforms during splitting.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
pub struct Module {
    /// Symbols defined directly in this module, keyed by symbol name. Multiple
    /// impl blocks with the same signature (e.g., two `impl Foo` blocks) are
    /// merged into a single Symbol with combined cost.
    pub symbols: HashMap<String, Symbol>,

    /// Child modules, keyed by module name. Omitted if empty.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub submodules: HashMap<String, Module>,
}

impl Module {
    /// Recursively counts symbols in this module tree.
    ///
    /// Why: symbol counts are used for logging, scheduling UI, and as a
    /// predictor in the cost model regression.
    pub fn count_symbols(&self) -> usize {
        let direct = self.symbols.len();
        let nested: usize =
            self.submodules.values().map(Self::count_symbols).sum();
        direct + nested
    }

    /// Recursively sums all symbol `event_times_ms` in this module tree.
    ///
    /// Frontend compilation work is serial, so we sum all per-symbol event
    /// times to get the total attributed frontend cost for a target.
    ///
    /// Why: attr cost is derived from per-symbol events and must include the
    /// full module hierarchy.
    pub fn collect_frontend_cost(&self) -> Duration {
        let mut total = Duration::ZERO;
        for symbol in self.symbols.values() {
            total += sum_event_times(&symbol.event_times_ms);
        }
        for submodule in self.submodules.values() {
            total += submodule.collect_frontend_cost();
        }
        total
    }

    /// Calls `f(module_path, symbol_name, symbol)` for every symbol in this
    /// subtree, where `module_path` is the absolute path from the target root.
    ///
    /// Handles root-vs-nested path construction internally so callers do
    /// not need the `if is_empty() { ... } else { format!(...) }` pattern.
    ///
    /// Why: centralizes the 15+ manual module traversals in the codebase.
    pub fn for_each_symbol(
        &self,
        path: &ModulePath,
        f: &mut impl FnMut(&ModulePath, &str, &Symbol),
    ) {
        for (name, symbol) in &self.symbols {
            f(path, name, symbol);
        }
        for (sub_name, sub) in &self.submodules {
            sub.for_each_symbol(&path.child(sub_name), f);
        }
    }

    /// Calls `f(module_path, module)` for every module in this subtree
    /// (including `self`).
    ///
    /// Why: some analyses need per-module metadata without per-symbol detail.
    pub fn for_each_module(
        &self,
        path: &ModulePath,
        f: &mut impl FnMut(&ModulePath, &Module),
    ) {
        f(path, self);
        for (sub_name, sub) in &self.submodules {
            sub.for_each_module(&path.child(sub_name), f);
        }
    }
}

/// A symbol in the crate - either a module-level definition or an impl block.
///
/// Symbols are the vertices in the dependency graph. Each symbol has a
/// source file, compilation cost estimates, dependencies, and kind-specific
/// details. Symbol names are stored as keys in the parent Module's `HashMap`.
///
/// Why: symbols are the unit of dependency analysis and cost attribution,
/// so they need their own identity and metadata.
#[serde_as]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct Symbol {
    /// Path to the symbol's file relative to the crate root.
    pub file: String,

    /// Self-time breakdown by event label for events attributed to this symbol
    /// after descendant aggregation (ms).
    ///
    /// Keys are profiler event labels (e.g. `"typeck"`, `"mir_borrowck"`,
    /// `"predicates_of"`). Values are accumulated self-time for that event
    /// across this symbol and all its descendants (closures, generic params,
    /// anonymous consts, opaque types, etc.).
    ///
    /// The total attributed cost for this symbol is `sum(event_times_ms.values())`.
    /// This replaces the old scalar `frontend_cost_ms` field and provides
    /// per-event visibility for regression testing.
    ///
    /// Serialized as floating-point milliseconds for JSON consumers.
    ///
    /// Why: `Duration` keeps internal arithmetic exact while preserving
    /// the existing on-disk schema.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    #[serde_as(as = "HashMap<Same, DurationMilliSecondsWithFrac>")]
    #[ts(type = "Record<string, number>")]
    pub event_times_ms: HashMap<String, Duration>,

    /// Fully qualified paths of symbols this symbol depends on. A dependency
    /// means this symbol's definition references the target in some way
    /// (type annotation, function call, trait bound, etc.).
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub dependencies: HashSet<String>,

    /// Symbol-specific fields depending on whether this is a module-level
    /// definition or an impl block.
    #[serde(flatten)]
    pub kind: SymbolKind,
}

/// Visibility of a symbol for crate-splitting purposes.
///
/// When splitting a crate, only `Public` items are guaranteed to remain
/// accessible without modification. All other visibilities (`pub(crate)`,
/// `pub(super)`, private) may require upgrading to `pub` if accessed across
/// the new crate boundary.
///
/// Defaults to `NonPublic` and is omitted from serialization when non-public.
///
/// Why: visibility drives whether a split can be performed without API
/// changes or requires public surface adjustments.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS,
)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Fully public (`pub`). No visibility changes needed when splitting.
    Public,
    /// Not fully public. Includes `pub(crate)`, `pub(super)`, `pub(in path)`,
    /// and private items. May need visibility upgrade when splitting crates.
    #[default]
    NonPublic,
}

impl Visibility {
    /// Returns true if this is non-public visibility.
    ///
    /// Takes `&self` because serde's `skip_serializing_if` passes by reference.
    ///
    /// Why: serde needs a stable predicate to omit default values so JSON
    /// stays compact and readable.
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "Serde's skip_serializing_if passes by reference; keep &self."
    )]
    fn is_non_public(&self) -> bool {
        matches!(self, Visibility::NonPublic)
    }
}

/// Metadata specific to the kind of symbol.
///
/// This enum distinguishes between regular module-level definitions (like
/// functions and structs) and impl blocks (which have different metadata).
///
/// Why: impl blocks carry orphan-rule anchors and do not behave like regular
/// module definitions during splitting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    /// A module-level definition, analogous to `ra_ap_hir::ModuleDef`.
    ///
    /// This includes functions, structs, enums, unions, traits, consts,
    /// statics, type aliases, and macros.
    ///
    /// Why: module defs can be moved based on visibility and dependencies,
    /// so we capture them separately from impl-specific constraints.
    ModuleDef {
        /// The kind of definition: "Function", "Struct", "Enum", etc.
        /// Uses `PascalCase` to match rust-analyzer's `SymbolKind` enum.
        kind: String,

        /// Visibility for crate-splitting purposes. Defaults to `NonPublic`
        /// and is omitted from serialization when non-public.
        #[serde(default, skip_serializing_if = "Visibility::is_non_public")]
        visibility: Visibility,
    },

    /// An impl block, analogous to `ra_ap_hir::Impl`.
    ///
    /// Impl blocks are distinct from `ModuleDefs`; they don't have visibility,
    /// and Rust's orphan rules dictate where they can be defined.
    ///
    /// Why: impls must remain co-located with at least one anchor, so they
    /// require extra metadata during splitting.
    Impl {
        /// Human-readable name of the impl block.
        ///
        /// For trait impls: `impl Trait for Type`
        /// For inherent impls: `impl Type`
        ///
        /// This provides readability since the symbol key uses the compiler's
        /// internal `DefPath` format (`{{impl}}[N]`).
        name: String,

        /// Workspace-local types and traits that can satisfy the orphan rule.
        ///
        /// For `impl<P1..=Pn> Trait<T1..=Tn> for T0`, the orphan rule allows:
        /// - The trait is local, OR
        /// - At least one of T0..=Tn is local (including trait type params)
        ///
        /// This set contains the fully qualified paths of all local types and
        /// the trait (if local). Phase 2 maps these to SCCs, and Phase 3
        /// ensures the impl ends up in a crate with at least one anchor.
        #[serde(default, skip_serializing_if = "HashSet::is_empty")]
        anchors: HashSet<String>,
    },
}

#[cfg(test)]
mod tests {
    use proptest::collection::{hash_map, hash_set};
    use proptest::prelude::*;

    use super::*;
    use crate::testutil::{arb_name, arb_path};

    // -------------------------------------------------------------------------
    // Proptest strategies for generating arbitrary schema instances.
    //
    // These are defined here to keep the production types clean of test
    // annotations. The strategies generate bounded instances to avoid
    // stack overflow from unbounded recursion.
    //
    // TODO: Implement proptest strategies for generating structurally valid
    // SymbolGraph instances. Dependencies and impl paths must reference actual
    // symbols. Requires two-phase generation: create symbol structure first,
    // then populate dependencies from the set of valid symbol paths.
    // -------------------------------------------------------------------------

    /// Strategy for generating arbitrary Visibility values.
    ///
    /// Why: proptest needs coverage of both public and non-public cases
    /// to validate serde and split-related logic.
    fn arb_visibility() -> impl Strategy<Value = Visibility> {
        prop_oneof![Just(Visibility::Public), Just(Visibility::NonPublic)]
    }

    /// Strategy for generating arbitrary `SymbolKind` values.
    ///
    /// Why: we must exercise both module definitions and impl blocks so
    /// serde roundtrips cover the distinct shape of each variant.
    fn arb_symbol_kind() -> impl Strategy<Value = SymbolKind> {
        prop_compose! {
            fn arb_module_def()
                (kind in arb_name(), visibility in arb_visibility())
            -> SymbolKind {
                SymbolKind::ModuleDef { kind, visibility }
            }
        }

        prop_compose! {
            fn arb_impl()
                (name in arb_name(), anchors in hash_set(arb_path(), 0..5))
            -> SymbolKind {
                SymbolKind::Impl { name, anchors }
            }
        }

        prop_oneof![arb_module_def(), arb_impl()]
    }

    prop_compose! {
        /// Strategy for generating arbitrary Symbol values with event_times_ms
        /// containing integer-valued costs (floats don't survive JSON roundtrip
        /// perfectly).
        ///
        /// Why: integer costs keep roundtrips stable while still exercising
        /// the cost accounting paths.
        fn arb_symbol()
            (
                file in arb_name(),
                event_times_ms in hash_map(
                    arb_name(),
                    (0..1_000_000u64).prop_map(Duration::from_millis),
                    0..3,
                ),
                dependencies in hash_set(arb_path(), 0..3),
                kind in arb_symbol_kind(),
            )
        -> Symbol {
            Symbol { file, event_times_ms, dependencies, kind }
        }
    }

    /// Strategy for generating a module with bounded recursive submodules.
    ///
    /// Why: bounded recursion avoids stack overflow while still exercising
    /// tree shape and aggregation behavior.
    fn arb_module() -> impl Strategy<Value = Module> {
        prop_compose! {
            fn arb_leaf_module()
                (symbols in hash_map(arb_name(), arb_symbol(), 0..4))
            -> Module {
                Module { symbols, submodules: HashMap::new() }
            }
        }
        arb_leaf_module().prop_recursive(
            2, // max depth
            2, // we want about 2 submodules total
            1, // average of 1 submodule per module
            |inner| {
                (
                    hash_map(arb_name(), arb_symbol(), 0..4),
                    hash_map(arb_name(), inner, 0..2),
                )
                    .prop_map(|(symbols, submodules)| Module {
                        symbols,
                        submodules,
                    })
            },
        )
    }

    /// Strategy for generating arbitrary `TargetTimings` values.
    ///
    /// Generates timing values with an optional `event_times_ms` map containing
    /// integer-valued costs (floats don't survive JSON roundtrip perfectly).
    /// Wall time uses integer milliseconds to survive f64 JSON roundtrip.
    ///
    /// Why: targets drive scheduling and cost models, so timings must roundtrip
    /// without precision drift.
    fn arb_target_timings() -> impl Strategy<Value = TargetTimings> {
        (
            (0..1_000_000u64).prop_map(Duration::from_millis),
            hash_map(
                arb_name(),
                (0..1_000_000u64).prop_map(Duration::from_millis),
                0..3,
            ),
        )
            .prop_map(|(wall_time, event_times_ms)| TargetTimings {
                wall_time,
                event_times_ms,
            })
    }

    /// Strategy for generating arbitrary Target values (compilation units).
    ///
    /// Why: validates serde for the target shape that scheduling and condense
    /// operate on.
    fn arb_target() -> impl Strategy<Value = Target> {
        (
            arb_target_timings(),
            // Dependencies are target references like "package/lib" or "package/bin/name"
            hash_set(arb_path(), 0..3),
            arb_module(),
        )
            .prop_map(|(timings, dependencies, root)| Target {
                timings,
                dependencies,
                root,
            })
    }

    /// Strategy for generating target keys (lib, test, bin/name, etc.).
    ///
    /// Why: cover the key formats used by Cargo to avoid missing variants.
    fn arb_target_key() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("lib".to_string()),
            Just("test".to_string()),
            arb_name().prop_map(|n| format!("bin/{n}")),
            arb_name().prop_map(|n| format!("example/{n}")),
        ]
    }

    /// Strategy for generating arbitrary Package values.
    ///
    /// Why: packages group targets and must survive roundtrips intact.
    fn arb_package() -> impl Strategy<Value = Package> {
        // Generate 1-3 targets per package
        hash_map(arb_target_key(), arb_target(), 1..4)
            .prop_map(|targets| Package { targets })
    }

    prop_compose! {
        /// Strategy for generating arbitrary SymbolGraph values.
        ///
        /// Why: covers the full top-level schema for serde stability.
        fn arb_symbol_graph()
            (packages in hash_map(arb_name(), arb_package(), 1..6))
        -> SymbolGraph {
            SymbolGraph { packages }
        }
    }

    proptest! {
        /// Test serialization roundtrip for arbitrary SymbolGraph instances.
        ///
        /// This exercises the Serialize/Deserialize derives by generating
        /// arbitrary graphs and verifying they survive a JSON roundtrip.
        ///
        /// Why: schema stability depends on serde roundtrips across versions.
        #[test]
        fn test_symbol_graph_roundtrip(graph in arb_symbol_graph()) {
            let json = serde_json::to_string(&graph).expect("serialize");
            let parsed: SymbolGraph =
                serde_json::from_str(&json).expect("deserialize");
            prop_assert_eq!(parsed, graph);
        }
    }

    // -----------------------------------------------------------------
    // TargetId tests
    // -----------------------------------------------------------------

    #[test]
    fn target_id_new_and_accessors() {
        let id = TargetId::new("tokio", "lib");
        assert_eq!(id.as_str(), "tokio/lib");
        assert_eq!(id.package(), "tokio");
        assert_eq!(id.target(), "lib");
    }

    #[test]
    fn target_id_nested_target() {
        // Target keys like "bin/my-tool" have a second slash.
        let id = TargetId::new("my-pkg", "bin/my-tool");
        assert_eq!(id.package(), "my-pkg");
        assert_eq!(id.target(), "bin/my-tool");
    }

    #[test]
    fn target_id_parse_valid() {
        let id = TargetId::parse("serde/lib").expect("should parse");
        assert_eq!(id.package(), "serde");
        assert_eq!(id.target(), "lib");
    }

    #[test]
    fn target_id_parse_missing_slash() {
        assert!(TargetId::parse("noslash").is_none());
    }

    #[test]
    fn target_id_display() {
        let id = TargetId::new("foo", "test");
        assert_eq!(format!("{id}"), "foo/test");
    }

    #[test]
    fn target_id_serde_transparent() {
        // TargetId should serialize as a bare string, not an object.
        let id = TargetId::new("pkg", "lib");
        let json = serde_json::to_string(&id).expect("serialize");
        assert_eq!(json, "\"pkg/lib\"");
        let roundtrip: TargetId =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtrip, id);
    }

    #[test]
    fn target_id_borrow_str_lookup() {
        // Borrow<str> enables HashMap<TargetId, _>::get("key") lookups.
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(TargetId::new("a", "lib"), 42);
        assert_eq!(map.get("a/lib"), Some(&42));
    }

    // -----------------------------------------------------------------
    // ModulePath tests
    // -----------------------------------------------------------------

    #[test]
    fn module_path_root() {
        let root = ModulePath::root();
        assert!(root.is_root());
        assert_eq!(root.as_str(), "");
    }

    #[test]
    fn module_path_child_from_root() {
        let child = ModulePath::root().child("foo");
        assert!(!child.is_root());
        assert_eq!(child.as_str(), "foo");
    }

    #[test]
    fn module_path_nested_child() {
        let path = ModulePath::root().child("foo").child("bar");
        assert_eq!(path.as_str(), "foo::bar");
    }

    #[test]
    fn module_path_display() {
        let path = ModulePath::new("a::b");
        assert_eq!(format!("{path}"), "a::b");
    }

    // -----------------------------------------------------------------
    // Module visitor tests
    // -----------------------------------------------------------------

    /// Helper: creates a Symbol with no cost or deps.
    fn stub_symbol() -> Symbol {
        Symbol {
            file: "f.rs".to_string(),
            event_times_ms: HashMap::new(),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::NonPublic,
            },
        }
    }

    #[test]
    fn for_each_symbol_visits_all() {
        let mut sub = Module::default();
        sub.symbols.insert("B".to_string(), stub_symbol());

        let mut root = Module::default();
        root.symbols.insert("A".to_string(), stub_symbol());
        root.submodules.insert("child".to_string(), sub);

        let mut visited = Vec::new();
        root.for_each_symbol(&ModulePath::root(), &mut |path, name, _sym| {
            let full = if path.is_root() {
                name.to_string()
            } else {
                format!("{path}::{name}")
            };
            visited.push(full);
        });
        visited.sort();
        assert_eq!(visited, vec!["A", "child::B"]);
    }

    #[test]
    fn for_each_module_visits_all() {
        let sub = Module::default();
        let mut root = Module::default();
        root.submodules.insert("m".to_string(), sub);

        let mut paths = Vec::new();
        root.for_each_module(&ModulePath::root(), &mut |path, _mod| {
            paths.push(path.as_str().to_string());
        });
        paths.sort();
        assert_eq!(paths, vec!["", "m"]);
    }

    #[test]
    fn qualified_symbol_path_root_module() {
        let tid = TargetId::new("pkg", "lib");
        let p = QualifiedSymbolPath::new(&tid, &ModulePath::root(), "Foo");
        assert_eq!(p.as_str(), "[pkg/lib]::Foo");
    }

    #[test]
    fn qualified_symbol_path_nested_module() {
        let tid = TargetId::new("pkg", "lib");
        let mp = ModulePath::new("a::b");
        let p = QualifiedSymbolPath::new(&tid, &mp, "Bar");
        assert_eq!(p.as_str(), "[pkg/lib]::a::b::Bar");
    }

    #[test]
    fn qualified_symbol_path_display() {
        let tid = TargetId::new("pkg", "lib");
        let p = QualifiedSymbolPath::new(&tid, &ModulePath::root(), "X");
        assert_eq!(p.to_string(), "[pkg/lib]::X");
    }

    #[test]
    fn parse_prefix_valid() {
        let (prefix, rest) =
            QualifiedSymbolPath::parse_prefix("[pkg/lib]::mod::Foo").unwrap();
        assert_eq!(prefix, "[pkg/lib]");
        assert_eq!(rest, "mod::Foo");
    }

    #[test]
    fn parse_prefix_not_bracketed() {
        assert!(QualifiedSymbolPath::parse_prefix("crate::Foo").is_none());
    }
}
