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
//! - **Crate**: What rustc compiles. Each target compiles to a crate.
//!
//! A Package contains multiple Targets, and each Target compiles to a Crate.

use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Returns true if the value is zero (for serde `skip_serializing_if`).
///
/// Takes `&f64` because serde's `skip_serializing_if` passes by reference.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if requires &self"
)]
fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

/// Helper to sum all values in a `HashMap<String, f64>`.
///
/// Used by both `Symbol.event_times_ms` (total attributed cost) and
/// `TargetTimings.event_times_ms` (total unattributed cost).
pub fn sum_event_times<S: ::std::hash::BuildHasher>(
    map: &HashMap<String, f64, S>,
) -> f64 {
    map.values().sum()
}

/// Root structure representing the entire symbol graph of a workspace.
///
/// The symbol graph captures all symbols and their dependencies within a
/// Rust workspace. Each package contains targets (lib, bin, test, etc.),
/// and each target compiles to a crate with its own module tree.
/// Dependencies are stored directly on each symbol rather than as a
/// separate edge list.
#[derive(
    Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema,
)]
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
#[derive(
    Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema,
)]
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
#[derive(
    Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema,
)]
pub struct TargetTimings {
    /// Wall-clock time in milliseconds.
    ///
    /// Wall-clock elapsed time of all profiled events. Measured as
    /// max(end) - min(start) across all events in the profile.
    #[serde(default, skip_serializing_if = "is_zero")]
    #[schemars(range(min = 0.0))]
    pub wall_time_ms: f64,

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
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub event_times_ms: HashMap<String, f64>,
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
#[derive(
    Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema,
)]
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

/// Type alias for backwards compatibility during the `Crate` → `Target` rename.
///
/// TODO: Remove once all downstream code uses `Target` directly.
pub type Crate = Target;

/// A module (or crate root) containing symbols and submodules.
///
/// Modules form a tree structure where each module can contain:
/// - Symbols (functions, structs, enums, traits, impl blocks, etc.)
/// - Child submodules
///
/// The crate's root module (lib.rs or main.rs) is the top of this tree.
/// Both module and symbol names are stored as `HashMap` keys, not in the
/// structs themselves.
#[derive(
    Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema,
)]
pub struct Module {
    /// Symbols defined directly in this module, keyed by symbol name. Multiple
    /// impl blocks with the same signature (e.g., two `impl Foo` blocks) are
    /// merged into a single Symbol with combined cost.
    pub symbols: HashMap<String, Symbol>,

    /// Child modules, keyed by module name. Omitted if empty.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub submodules: HashMap<String, Module>,
}

/// A symbol in the crate - either a module-level definition or an impl block.
///
/// Symbols are the vertices in the dependency graph. Each symbol has a
/// source file, compilation cost estimates, dependencies, and kind-specific
/// details. Symbol names are stored as keys in the parent Module's `HashMap`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
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
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub event_times_ms: HashMap<String, f64>,

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
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Serialize,
    Deserialize,
    JsonSchema,
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
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "serde skip_serializing_if requires &self"
    )]
    fn is_non_public(&self) -> bool {
        *self == Visibility::NonPublic
    }
}

/// Metadata specific to the kind of symbol.
///
/// This enum distinguishes between regular module-level definitions (like
/// functions and structs) and impl blocks (which have different metadata).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    /// A module-level definition, analogous to `ra_ap_hir::ModuleDef`.
    ///
    /// This includes functions, structs, enums, unions, traits, consts,
    /// statics, type aliases, and macros.
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
    fn arb_visibility() -> impl Strategy<Value = Visibility> {
        prop_oneof![Just(Visibility::Public), Just(Visibility::NonPublic)]
    }

    /// Strategy for generating arbitrary `SymbolKind` values.
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
        fn arb_symbol()
            (
                file in arb_name(),
                event_times_ms in hash_map(
                    arb_name(),
                    (0..1_000_000).prop_map(f64::from),
                    0..5,
                ),
                dependencies in hash_set(arb_path(), 0..5),
                kind in arb_symbol_kind(),
            )
        -> Symbol {
            Symbol { file, event_times_ms, dependencies, kind }
        }
    }

    /// Strategy for generating a module with bounded recursive submodules.
    fn arb_module() -> impl Strategy<Value = Module> {
        prop_compose! {
            fn arb_leaf_module()
                (symbols in hash_map(arb_name(), arb_symbol(), 0..8))
            -> Module {
                Module { symbols, submodules: HashMap::new() }
            }
        }
        arb_leaf_module().prop_recursive(
            3, // max depth
            3, // we want about 3 submodules total
            1, // average of 1 submodule per module
            |inner| {
                (
                    hash_map(arb_name(), arb_symbol(), 0..8),
                    hash_map(arb_name(), inner, 0..3),
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
    fn arb_target_timings() -> impl Strategy<Value = TargetTimings> {
        (
            (0..1_000_000).prop_map(f64::from),
            hash_map(arb_name(), (0..1_000_000).prop_map(f64::from), 0..5),
        )
            .prop_map(|(wall_time_ms, event_times_ms)| TargetTimings {
                wall_time_ms,
                event_times_ms,
            })
    }

    /// Strategy for generating arbitrary Target values (compilation units).
    fn arb_target() -> impl Strategy<Value = Target> {
        (
            arb_target_timings(),
            // Dependencies are target references like "package/lib" or "package/bin/name"
            hash_set(arb_path(), 0..5),
            arb_module(),
        )
            .prop_map(|(timings, dependencies, root)| Target {
                timings,
                dependencies,
                root,
            })
    }

    /// Strategy for generating target keys (lib, test, bin/name, etc.).
    fn arb_target_key() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("lib".to_string()),
            Just("test".to_string()),
            arb_name().prop_map(|n| format!("bin/{n}")),
            arb_name().prop_map(|n| format!("example/{n}")),
        ]
    }

    /// Strategy for generating arbitrary Package values.
    fn arb_package() -> impl Strategy<Value = Package> {
        // Generate 1-4 targets per package
        hash_map(arb_target_key(), arb_target(), 1..5)
            .prop_map(|targets| Package { targets })
    }

    prop_compose! {
        /// Strategy for generating arbitrary SymbolGraph values.
        fn arb_symbol_graph()
            (packages in hash_map(arb_name(), arb_package(), 1..10))
        -> SymbolGraph {
            SymbolGraph { packages }
        }
    }

    proptest! {
        /// Test serialization roundtrip for arbitrary SymbolGraph instances.
        ///
        /// This exercises the Serialize/Deserialize derives by generating
        /// arbitrary graphs and verifying they survive a JSON roundtrip.
        #[test]
        fn test_symbol_graph_roundtrip(graph in arb_symbol_graph()) {
            let json = serde_json::to_string(&graph).expect("serialize");
            let parsed: SymbolGraph =
                serde_json::from_str(&json).expect("deserialize");
            prop_assert_eq!(parsed, graph);
        }
    }
}
