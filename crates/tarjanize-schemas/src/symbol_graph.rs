//! Symbol graph schema for representing extracted dependency information.
//!
//! The symbol graph captures all symbols (functions, structs, traits, impl
//! blocks, etc.) in a workspace along with their dependency relationships.
//! This is the output of the extraction phase and an input to subsequent
//! analysis phases.

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

/// Root structure representing the entire symbol graph of a workspace.
///
/// The symbol graph captures all symbols and their dependencies within a
/// Rust workspace. Each crate contains its root module (with symbols and
/// submodules) plus crate-level overhead costs. Dependencies are stored
/// directly on each symbol rather than as a separate edge list.
#[derive(
    Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema,
)]
pub struct SymbolGraph {
    /// All crates in the workspace, keyed by crate name.
    pub crates: HashMap<String, Crate>,
}

/// A crate in the workspace with its module tree and overhead costs.
///
/// Each crate has:
/// - A root module containing all symbols and submodules
/// - Fixed overhead costs (linking, metadata generation) that apply
///   regardless of which symbols are included
#[derive(
    Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema,
)]
pub struct Crate {
    /// Linking time in milliseconds.
    ///
    /// Fixed cost for linking the crate's artifacts. This includes `link_crate`,
    /// `link_binary`, and `link_rlib` events from self-profile. Not parallelizable.
    #[serde(default, skip_serializing_if = "is_zero")]
    #[schemars(range(min = 0.0))]
    pub linking_ms: f64,

    /// Metadata generation time in milliseconds.
    ///
    /// Fixed cost for generating crate metadata (`generate_crate_metadata`).
    /// This is required for downstream crates to depend on this one.
    #[serde(default, skip_serializing_if = "is_zero")]
    #[schemars(range(min = 0.0))]
    pub metadata_ms: f64,

    /// The root module containing all symbols and submodules.
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

    /// Frontend compilation cost in milliseconds.
    ///
    /// Includes parsing, type checking, borrow checking, and other serial
    /// compilation phases. Populated from rustc's self-profile data when
    /// profiling is enabled. Defaults to 0.0.
    ///
    /// Frontend costs are serial - when compiling symbols A and B:
    /// `total_frontend = A.frontend_cost_ms + B.frontend_cost_ms`
    #[serde(default, skip_serializing_if = "is_zero")]
    #[schemars(range(min = 0.0))]
    pub frontend_cost_ms: f64,

    /// Backend compilation cost in milliseconds.
    ///
    /// Includes LLVM codegen, which can run in parallel across CGUs.
    /// Populated by distributing CGU costs to symbols via mono-items mapping.
    /// Defaults to 0.0.
    ///
    /// Backend costs can parallelize - when compiling symbols A and B in
    /// parallel: `total_backend = max(A.backend_cost_ms, B.backend_cost_ms)`
    #[serde(default, skip_serializing_if = "is_zero")]
    #[schemars(range(min = 0.0))]
    pub backend_cost_ms: f64,

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
        /// Strategy for generating arbitrary Symbol values with non-negative
        /// integer costs.
        ///
        /// Note that floating-point values don't survive roundtrip
        /// serialization/deserialization perfectly.
        fn arb_symbol()
            (
                file in arb_name(),
                frontend_cost_ms in (0..1_000_000).prop_map(f64::from),
                backend_cost_ms in (0..1_000_000).prop_map(f64::from),
                dependencies in hash_set(arb_path(), 0..5),
                kind in arb_symbol_kind(),
            )
        -> Symbol {
            Symbol { file, frontend_cost_ms, backend_cost_ms, dependencies, kind }
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

    /// Strategy for generating arbitrary Crate values.
    fn arb_crate() -> impl Strategy<Value = Crate> {
        (
            (0..1_000_000).prop_map(f64::from),
            (0..1_000_000).prop_map(f64::from),
            arb_module(),
        )
            .prop_map(|(linking_ms, metadata_ms, root)| Crate {
                linking_ms,
                metadata_ms,
                root,
            })
    }

    prop_compose! {
        /// Strategy for generating arbitrary SymbolGraph values.
        fn arb_symbol_graph()
            (crates in hash_map(arb_name(), arb_crate(), 1..10))
        -> SymbolGraph {
            SymbolGraph { crates }
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
