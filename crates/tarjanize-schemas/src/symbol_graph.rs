//! Symbol graph schema for representing extracted dependency information.
//!
//! The symbol graph captures all symbols (functions, structs, traits, impl
//! blocks, etc.) in a workspace along with their dependency relationships.
//! This is the output of the extraction phase and an input to subsequent
//! analysis phases.

use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Root structure representing the entire symbol graph of a workspace.
///
/// The symbol graph captures all symbols and their dependencies within a
/// Rust workspace. Crates are represented as their root modules, which
/// contain symbols and nested submodules. Dependencies are stored directly
/// on each symbol rather than as a separate edge list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SymbolGraph {
    /// All crates in the workspace, keyed by crate name. Each crate is
    /// represented as its root module, which contains all symbols and
    /// submodules.
    pub crates: HashMap<String, Module>,
}

/// A module (or crate root) containing symbols and submodules.
///
/// Modules form a tree structure where each module can contain:
/// - Symbols (functions, structs, enums, traits, impl blocks, etc.)
/// - Child submodules
///
/// The crate's root module (lib.rs or main.rs) is the top of this tree.
/// Both module and symbol names are stored as HashMap keys, not in the
/// structs themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
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
/// source file, compilation cost estimate, dependencies, and kind-specific
/// details. Symbol names are stored as keys in the parent Module's HashMap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Symbol {
    /// Path to the symbol's file relative to the crate root.
    pub file: String,

    /// Approximate cost (unitless) of compiling this symbol. Given
    /// two symbols A and B:
    /// - The cost of compiling A and B in sequence is A.cost + B.cost
    /// - The cost of compiling A and B in parallel is max(A.cost, B.cost)
    ///
    /// Currently estimated from syntax node size in bytes.
    #[schemars(range(min = 0.0))]
    pub cost: f64,

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
        /// Uses PascalCase to match rust-analyzer's `SymbolKind` enum.
        kind: String,

        /// Visibility for crate-splitting purposes. Defaults to `NonPublic`
        /// and is omitted from serialization when non-public.
        #[serde(default, skip_serializing_if = "Visibility::is_non_public")]
        visibility: Visibility,
    },

    /// An impl block, analogous to `ra_ap_hir::Impl`.
    ///
    /// Impl blocks are distinct from ModuleDefs; they don't have visibility,
    /// and Rust's orphan rules dictate where they can be defined.
    Impl {
        /// Fully qualified path to the self type (the type being implemented).
        /// Present when the self type is a struct, enum or union; absent for
        /// type parameters, primitives, references, `dyn Trait`, etc.
        #[serde(skip_serializing_if = "Option::is_none")]
        self_type: Option<String>,

        /// Fully qualified path to the trait being implemented. Present for
        /// trait impls, absent for inherent impls.
        #[serde(rename = "trait", skip_serializing_if = "Option::is_none")]
        trait_: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use proptest::collection::{hash_map, hash_set};
    use proptest::prelude::*;

    use super::*;

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

    /// Strategy for generating arbitrary identifier-like symbol names.
    fn arb_name() -> impl Strategy<Value = String> {
        "[a-z_][a-z0-9_]{0,19}"
    }

    /// Strategy for generating optional names.
    fn arb_opt_name() -> impl Strategy<Value = Option<String>> {
        prop::option::of(arb_name())
    }

    /// Strategy for generating arbitrary Visibility values.
    fn arb_visibility() -> impl Strategy<Value = Visibility> {
        prop_oneof![Just(Visibility::Public), Just(Visibility::NonPublic)]
    }

    /// Strategy for generating arbitrary SymbolKind values.
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
                (self_type in arb_opt_name(), trait_ in arb_opt_name())
            -> SymbolKind {
                SymbolKind::Impl { self_type, trait_ }
            }
        }

        prop_oneof![arb_module_def(), arb_impl()]
    }

    prop_compose! {
        /// Strategy for generating arbitrary Symbol values with non-negative
        /// integer cost.
        ///
        /// Note that floating-point values don't survive roundtrip
        /// serialization/deserialization perfectly.
        fn arb_symbol()
            (
                file in arb_name(),
                cost in (0..1_000_000).prop_map(f64::from),
                dependencies in hash_set(arb_name(), 0..5),
                kind in arb_symbol_kind(),
            )
        -> Symbol {
            Symbol { file, cost, dependencies, kind }
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

    prop_compose! {
        /// Strategy for generating arbitrary SymbolGraph values.
        fn arb_symbol_graph()
            (crates in hash_map(arb_name(), arb_module(), 1..10))
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
