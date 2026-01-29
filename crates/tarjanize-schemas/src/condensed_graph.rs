//! Condensed graph schema for representing SCCs and their dependencies.
//!
//! The condensed graph is a DAG where each node is a strongly connected
//! component (SCC) from the original symbol graph. This is the output of
//! Phase 2 (SCC computation) and the input to Phase 3 (optimal partitioning).
//!
//! SCCs are stored in topological order (dependents before dependencies),
//! which is the order needed for Phase 3's union-find algorithm.
//!
//! ## Impl Anchors
//!
//! Rust's orphan rule requires trait impls to be in the same crate as either
//! the trait or at least one of the covered types. Phase 2 captures each
//! impl's "anchor set": the SCCs containing local types/traits that can
//! satisfy this rule. Phase 3 uses these anchor sets when grouping SCCs.

use std::collections::{BTreeSet, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Root structure representing the condensed graph of a workspace.
///
/// The condensed graph is a complete flattening of the `SymbolGraph` structure.
/// All SCCs from all crates are stored in a single vector, topologically
/// sorted so that dependent SCCs appear before their dependencies. This
/// ordering is essential for Phase 3's union-find merging algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CondensedGraph {
    /// All SCCs in the workspace, topologically sorted (dependents before
    /// dependencies). Each SCC knows which crate it belongs to and which
    /// other SCCs it depends on.
    pub sccs: Vec<Scc>,
}

/// A strongly connected component (SCC) in the dependency graph.
///
/// An SCC is a maximal set of symbols where every symbol can reach every
/// other symbol through dependencies. Single-symbol SCCs represent symbols
/// with no cyclic dependencies. Multi-symbol SCCs represent cycles that
/// must stay together in any valid crate split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Scc {
    /// Unique identifier for this SCC. Generated sequentially during
    /// condensation.
    pub id: u32,

    /// Fully qualified paths of symbols in this SCC. These paths reference
    /// symbols in the original `SymbolGraph`.
    #[schemars(length(min = 1))]
    pub symbols: HashSet<String>,

    /// IDs of SCCs that this SCC depends on. These are the SCCs containing
    /// symbols that symbols in this SCC reference. Due to topological
    /// ordering, all referenced IDs appear later in the sccs vector.
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub dependencies: HashSet<u32>,

    /// Anchor sets for impl blocks in this SCC.
    ///
    /// Each impl block has an anchor set: the SCCs containing workspace-local
    /// types/traits that can satisfy the orphan rule. When merging SCCs,
    /// Phase 3 must solve a hitting set problem to find SCCs that satisfy
    /// all anchor constraints. Stored as a set to deduplicate identical anchors.
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub anchor_sets: HashSet<AnchorSet>,
}

/// Anchors for an impl block.
///
/// An anchor is an SCC containing a workspace-local type or trait that can
/// satisfy the orphan rule. For `impl<P1..=Pn> Trait<T1..=Tn> for T0`, anchors
/// include:
/// - The trait (if workspace-local)
/// - The self type T0 (if workspace-local)
/// - The trait's type parameters T1..=Tn (if workspace-local)
///
/// The impl must end up in the same output crate as at least one anchor.
/// When merging SCCs with multiple anchor sets, Phase 3 must solve a
/// **hitting set problem**: find a minimal set of SCCs that satisfies all
/// anchor sets. For example, if one impl has anchors {x, y} and another has
/// {x, z}, the minimal solution is {x} since it hits both sets.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct AnchorSet {
    /// SCC IDs containing types/traits that anchor this impl. The impl must
    /// end up in the same output crate as at least one of these SCCs to
    /// satisfy the orphan rule. Uses `BTreeSet` for deterministic iteration
    /// order (required for Hash impl).
    #[schemars(length(min = 1))]
    pub anchors: BTreeSet<u32>,
}

#[cfg(test)]
mod tests {
    use proptest::collection::{btree_set, hash_set, vec};
    use proptest::prelude::*;

    use super::*;
    use crate::testutil::arb_path;

    // -------------------------------------------------------------------------
    // Proptest strategies for generating arbitrary schema instances.
    //
    // These strategies generate bounded instances to ensure reasonable test
    // performance. The roundtrip test verifies that any generated instance
    // survives JSON serialization and deserialization.
    // -------------------------------------------------------------------------

    prop_compose! {
        /// Strategy for generating arbitrary AnchorSet values.
        /// Anchors must be non-empty (min 1 element).
        fn arb_anchor_set()
            (anchors in btree_set(any::<u32>(), 1..5))
        -> AnchorSet {
            AnchorSet { anchors }
        }
    }

    prop_compose! {
        /// Strategy for generating arbitrary Scc values.
        /// Symbols must be non-empty (min 1 element).
        fn arb_scc()
            (
                id in any::<u32>(),
                symbols in hash_set(arb_path(), 1..8),
                dependencies in hash_set(any::<u32>(), 0..5),
                anchor_sets in hash_set(arb_anchor_set(), 0..3),
            )
        -> Scc {
            Scc { id, symbols, dependencies, anchor_sets }
        }
    }

    prop_compose! {
        /// Strategy for generating arbitrary CondensedGraph values.
        fn arb_condensed_graph()
            (sccs in vec(arb_scc(), 0..10))
        -> CondensedGraph {
            CondensedGraph { sccs }
        }
    }

    proptest! {
        /// Test serialization roundtrip for arbitrary CondensedGraph instances.
        ///
        /// This exercises the Serialize/Deserialize derives by generating
        /// arbitrary graphs and verifying they survive a JSON roundtrip.
        #[test]
        fn test_condensed_graph_roundtrip(graph in arb_condensed_graph()) {
            let json = serde_json::to_string(&graph).expect("serialize");
            let parsed: CondensedGraph =
                serde_json::from_str(&json).expect("deserialize");
            prop_assert_eq!(parsed, graph);
        }
    }
}
