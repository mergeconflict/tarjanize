//! SCC computation and condensed graph construction.
//!
//! Uses petgraph's `condensation` to find strongly connected components
//! and build a condensed DAG.

use std::collections::{BTreeSet, HashMap, HashSet};

use indexmap::IndexSet;
use petgraph::algo::condensation;
use petgraph::graph::{DiGraph, NodeIndex};
use tarjanize_schemas::{
    AnchorSet, CondensedGraph, Module, Scc, Symbol, SymbolGraph, SymbolKind,
};
use tracing::{debug, trace};

/// Indexes into a `SymbolGraph`.
///
/// Symbols and their paths are indexed by their insertion order; when we build
/// the view, we traverse the module tree and insert each symbol and its path
/// into `symbols` and `paths` respectively.
#[derive(Default)]
struct SymbolIndex<'a> {
    /// Symbols indexed by position.
    symbols: Vec<&'a Symbol>,
    /// Paths indexed by position. Also provides reverse lookup.
    paths: IndexSet<String>,
}

impl<'a> SymbolIndex<'a> {
    /// Builds an index by walking the entire `SymbolGraph`.
    fn build(symbol_graph: &'a SymbolGraph) -> Self {
        let mut index = SymbolIndex::default();
        for (crate_name, root_module) in &symbol_graph.crates {
            index.add_module(crate_name, root_module);
        }
        index
    }

    /// Recursively adds symbols from a module to the index.
    fn add_module(&mut self, module_path: &str, module: &'a Module) {
        for (symbol_name, symbol) in &module.symbols {
            let path = format!("{module_path}::{symbol_name}");
            self.paths.insert(path);
            self.symbols.push(symbol);
        }

        for (submodule_name, submodule) in &module.submodules {
            let submodule_path = format!("{module_path}::{submodule_name}");
            self.add_module(&submodule_path, submodule);
        }
    }

    /// Returns the index for a path, if it exists.
    fn get_index(&self, path: &str) -> Option<usize> {
        self.paths.get_index_of(path)
    }

    /// Returns the path at the given index.
    fn get_path(&self, index: usize) -> &str {
        self.paths.get_index(index).expect("valid index")
    }

    /// Returns the symbol at the given index.
    fn get_symbol(&self, index: usize) -> &'a Symbol {
        self.symbols[index]
    }

    /// Returns the number of symbols in the index.
    fn len(&self) -> usize {
        self.symbols.len()
    }
}

/// Computes SCCs and builds the condensed graph.
///
/// Uses petgraph's `condensation` function which:
/// - Finds all SCCs using Kosaraju's algorithm
/// - Returns a new graph where each node is a Vec of the original nodes in that SCC
/// - Edges between SCCs are preserved
#[expect(
    clippy::cast_possible_truncation,
    reason = "SCC count limited by symbol count; real workspaces have far fewer than 2^32 symbols"
)]
pub(crate) fn compute_condensed_graph(
    symbol_graph: &SymbolGraph,
) -> CondensedGraph {
    // Step 1: Build symbol index for O(1) lookups.
    let index = SymbolIndex::build(symbol_graph);

    // Step 2: Build DiGraph with symbol indices as node weights.
    let mut graph = DiGraph::<usize, ()>::with_capacity(index.len(), 0);
    (0..index.len()).for_each(|i| {
        graph.add_node(i);
    });

    // Add dependency edges.
    for from in 0..index.len() {
        for dep in &index.get_symbol(from).dependencies {
            if let Some(to) = index.get_index(dep) {
                graph.add_edge(NodeIndex::new(from), NodeIndex::new(to), ());
            }
        }
    }

    // Step 3: Run condensation. Returns DiGraph<Vec<usize>, ()>.
    // make_acyclic=true removes self-loops in the condensed graph.
    let condensed_petgraph = condensation(graph, true);

    // Step 4: Build symbol_to_scc mapping for impl anchor lookup.
    let mut symbol_to_scc = vec![0u32; index.len()];
    for node_idx in condensed_petgraph.node_indices() {
        let scc_id = node_idx.index() as u32;
        for &symbol_idx in &condensed_petgraph[node_idx] {
            symbol_to_scc[symbol_idx] = scc_id;
        }
    }

    // Step 5: Collect impl anchors for orphan rule constraints.
    let mut impl_anchors: HashMap<usize, AnchorSet> = HashMap::new();
    for i in 0..index.len() {
        let SymbolKind::Impl { anchors } = &index.get_symbol(i).kind else {
            continue;
        };
        let scc_ids: BTreeSet<u32> = anchors
            .iter()
            .filter_map(|p| index.get_index(p).map(|i| symbol_to_scc[i]))
            .collect();
        assert!(!scc_ids.is_empty(), "impl has no local anchors");
        impl_anchors.insert(i, AnchorSet { anchors: scc_ids });
    }

    // Step 6: Build final SCCs.
    // Petgraph's condensation returns SCCs in reverse topological order (dependencies
    // before dependents). We reverse to get dependents before dependencies, which is
    // the order Phase 3 expects for union-find processing.
    let mut sccs: Vec<Scc> = condensed_petgraph
        .node_indices()
        .map(|node_idx| {
            let symbol_indices = &condensed_petgraph[node_idx];
            let scc_id = node_idx.index() as u32;

            // Collect symbols.
            let symbols: HashSet<String> = symbol_indices
                .iter()
                .map(|&i| index.get_path(i).to_string())
                .collect();

            // Collect dependencies (edges to other SCCs).
            let dependencies: HashSet<u32> = condensed_petgraph
                .neighbors(node_idx)
                .map(|neighbor_idx| neighbor_idx.index() as u32)
                .collect();

            // Collect anchor sets for impl blocks in this SCC.
            let anchor_sets: HashSet<_> = symbol_indices
                .iter()
                .filter_map(|&i| impl_anchors.get(&i).cloned())
                .collect();

            trace!(
                scc_id,
                symbol_count = symbols.len(),
                dep_count = dependencies.len(),
                anchor_count = anchor_sets.len(),
                "created SCC"
            );

            Scc {
                id: scc_id,
                symbols,
                dependencies,
                anchor_sets,
            }
        })
        .collect();

    // Reverse to get topological order: dependents before dependencies.
    sccs.reverse();

    debug!(scc_count = sccs.len(), "built condensed graph");

    CondensedGraph { sccs }
}

#[cfg(test)]
#[expect(clippy::similar_names, reason = "test variable names like bar_scc/baz_scc are clear")]
mod tests {
    use tarjanize_schemas::Visibility;

    use super::*;

    /// Helper to create a simple symbol for testing.
    fn make_symbol(cost: f64, deps: &[&str]) -> Symbol {
        Symbol {
            file: "test.rs".to_string(),
            cost,
            dependencies: deps.iter().map(|&s| s.to_string()).collect(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        }
    }

    #[test]
    fn test_single_symbol_single_scc() {
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0, &[]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            Module {
                symbols,
                submodules: HashMap::new(),
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let condensed = compute_condensed_graph(&symbol_graph);

        assert_eq!(condensed.sccs.len(), 1);
        assert!(condensed.sccs[0].symbols.contains("my_crate::foo"));
        assert_eq!(condensed.sccs[0].symbols.len(), 1);
        assert!(condensed.sccs[0].dependencies.is_empty());
        assert!(condensed.sccs[0].anchor_sets.is_empty());
    }

    #[test]
    fn test_two_independent_symbols() {
        let mut symbols = HashMap::new();
        symbols.insert("foo".to_string(), make_symbol(10.0, &[]));
        symbols.insert("bar".to_string(), make_symbol(20.0, &[]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            Module {
                symbols,
                submodules: HashMap::new(),
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let condensed = compute_condensed_graph(&symbol_graph);

        // Two independent symbols = two SCCs.
        assert_eq!(condensed.sccs.len(), 2);
        // No dependencies between them.
        assert!(condensed.sccs.iter().all(|s| s.dependencies.is_empty()));
    }

    #[test]
    fn test_cycle_forms_single_scc() {
        let mut symbols = HashMap::new();
        // foo depends on bar, bar depends on foo (cycle).
        symbols
            .insert("foo".to_string(), make_symbol(10.0, &["my_crate::bar"]));
        symbols
            .insert("bar".to_string(), make_symbol(20.0, &["my_crate::foo"]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            Module {
                symbols,
                submodules: HashMap::new(),
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let condensed = compute_condensed_graph(&symbol_graph);

        // Cycle = one SCC containing both symbols.
        assert_eq!(condensed.sccs.len(), 1);
        assert_eq!(condensed.sccs[0].symbols.len(), 2);
        assert!(condensed.sccs[0].dependencies.is_empty());
    }

    #[test]
    fn test_chain_creates_dependencies() {
        let mut symbols = HashMap::new();
        // foo → bar → baz (chain, no cycles).
        symbols
            .insert("foo".to_string(), make_symbol(10.0, &["my_crate::bar"]));
        symbols
            .insert("bar".to_string(), make_symbol(20.0, &["my_crate::baz"]));
        symbols.insert("baz".to_string(), make_symbol(30.0, &[]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            Module {
                symbols,
                submodules: HashMap::new(),
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let condensed = compute_condensed_graph(&symbol_graph);

        // Three symbols, no cycles = three SCCs.
        assert_eq!(condensed.sccs.len(), 3);

        // Total dependencies = 2 (foo->bar, bar->baz).
        let total_deps: usize =
            condensed.sccs.iter().map(|s| s.dependencies.len()).sum();
        assert_eq!(total_deps, 2);

        // Find the SCCs by their symbol content.
        let foo_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("my_crate::foo"))
            .unwrap();
        let bar_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("my_crate::bar"))
            .unwrap();
        let baz_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("my_crate::baz"))
            .unwrap();

        // foo depends on bar.
        assert!(foo_scc.dependencies.contains(&bar_scc.id));
        // bar depends on baz.
        assert!(bar_scc.dependencies.contains(&baz_scc.id));
        // baz has no dependencies.
        assert!(baz_scc.dependencies.is_empty());
    }

    #[test]
    fn test_topological_order_dependents_before_dependencies() {
        // Verify SCCs are ordered with dependents before dependencies.
        // This ordering is required by Phase 3's union-find algorithm.
        let mut symbols = HashMap::new();
        // a → b → c (chain).
        symbols.insert("a".to_string(), make_symbol(10.0, &["my_crate::b"]));
        symbols.insert("b".to_string(), make_symbol(10.0, &["my_crate::c"]));
        symbols.insert("c".to_string(), make_symbol(10.0, &[]));

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            Module {
                symbols,
                submodules: HashMap::new(),
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let condensed = compute_condensed_graph(&symbol_graph);

        // Build position map: SCC ID → index in sccs vector.
        let position: HashMap<u32, usize> = condensed
            .sccs
            .iter()
            .enumerate()
            .map(|(i, scc)| (scc.id, i))
            .collect();

        // For every SCC, all its dependencies should appear later in the vector.
        for scc in &condensed.sccs {
            let scc_pos = position[&scc.id];
            for &dep_id in &scc.dependencies {
                let dep_pos = position[&dep_id];
                assert!(
                    scc_pos < dep_pos,
                    "SCC {} at position {} should come before its dependency {} at position {}",
                    scc.id,
                    scc_pos,
                    dep_id,
                    dep_pos
                );
            }
        }
    }

    #[test]
    fn test_cross_crate_dependency() {
        let mut crate_a_symbols = HashMap::new();
        crate_a_symbols
            .insert("foo".to_string(), make_symbol(10.0, &["crate_b::bar"]));

        let mut crate_b_symbols = HashMap::new();
        crate_b_symbols.insert("bar".to_string(), make_symbol(20.0, &[]));

        let mut crates = HashMap::new();
        crates.insert(
            "crate_a".to_string(),
            Module {
                symbols: crate_a_symbols,
                submodules: HashMap::new(),
            },
        );
        crates.insert(
            "crate_b".to_string(),
            Module {
                symbols: crate_b_symbols,
                submodules: HashMap::new(),
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let condensed = compute_condensed_graph(&symbol_graph);

        // Two SCCs (one per crate).
        assert_eq!(condensed.sccs.len(), 2);

        // Find the SCCs by their symbol content.
        let foo_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("crate_a::foo"))
            .unwrap();
        let bar_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("crate_b::bar"))
            .unwrap();

        // foo depends on bar (cross-crate).
        assert!(foo_scc.dependencies.contains(&bar_scc.id));
        assert!(bar_scc.dependencies.is_empty());
    }

    #[test]
    fn test_impl_anchors_captured() {
        // Create a struct, trait, and impl.
        let mut symbols = HashMap::new();
        symbols.insert(
            "Foo".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                cost: 0.0,
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Struct".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        symbols.insert(
            "MyTrait".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                cost: 0.0,
                dependencies: HashSet::new(),
                kind: SymbolKind::ModuleDef {
                    kind: "Trait".to_string(),
                    visibility: Visibility::Public,
                },
            },
        );
        symbols.insert(
            "impl MyTrait for Foo".to_string(),
            Symbol {
                file: "test.rs".to_string(),
                cost: 0.0,
                dependencies: [
                    "my_crate::Foo".to_string(),
                    "my_crate::MyTrait".to_string(),
                ]
                .into_iter()
                .collect(),
                kind: SymbolKind::Impl {
                    anchors: [
                        "my_crate::Foo".to_string(),
                        "my_crate::MyTrait".to_string(),
                    ]
                    .into_iter()
                    .collect(),
                },
            },
        );

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            Module {
                symbols,
                submodules: HashMap::new(),
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let condensed = compute_condensed_graph(&symbol_graph);

        // No artificial cycles, so we get 3 SCCs.
        // Foo and MyTrait are independent, impl depends on both.
        assert_eq!(condensed.sccs.len(), 3);

        // Find the impl's SCC.
        let impl_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.iter().any(|p| p.contains("impl")))
            .unwrap();

        // The impl should have one anchor set entry.
        assert_eq!(impl_scc.anchor_sets.len(), 1);

        // Find the type and trait SCCs.
        let foo_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("my_crate::Foo"))
            .unwrap();
        let trait_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("my_crate::MyTrait"))
            .unwrap();

        // The anchor set should reference both type and trait SCCs.
        let anchor_set = impl_scc.anchor_sets.iter().next().unwrap();
        assert!(anchor_set.anchors.contains(&foo_scc.id));
        assert!(anchor_set.anchors.contains(&trait_scc.id));
        assert_eq!(anchor_set.anchors.len(), 2);
    }

    #[test]
    fn test_empty_graph() {
        let symbol_graph = SymbolGraph {
            crates: HashMap::new(),
        };

        let condensed = compute_condensed_graph(&symbol_graph);

        assert!(condensed.sccs.is_empty());
    }

    #[test]
    fn test_submodule_symbol() {
        let mut inner_symbols = HashMap::new();
        inner_symbols.insert("bar".to_string(), make_symbol(20.0, &[]));

        let mut root_symbols = HashMap::new();
        root_symbols.insert(
            "foo".to_string(),
            make_symbol(10.0, &["my_crate::inner::bar"]),
        );

        let mut submodules = HashMap::new();
        submodules.insert(
            "inner".to_string(),
            Module {
                symbols: inner_symbols,
                submodules: HashMap::new(),
            },
        );

        let mut crates = HashMap::new();
        crates.insert(
            "my_crate".to_string(),
            Module {
                symbols: root_symbols,
                submodules,
            },
        );

        let symbol_graph = SymbolGraph { crates };
        let condensed = compute_condensed_graph(&symbol_graph);

        // Two symbols = two SCCs.
        assert_eq!(condensed.sccs.len(), 2);

        // Find the SCCs.
        let foo_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("my_crate::foo"))
            .unwrap();
        let bar_scc = condensed
            .sccs
            .iter()
            .find(|s| s.symbols.contains("my_crate::inner::bar"))
            .unwrap();

        // foo depends on bar.
        assert!(foo_scc.dependencies.contains(&bar_scc.id));
    }
}
