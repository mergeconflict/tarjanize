//! SCC computation and condensed graph construction.
//!
//! Uses petgraph's `condensation` to find strongly connected components
//! and build a condensed DAG.

use std::collections::{BTreeSet, HashSet};

use indexmap::IndexSet;
use petgraph::algo::condensation;
use petgraph::graph::{DiGraph, NodeIndex};
use tarjanize_schemas::{
    AnchorSet, CondensedGraph, Module, Scc, Symbol, SymbolGraph, SymbolKind,
};

/// Extracts impl anchors from a symbol kind, if it's an impl.
fn impl_anchors(kind: &SymbolKind) -> Option<&HashSet<String>> {
    match kind {
        SymbolKind::Impl { anchors, .. } => Some(anchors),
        SymbolKind::ModuleDef { .. } => None,
    }
}

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
    for i in 0..index.len() {
        graph.add_node(i);
    }
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

    // Step 5: Build final SCCs.
    // Petgraph's condensation returns SCCs in postorder (dependencies before dependents),
    // which is the natural compilation order.
    let sccs: Vec<Scc> = condensed_petgraph
        .node_indices()
        .map(|node_idx| {
            let symbol_indices = &condensed_petgraph[node_idx];

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
            // Filter out anchors that don't exist in the symbol index - these are
            // external types that were included during AST-based extraction.
            let anchor_sets: HashSet<_> = symbol_indices
                .iter()
                .filter_map(|&i| impl_anchors(&index.get_symbol(i).kind))
                .filter_map(|anchors| {
                    let valid_anchors: BTreeSet<u32> = anchors
                        .iter()
                        .filter_map(|p| {
                            index.get_index(p).map(|idx| symbol_to_scc[idx])
                        })
                        .collect();
                    // Only include anchor sets that have at least one valid anchor.
                    (!valid_anchors.is_empty()).then_some(AnchorSet {
                        anchors: valid_anchors,
                    })
                })
                .collect();

            Scc {
                symbols,
                dependencies,
                anchor_sets,
            }
        })
        .collect();
    CondensedGraph { sccs }
}

#[cfg(test)]
#[expect(
    clippy::similar_names,
    reason = "test variable names like bar_scc/baz_scc are clear"
)]
mod tests {
    use std::collections::HashMap;

    use tarjanize_schemas::Visibility;

    use super::*;

    /// Helper to find the index of an SCC containing a symbol.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "test helper, SCC count fits in u32"
    )]
    fn find_scc_idx(condensed: &CondensedGraph, symbol: &str) -> u32 {
        condensed
            .sccs
            .iter()
            .position(|s| s.symbols.contains(symbol))
            .unwrap() as u32
    }

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
        let foo_idx = find_scc_idx(&condensed, "my_crate::foo");
        let bar_idx = find_scc_idx(&condensed, "my_crate::bar");
        let baz_idx = find_scc_idx(&condensed, "my_crate::baz");

        let foo_scc = &condensed.sccs[foo_idx as usize];
        let bar_scc = &condensed.sccs[bar_idx as usize];
        let baz_scc = &condensed.sccs[baz_idx as usize];

        // foo depends on bar.
        assert!(foo_scc.dependencies.contains(&bar_idx));
        // bar depends on baz.
        assert!(bar_scc.dependencies.contains(&baz_idx));
        // baz has no dependencies.
        assert!(baz_scc.dependencies.is_empty());
    }

    #[test]
    fn test_postorder_dependencies_before_dependents() {
        // Verify SCCs are in postorder: dependencies before dependents.
        // This is the natural compilation order.
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

        // For every SCC, all its dependencies should appear earlier in the vector.
        for (scc_idx, scc) in condensed.sccs.iter().enumerate() {
            for &dep_idx in &scc.dependencies {
                assert!(
                    (dep_idx as usize) < scc_idx,
                    "dependency {dep_idx} should come before SCC {scc_idx}"
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
        let foo_idx = find_scc_idx(&condensed, "crate_a::foo");
        let bar_idx = find_scc_idx(&condensed, "crate_b::bar");

        let foo_scc = &condensed.sccs[foo_idx as usize];
        let bar_scc = &condensed.sccs[bar_idx as usize];

        // foo depends on bar (cross-crate).
        assert!(foo_scc.dependencies.contains(&bar_idx));
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
                    name: "impl MyTrait for Foo".to_string(),
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
        let impl_idx = condensed
            .sccs
            .iter()
            .position(|s| s.symbols.iter().any(|p| p.contains("impl")))
            .unwrap();
        let impl_scc = &condensed.sccs[impl_idx];

        // The impl should have one anchor set entry.
        assert_eq!(impl_scc.anchor_sets.len(), 1);

        // Find the type and trait SCCs.
        let foo_idx = find_scc_idx(&condensed, "my_crate::Foo");
        let trait_idx = find_scc_idx(&condensed, "my_crate::MyTrait");

        // The anchor set should reference both type and trait SCCs.
        let anchor_set = impl_scc.anchor_sets.iter().next().unwrap();
        assert!(anchor_set.anchors.contains(&foo_idx));
        assert!(anchor_set.anchors.contains(&trait_idx));
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
        let foo_idx = find_scc_idx(&condensed, "my_crate::foo");
        let bar_idx = find_scc_idx(&condensed, "my_crate::inner::bar");

        let foo_scc = &condensed.sccs[foo_idx as usize];

        // foo depends on bar.
        assert!(foo_scc.dependencies.contains(&bar_idx));
    }
}
