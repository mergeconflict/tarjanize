//! Symbol path remapping and module tree reconstruction for condensed crates.
//!
//! After SCC condensation groups symbols into new synthetic crates, this module
//! handles two responsibilities:
//! 1. Computing old-path to new-path mappings (with conflict resolution)
//! 2. Rebuilding the nested module tree from the remapped symbols
//!
//! Conflict resolution: when two symbols from different original targets would
//! have the same path in the new crate, they are placed in
//! `conflict_from_{package_target}` submodules (with `/` replaced by `_`).

use std::collections::HashMap;

use tarjanize_schemas::{Module, Symbol, SymbolKind};

use crate::scc::{SymbolIndex, parse_bracketed_path};

/// Key for grouping symbols: (module path segments, symbol name).
///
/// Handles conflict detection: if two symbols from different original targets
/// would have the same path, they are placed in `conflict_from_{package_target}`
/// submodules (with `/` replaced by `_`).
///
/// Why: rewritten symbols must be reassembled into a valid nested module tree.
pub(crate) type SymbolPathKey = (Vec<String>, String);

/// Value for symbol grouping: list of (symbol index, original target).
///
/// The target identifier uses the `"package/target"` format (e.g.,
/// `"mypkg/lib"`, `"mypkg/test"`) rather than just the package name.
/// This ensures symbols from different targets of the same package
/// (lib vs test) produce distinct conflict submodules.
pub(crate) type SymbolOccurrences = Vec<(usize, String)>;

/// Computes a mapping from old symbol paths to new symbol paths.
///
/// This handles conflict resolution: if two symbols from different original
/// targets would have the same path in the new crate, they get placed in
/// `conflict_from_{package_target}` submodules (with `/` replaced by `_`).
///
/// Why: path uniqueness is required for a valid module tree and dependency rewrites.
pub(crate) fn compute_path_mapping(
    index: &SymbolIndex<'_>,
    set_crate_names: &[(usize, String, Vec<usize>)],
) -> HashMap<String, String> {
    let mut mapping = HashMap::new();

    for (_set_id, crate_name, symbol_indices) in set_crate_names {
        // Group symbols by their relative path (module path + symbol name).
        let mut path_to_symbols: HashMap<SymbolPathKey, SymbolOccurrences> =
            HashMap::new();

        for &symbol_idx in symbol_indices {
            let full_path = index.get_path(symbol_idx);
            let original_target = index.get_original_target(symbol_idx);

            // Parse path in new format: [package/target]::module::symbol
            let rest = if let Some((_, rest)) = parse_bracketed_path(full_path)
            {
                rest
            } else {
                // Fallback to old format: crate::module::symbol
                full_path.split_once("::").map_or(full_path, |(_, r)| r)
            };

            let parts: Vec<&str> = rest.split("::").collect();
            if parts.is_empty() {
                continue;
            }

            // Last part is symbol name, everything else is module path.
            let symbol_name = parts[parts.len() - 1].to_string();
            let module_parts: Vec<String> = parts[..parts.len() - 1]
                .iter()
                .map(|s| (*s).to_string())
                .collect();

            path_to_symbols
                .entry((module_parts, symbol_name))
                .or_default()
                .push((symbol_idx, original_target.to_string()));
        }

        // Compute new paths, handling conflicts.
        // Output paths use the same bracketed format as input: [package/synthetic]::path
        for ((module_path, symbol_name), occurrences) in path_to_symbols {
            if occurrences.len() == 1 {
                // No conflict - symbol keeps its relative path in the new crate.
                let (symbol_idx, _) = &occurrences[0];
                let old_path = index.get_path(*symbol_idx).to_string();

                let new_path = if module_path.is_empty() {
                    format!("[{crate_name}/synthetic]::{symbol_name}")
                } else {
                    format!(
                        "[{}/synthetic]::{}::{}",
                        crate_name,
                        module_path.join("::"),
                        symbol_name
                    )
                };

                mapping.insert(old_path, new_path);
            } else {
                // Conflict — use the full target ID (e.g., "pkg/lib")
                // with `/` sanitized to `_` for valid module names.
                for (symbol_idx, original_target) in &occurrences {
                    let old_path = index.get_path(*symbol_idx).to_string();

                    let conflict_module = format!(
                        "conflict_from_{}",
                        original_target.replace('/', "_")
                    );
                    let new_path = if module_path.is_empty() {
                        format!(
                            "[{crate_name}/synthetic]::{conflict_module}::{symbol_name}"
                        )
                    } else {
                        format!(
                            "[{}/synthetic]::{}::{conflict_module}::{symbol_name}",
                            crate_name,
                            module_path.join("::")
                        )
                    };

                    mapping.insert(old_path, new_path);
                }
            }
        }
    }

    mapping
}

/// Builds the nested module tree for a crate from its symbols.
///
/// Why: the rewritten `SymbolGraph` must reconstruct a valid module
/// hierarchy after path remapping and conflict shimming.
pub(crate) fn build_module_tree(
    index: &SymbolIndex<'_>,
    symbol_indices: &[usize],
    path_mapping: &HashMap<String, String>,
) -> Module {
    // Group symbols by their module path (relative to the new crate).
    let mut path_to_symbols: HashMap<SymbolPathKey, SymbolOccurrences> =
        HashMap::new();

    for &symbol_idx in symbol_indices {
        let full_path = index.get_path(symbol_idx);
        let original_target = index.get_original_target(symbol_idx);

        // Parse path in new format: [package/target]::module::symbol
        let rest = if let Some((_, rest)) = parse_bracketed_path(full_path) {
            rest
        } else {
            // Fallback to old format: crate::module::symbol
            full_path.split_once("::").map_or(full_path, |(_, r)| r)
        };

        let parts: Vec<&str> = rest.split("::").collect();
        if parts.is_empty() {
            continue; // Invalid path
        }

        // Last part is symbol name, everything else is module path.
        let symbol_name = parts[parts.len() - 1].to_string();
        let module_parts: Vec<String> = parts[..parts.len() - 1]
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        path_to_symbols
            .entry((module_parts, symbol_name))
            .or_default()
            .push((symbol_idx, original_target.to_string()));
    }

    // Build the module tree, handling conflicts.
    let mut root = Module {
        symbols: HashMap::new(),
        submodules: HashMap::new(),
    };

    for ((module_path, symbol_name), occurrences) in path_to_symbols {
        if occurrences.len() == 1 {
            // No conflict - place symbol at its original path.
            let (symbol_idx, _) = &occurrences[0];
            let symbol = index.get_symbol(*symbol_idx);
            let target_module = get_or_create_module(&mut root, &module_path);
            let new_symbol = rewrite_symbol(symbol, path_mapping);
            target_module.symbols.insert(symbol_name, new_symbol);
        } else {
            // Conflict — use the full target ID (e.g., "pkg/lib")
            // with `/` sanitized to `_` for valid module names.
            for (symbol_idx, original_target) in &occurrences {
                let symbol = index.get_symbol(*symbol_idx);

                // Create path with conflict submodule.
                let mut conflict_path = module_path.clone();
                conflict_path.push(format!(
                    "conflict_from_{}",
                    original_target.replace('/', "_")
                ));

                let target_module =
                    get_or_create_module(&mut root, &conflict_path);
                let new_symbol = rewrite_symbol(symbol, path_mapping);
                target_module
                    .symbols
                    .insert(symbol_name.clone(), new_symbol);
            }
        }
    }

    root
}

/// Rewrites a symbol's paths (dependencies and anchors) using the path mapping.
///
/// Why: dependency edges and anchor constraints must track the new crate layout.
fn rewrite_symbol(
    symbol: &Symbol,
    path_mapping: &HashMap<String, String>,
) -> Symbol {
    let rewrite = |path: &String| -> String {
        path_mapping
            .get(path)
            .cloned()
            .unwrap_or_else(|| path.clone())
    };

    let new_dependencies = symbol.dependencies.iter().map(rewrite).collect();

    let new_kind = match &symbol.kind {
        SymbolKind::ModuleDef { kind, visibility } => SymbolKind::ModuleDef {
            kind: kind.clone(),
            visibility: *visibility,
        },
        SymbolKind::Impl { name, anchors } => SymbolKind::Impl {
            name: name.clone(),
            anchors: anchors.iter().map(rewrite).collect(),
        },
    };

    Symbol {
        file: symbol.file.clone(),
        event_times_ms: symbol.event_times_ms.clone(),
        dependencies: new_dependencies,
        kind: new_kind,
    }
}

/// Gets or creates a nested module at the given path.
///
/// Why: module tree construction needs a single, shared builder for nesting.
fn get_or_create_module<'a>(
    root: &'a mut Module,
    path: &[String],
) -> &'a mut Module {
    let mut current = root;
    for segment in path {
        current =
            current
                .submodules
                .entry(segment.clone())
                .or_insert_with(|| Module {
                    symbols: HashMap::new(),
                    submodules: HashMap::new(),
                });
    }
    current
}
