//! Extraction of symbol graphs from rust-analyzer's semantic model.
//!
//! This module is the entry point for Phase 1 of tarjanize. It coordinates
//! the extraction process by iterating workspace crates and assembling the
//! final SymbolGraph.
//!
//! The actual extraction logic lives in:
//! - `crates.rs`: Crate-level extraction
//! - `modules.rs`: Module and symbol extraction
//! - `dependencies.rs`: Dependency analysis

use std::collections::HashSet;

use ra_ap_base_db::SourceDatabase;
use ra_ap_hir::{Crate, Semantics};
use ra_ap_ide_db::RootDatabase;

use crate::crates::extract_crate;
use crate::schemas::SymbolGraph;

/// Extracts a complete symbol graph from a rust-analyzer database.
///
/// This is the main entry point for Phase 1 of tarjanize. It walks all
/// workspace crates and extracts their symbols and dependency relationships.
///
/// The workspace_name parameter is provided by the caller (typically derived
/// from the root directory or Cargo.toml) since rust-analyzer doesn't have
/// a single "workspace name" concept.
pub fn extract_symbol_graph(db: &RootDatabase, workspace_name: &str) -> SymbolGraph {
    let sema = Semantics::new(db);
    let crate_graph = db.crate_graph();

    // Filter to workspace members. The crate graph includes ALL crates
    // (workspace + all transitive dependencies), but we only analyze
    // workspace members since those are what we might split.
    let workspace_crates: Vec<_> = crate_graph
        .iter()
        .filter(|&krate| crate_graph[krate].origin.is_local())
        .map(Crate::from)
        .collect();

    let mut edges = HashSet::new();
    let mut crates = Vec::new();

    for krate in workspace_crates {
        let crate_module = extract_crate(&sema, krate, &mut edges);
        crates.push(crate_module);
    }

    SymbolGraph {
        workspace_name: workspace_name.to_string(),
        crates,
        edges,
    }
}
