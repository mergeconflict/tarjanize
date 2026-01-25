mod crates;
mod dependencies;
mod modules;
mod test_method_resolution;
mod workspaces;

use anyhow::Result;
use clap::Parser;
use ra_ap_base_db::SourceDatabase;
use ra_ap_hir::{Crate, Semantics};

use crates::visit_crate;
use workspaces::load_workspace;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Inspect a Rust workspace using rust-analyzer API"
)]
struct Args {
    /// Path to the workspace root (directory containing Cargo.toml)
    #[arg(default_value = ".")]
    workspace_path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading workspace at: {}", args.workspace_path);
    let db = load_workspace(&args.workspace_path)?;

    // Semantics provides name resolution and type inference for syntax nodes.
    // We need it to resolve paths to their definitions.
    let sema = Semantics::new(&db);

    println!("\n=== Workspace Crates ===\n");

    // The crate graph includes ALL crates: workspace members plus all
    // transitive dependencies from crates.io. For tarjanize, we only care
    // about workspace members (the crates we're analyzing for splitting).
    let crate_graph = db.crate_graph();

    // Filter to workspace members using CrateOrigin. rust-analyzer tracks
    // each crate's provenance: Local (workspace members), Library (crates.io
    // dependencies), Lang (std/core/alloc), or Rustc (compiler crates).
    let workspace_crates: Vec<_> = crate_graph
        .iter()
        .filter(|&krate| crate_graph[krate].origin.is_local())
        .collect();

    println!(
        "Found {} workspace crate(s) ({} total including external dependencies):\n",
        workspace_crates.len(),
        crate_graph.len()
    );

    for krate in workspace_crates {
        let hir_crate = Crate::from(krate);
        visit_crate(&sema, hir_crate);
    }

    Ok(())
}
