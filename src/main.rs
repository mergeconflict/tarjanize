mod crates;
mod dependencies;
mod extract;
mod modules;
mod schemas;
mod workspaces;

use anyhow::Result;
use clap::Parser;

use extract::extract_symbol_graph;
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

    eprintln!("Loading workspace at: {}", args.workspace_path);
    let db = load_workspace(&args.workspace_path)?;

    // Derive workspace name from the path. In a real scenario, we might
    // read this from Cargo.toml's [workspace] section or the package name.
    let workspace_name = std::path::Path::new(&args.workspace_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("workspace")
        .to_string();

    // Extract the symbol graph from the workspace.
    let symbol_graph = extract_symbol_graph(&db, &workspace_name);

    eprintln!(
        "Extracted {} crate(s) with {} edges",
        symbol_graph.crates.len(),
        symbol_graph.edges.len()
    );

    // Serialize to JSON and print to stdout.
    // Using stdout for the JSON allows piping to other tools (jq, etc.)
    // while keeping status messages on stderr.
    let json = serde_json::to_string_pretty(&symbol_graph)?;
    println!("{}", json);

    Ok(())
}
