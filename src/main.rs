// Use mimalloc for better performance. Per M-MIMALLOC-APPS, this can provide
// up to 25% performance improvement for allocation-heavy workloads.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod extract;
mod schemas;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{Level, info, warn};
use tracing_subscriber::EnvFilter;

use extract::extract_symbol_graph;
use extract::workspaces::load_workspace;

/// Analyze Rust workspace dependency structures to identify opportunities for
/// splitting crates into smaller, parallelizable units for improved build times.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Extract symbol graph from a Rust workspace (Phase 1)
    ///
    /// Analyzes all workspace member crates and produces a JSON file containing
    /// all symbols and their dependency relationships.
    Extract {
        /// Path to the workspace root (directory containing Cargo.toml)
        #[arg(default_value = ".")]
        workspace_path: String,

        /// Output file path (writes to stdout if not specified)
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize structured logging. Output goes to stderr so JSON output
    // on stdout remains clean for piping.
    let filter = match cli.verbose {
        0 => EnvFilter::new("tarjanize=info"),
        1 => EnvFilter::new("tarjanize=debug"),
        _ => EnvFilter::new("tarjanize=trace"),
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_max_level(Level::TRACE)
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
        Commands::Extract {
            workspace_path,
            output,
        } => run_extract(&workspace_path, output.as_deref()),
    }
}

/// Run the extract subcommand (Phase 1).
///
/// Loads a workspace, extracts the symbol graph, and writes it to the
/// specified output (file or stdout).
fn run_extract(workspace_path: &str, output: Option<&str>) -> Result<()> {
    info!(path = %workspace_path, "workspace.loading");
    let (db, vfs) = load_workspace(workspace_path)?;

    // Derive workspace name from Cargo.toml, falling back to directory name.
    let workspace_name = read_workspace_name(workspace_path).unwrap_or_else(|| {
        let name = Path::new(workspace_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("workspace")
            .to_string();
        warn!(
            fallback = %name,
            "Could not read workspace name from Cargo.toml, using directory name"
        );
        name
    });

    // Extract the symbol graph from the workspace.
    let symbol_graph = extract_symbol_graph(&db, &vfs, &workspace_name);

    info!(
        crates = symbol_graph.crates.len(),
        edges = symbol_graph.edges.len(),
        "extraction.complete"
    );

    // Serialize to JSON.
    let json = serde_json::to_string_pretty(&symbol_graph)?;

    // Write to output file or stdout.
    match output {
        Some(path) => {
            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{}", json)?;
            info!(path = %path, "output.written");
        }
        None => {
            // Write to stdout for piping to other tools.
            println!("{}", json);
        }
    }

    Ok(())
}

/// Read the workspace/package name from Cargo.toml.
///
/// Tries to read [package].name first, then falls back to parsing the
/// [workspace] section if present. Returns None if Cargo.toml doesn't
/// exist or doesn't contain a usable name.
fn read_workspace_name(workspace_path: &str) -> Option<String> {
    let cargo_toml_path = Path::new(workspace_path).join("Cargo.toml");
    let contents = std::fs::read_to_string(&cargo_toml_path).ok()?;
    let toml: toml::Table = contents.parse().ok()?;

    // Try [package].name first (most common case: single-crate workspace)
    if let Some(package) = toml.get("package").and_then(|p| p.as_table())
        && let Some(name) = package.get("name").and_then(|n| n.as_str())
    {
        return Some(name.to_string());
    }

    // For multi-crate workspaces, there's no standard "name" field.
    // We could parse [workspace].members but that gives crate names, not
    // a workspace name. Just return None and let caller use directory name.
    None
}
