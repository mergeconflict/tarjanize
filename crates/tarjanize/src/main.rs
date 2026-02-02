use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use anyhow::Result;
use clap::{Parser, Subcommand};
use clap_verbosity_flag::{InfoLevel, Verbosity};
use itertools::Itertools;
use mimalloc::MiMalloc;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

// Use mimalloc for better performance. Per M-MIMALLOC-APPS, this can provide
// up to 25% performance improvement for allocation-heavy workloads.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Crates to include in the logging allowlist.
const CRATES: &[&str] =
    &["tarjanize", "tarjanize_condense", "tarjanize_schemas"];

/// Analyze Rust workspace dependency structures to identify opportunities for
/// splitting crates into smaller, parallelizable units for improved build times.
///
/// For symbol extraction, use `cargo tarjanize` instead. This binary provides
/// post-processing commands for the extracted symbol graph.
#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(flatten)]
    verbose: Verbosity<InfoLevel>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Condense symbol graph into DAG of SCCs
    ///
    /// Computes strongly connected components from the symbol graph and produces
    /// a condensed graph where cycles become single nodes.
    Condense {
        /// Input `symbol_graph.json` file (reads from stdin if not specified)
        input: Option<String>,

        /// Output file path (writes to stdout if not specified)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Compute critical path cost of a symbol graph
    ///
    /// The critical path is the longest weighted path through the dependency
    /// graph. This represents the minimum build time with infinite parallelism.
    CriticalPath {
        /// Input `symbol_graph.json` file (reads from stdin if not specified)
        input: Option<String>,

        /// Show the full critical path (list of symbols)
        #[arg(short = 'p', long)]
        show_path: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize structured logging. Output goes to stderr so JSON output
    // on stdout remains clean for piping. Default to warn, allowlist our crates.
    let level = cli.verbose.tracing_level_filter();
    let allowlist = CRATES.iter().map(|c| format!("{c}={level}")).join(",");
    let filter = EnvFilter::new(format!("warn,{allowlist}"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .init();

    match cli.command {
        Commands::Condense { input, output } => {
            // Set up input reader.
            let stdin = std::io::stdin();
            let reader: Box<dyn std::io::Read> = match input {
                Some(path) => Box::new(BufReader::new(File::open(path)?)),
                None => Box::new(stdin.lock()),
            };

            // Set up output writer.
            let stdout = std::io::stdout();
            let mut writer: Box<dyn Write> = match output {
                Some(path) => Box::new(BufWriter::new(File::create(path)?)),
                None => Box::new(stdout.lock()),
            };

            tarjanize_condense::run(reader, &mut *writer)?;
            Ok(())
        }

        Commands::CriticalPath { input, show_path } => {
            let stdin = std::io::stdin();
            let reader: Box<dyn std::io::Read> = match input {
                Some(path) => Box::new(BufReader::new(File::open(path)?)),
                None => Box::new(stdin.lock()),
            };

            let result =
                tarjanize_cost::critical_path_from_reader(reader, show_path)?;

            println!("Critical path cost: {:.2} ms", result.cost);
            println!("Total cost:         {:.2} ms", result.total_cost);
            println!("Crate count:        {}", result.crate_count);
            println!("Symbol count:       {}", result.symbol_count);
            println!(
                "Parallelism ratio:  {:.2}x",
                result.total_cost / result.cost
            );

            if show_path && !result.path.is_empty() {
                println!("\nCritical path ({} crates):", result.path.len());
                for crate_name in &result.path {
                    println!("  {crate_name}");
                }
            }

            Ok(())
        }
    }
}
