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

    /// Compute build costs and critical path of a symbol graph
    ///
    /// Shows per-crate costs and the critical path (longest weighted path through
    /// the dependency graph), which represents minimum build time with infinite
    /// parallelism.
    Cost {
        /// Input `symbol_graph.json` file (reads from stdin if not specified)
        input: Option<String>,
    },
}

/// Prints a single target's details in table format.
fn print_target_detail(target_detail: &tarjanize_cost::TargetOnPath) {
    // Format dependencies: show first few, then count if many.
    let deps_str = if target_detail.dependencies.is_empty() {
        "(none)".to_string()
    } else if target_detail.dependencies.len() <= 3 {
        target_detail.dependencies.join(", ")
    } else {
        format!(
            "{}, ... (+{} more)",
            target_detail.dependencies[..3].join(", "),
            target_detail.dependencies.len() - 3
        )
    };

    println!(
        "{:>12.2}  {:>12.2}  {:<40}  {}",
        target_detail.cost,
        target_detail.cumulative_cost,
        target_detail.name,
        deps_str
    );
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

        Commands::Cost { input } => {
            let stdin = std::io::stdin();
            let reader: Box<dyn std::io::Read> = match input {
                Some(path) => Box::new(BufReader::new(File::open(path)?)),
                None => Box::new(stdin.lock()),
            };

            let result = tarjanize_cost::critical_path_from_reader(reader)?;

            println!("Critical path cost: {:.2} ms", result.cost);
            println!("Total cost:         {:.2} ms", result.total_cost);
            println!("Target count:       {}", result.target_count);
            println!("Symbol count:       {}", result.symbol_count);
            println!(
                "Parallelism ratio:  {:.2}x",
                result.total_cost / result.cost
            );

            if !result.path_details.is_empty() {
                println!("\nCritical path ({} targets):\n", result.path.len());

                println!(
                    "{:>12}  {:>12}  {:<40}  Dependencies",
                    "Cost (ms)", "Cumulative", "Target"
                );
                println!("{}", "-".repeat(100));

                for target_detail in &result.path_details {
                    print_target_detail(target_detail);
                }
            }

            if !result.all_targets.is_empty() {
                println!(
                    "\nAll targets by cost ({} targets):\n",
                    result.all_targets.len()
                );

                println!(
                    "{:>12}  {:>12}  {:<40}  Dependencies",
                    "Cost (ms)", "Cumulative", "Target"
                );
                println!("{}", "-".repeat(100));

                for target_detail in &result.all_targets {
                    print_target_detail(target_detail);
                }
            }

            Ok(())
        }
    }
}
