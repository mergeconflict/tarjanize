use std::fs::File;
use std::io::{BufWriter, Write};

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

/// Analyze Rust workspace dependency structures to identify opportunities for
/// splitting crates into smaller, parallelizable units for improved build times.
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
    /// Extract symbol graph from a Rust workspace
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
    // on stdout remains clean for piping. Default to warn, allowlist our crates.
    const CRATES: &[&str] =
        &["tarjanize", "tarjanize_extract", "tarjanize_schemas"];
    let level = cli.verbose.tracing_level_filter();
    let allowlist = CRATES.iter().map(|c| format!("{c}={level}")).join(",");
    let filter = EnvFilter::new(format!("warn,{allowlist}"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .init();

    match cli.command {
        Commands::Extract {
            workspace_path,
            output,
        } => {
            // Lock stdout once up front rather than on each write call.
            // Stdout must outlive the lock, so we bind it here first.
            let stdout = std::io::stdout();
            let mut writer: Box<dyn Write> = match output {
                Some(path) => Box::new(BufWriter::new(File::create(path)?)),
                None => Box::new(stdout.lock()),
            };
            tarjanize_extract::run(&workspace_path, &mut *writer)?;
            Ok(())
        }
    }
}
