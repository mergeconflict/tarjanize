use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

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
const CRATES: &[&str] = &[
    "tarjanize",
    "tarjanize_condense",
    "tarjanize_schemas",
    "tarjanize_viz",
];

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

/// Common I/O arguments for commands that read/write files or stdio.
#[derive(Parser)]
struct IoArgs {
    /// Input file (reads from stdin if not specified)
    #[arg(short, long)]
    input: Option<String>,

    /// Output file (writes to stdout if not specified)
    #[arg(short, long)]
    output: Option<String>,
}

impl IoArgs {
    /// Sets up stdio/file I/O and calls the provided function.
    fn run<F>(self, f: F) -> Result<()>
    where
        F: FnOnce(Box<dyn Read>, Box<dyn Write>) -> Result<()>,
    {
        let stdin = std::io::stdin();
        let reader: Box<dyn Read> = match self.input {
            Some(path) => Box::new(BufReader::new(File::open(path)?)),
            None => Box::new(stdin.lock()),
        };

        let stdout = std::io::stdout();
        let writer: Box<dyn Write> = match self.output {
            Some(path) => Box::new(BufWriter::new(File::create(path)?)),
            None => Box::new(stdout.lock()),
        };

        f(reader, writer)
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Condense symbol graph into DAG of SCCs
    ///
    /// Computes strongly connected components from the symbol graph and produces
    /// a condensed graph where cycles become single nodes.
    Condense {
        #[command(flatten)]
        io: IoArgs,

        /// Path to a `CostModel` JSON file (from `tarjanize cost --output-model`).
        /// When provided, uses pre-fitted regression coefficients for synthetic
        /// crate wall time predictions instead of re-fitting internally.
        #[arg(long, value_name = "PATH")]
        model: Option<String>,
    },

    /// Compute build costs and critical path of a symbol graph
    ///
    /// Shows per-crate costs and the critical path (longest weighted path through
    /// the dependency graph), which represents minimum build time with infinite
    /// parallelism.
    Cost {
        #[command(flatten)]
        io: IoArgs,

        /// Fit the cost model using lib targets only.
        #[arg(long)]
        fit_libs_only: bool,

        /// Write the fitted `CostModel` as JSON to this path.
        /// The model can later be consumed by `tarjanize condense --model`.
        #[arg(long, value_name = "PATH")]
        output_model: Option<String>,
    },

    /// Visualize the build schedule as an interactive HTML report
    ///
    /// Generates a self-contained HTML file with a `PixiJS` canvas-based Gantt
    /// chart. Hover over targets to see the critical path through them.
    Viz {
        #[command(flatten)]
        io: IoArgs,

        /// Path to the fitted cost model (from `tarjanize cost --output-model`).
        /// When provided, uses model predictions for target costs.
        /// Without a model, falls back to wall-clock or per-symbol costs.
        #[arg(long, value_name = "PATH")]
        model: Option<PathBuf>,
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
        Commands::Condense { io, model } => {
            // Load the external cost model if a path was provided.
            let cost_model = model
                .map(|p| {
                    tarjanize_schemas::load_cost_model(std::path::Path::new(&p))
                })
                .transpose()?;
            io.run(|r, w| {
                Ok(tarjanize_condense::run(r, w, cost_model.as_ref())?)
            })
        }
        Commands::Viz { io, model } => {
            // Load the cost model if a path was provided.
            let cost_model = model
                .map(|p| tarjanize_schemas::load_cost_model(&p))
                .transpose()?;
            io.run(|r, w| {
                tarjanize_viz::run(r, cost_model.as_ref(), w)
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
        }
        Commands::Cost {
            io,
            fit_libs_only,
            output_model,
        } => io.run(|mut r, w| {
            let mut json = String::new();
            r.read_to_string(&mut json)?;
            let symbol_graph: tarjanize_schemas::SymbolGraph =
                serde_json::from_str(&json).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;

            let options = tarjanize_cost::CostOptions { fit_libs_only };
            let result = tarjanize_cost::fit(&symbol_graph, options);
            result.write_report(w)?;

            // Optionally build and write the cost model JSON.
            if let Some(model_path) = output_model {
                if let Some(model) = tarjanize_cost::build_cost_model(&result) {
                    let file = std::fs::File::create(&model_path)?;
                    let writer = std::io::BufWriter::new(file);
                    serde_json::to_writer_pretty(writer, &model)
                        .map_err(std::io::Error::other)?;
                } else {
                    eprintln!(
                        "WARNING: No cost model could be fitted \
                         (insufficient profiling data)"
                    );
                }
            }
            Ok(())
        }),
    }
}
