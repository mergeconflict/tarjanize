//! cargo-tarjanize: Extract symbol graphs from Rust workspaces using rustc.
//!
//! This binary operates in two modes:
//!
//! 1. **Orchestrator mode** (default): When invoked by the user via `cargo tarjanize`,
//!    runs `cargo check --all-targets` with `RUSTC_WRAPPER` set to this binary,
//!    then aggregates the per-crate extraction results.
//!
//! 2. **Driver mode**: When invoked by cargo as `RUSTC_WRAPPER`, acts as a custom
//!    rustc driver. For workspace crates, extracts symbol information and writes
//!    it to a temp file. For external crates, delegates to the real rustc.
//!
//! The mode is detected by checking if the first argument is "rustc" (driver mode)
//! or not (orchestrator mode).

#![feature(rustc_private)]
#![feature(box_patterns)]

// These extern crate declarations pull in the rustc internal crates.
// They must be declared here because rustc_private crates use a special
// linking mechanism that requires explicit extern crate statements.
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

mod driver;
mod extract;
mod orchestrator;
mod profile;

use std::env;
use std::process::ExitCode;

use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use itertools::Itertools;
use tracing_subscriber::EnvFilter;

/// Environment variable for passing verbosity level from orchestrator to driver.
pub const ENV_VERBOSITY: &str = "TARJANIZE_VERBOSITY";

/// Crates to include in the logging allowlist.
const CRATES: &[&str] = &["cargo_tarjanize"];

/// Extract symbol graphs from Rust workspaces using rustc.
///
/// When invoked as `cargo tarjanize`, analyzes all workspace member crates
/// and produces a JSON file containing all symbols and their dependency
/// relationships. Compilation costs are measured using rustc's `-Zself-profile`.
#[derive(Debug, Parser)]
#[command(author, version, about)]
#[command(bin_name = "cargo tarjanize")]
pub struct Cli {
    /// Verbosity level (-v for debug, -vv for trace, -q for warn, -qq for error)
    #[command(flatten)]
    pub verbose: Verbosity<InfoLevel>,

    /// Path to Cargo.toml manifest
    #[arg(long, value_name = "PATH")]
    pub manifest_path: Option<String>,

    /// Package(s) to analyze (can be specified multiple times).
    /// If not specified, analyzes all workspace members.
    #[arg(short, long, value_name = "SPEC")]
    pub package: Vec<String>,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    // Detect mode based on first argument.
    // When cargo invokes us as RUSTC_WRAPPER, it passes the path to rustc as argv[1].
    // The path might be absolute (e.g., /usr/bin/rustc) or just "rustc".
    // When the user runs "cargo tarjanize", cargo invokes us with "tarjanize" as argv[1].
    let is_driver_mode = args.get(1).is_some_and(|arg| {
        // Check if this looks like a rustc invocation.
        // Could be "rustc", "/path/to/rustc", or "rustc.exe" on Windows.
        let file_name = std::path::Path::new(arg)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        file_name == "rustc" || file_name.starts_with("rustc.")
    });

    if is_driver_mode {
        // Driver mode: act as a rustc wrapper.
        // Initialize tracing from environment variable set by orchestrator.
        init_tracing_from_env();

        // args[0] = this binary
        // args[1] = path to rustc
        // args[2..] = actual rustc arguments
        driver::run(&args[1..])
    } else {
        // Orchestrator mode: run cargo and aggregate results.
        // Handle both "cargo-tarjanize" direct invocation and "cargo tarjanize" subcommand.
        // In subcommand form, args[1] will be "tarjanize", which we skip for clap.
        let clap_args = if args.get(1).is_some_and(|a| a == "tarjanize") {
            // Skip "tarjanize" so clap sees: ["cargo-tarjanize", ...rest]
            std::iter::once(args[0].clone())
                .chain(args[2..].iter().cloned())
                .collect::<Vec<_>>()
        } else {
            args.clone()
        };

        let cli = Cli::parse_from(&clap_args);

        // Initialize tracing with the user's verbosity level.
        init_tracing(cli.verbose.tracing_level_filter());

        orchestrator::run(&cli)
    }
}

/// Initialize tracing with a specific level filter.
fn init_tracing(level: tracing::level_filters::LevelFilter) {
    let allowlist = CRATES.iter().map(|c| format!("{c}={level}")).join(",");
    let filter = EnvFilter::new(format!("warn,{allowlist}"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

/// Initialize tracing from the `TARJANIZE_VERBOSITY` environment variable.
/// Used in driver mode where we can't parse CLI args.
fn init_tracing_from_env() {
    let level = env::var(ENV_VERBOSITY)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(tracing::level_filters::LevelFilter::INFO);

    let allowlist = CRATES.iter().map(|c| format!("{c}={level}")).join(",");
    let filter = EnvFilter::new(format!("warn,{allowlist}"));

    // Use try_init because the subscriber might already be set if this is
    // called multiple times in the same process.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}
