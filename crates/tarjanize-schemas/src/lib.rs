//! Schema definitions for tarjanize output formats.
//!
//! This crate contains the data structures that define tarjanize's intermediate
//! and output formats. These types are serialized to JSON and represent the
//! various artifacts produced by tarjanize's pipeline:
//!
//! - **Symbol Graph**: The full dependency graph extracted from rust-analyzer
//! - **Condensed Graph**: The condensation graph after SCC computation (TODO)
//!
//! The schemas are designed to be:
//! - **Self-describing**: JSON Schema is auto-generated from Rust types
//! - **Stable**: Changes require updating the golden schema files
//! - **Complete**: All information needed for analysis is captured
//! - **Shared**: Used across all phases of the tarjanize pipeline

mod symbol_graph;

#[doc(inline)]
pub use symbol_graph::*;
