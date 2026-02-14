//! Schema definitions for tarjanize output formats.
//!
//! This crate contains the data structures that define tarjanize's intermediate
//! and output formats. These types are serialized to JSON and represent the
//! symbol graph used throughout the tarjanize pipeline.
//!
//! The schemas are designed to be:
//! - **Self-describing**: JSON Schema is auto-generated from Rust types
//! - **Stable**: Changes require updating the golden schema files
//! - **Complete**: All information needed for analysis is captured
//! - **Shared**: Used across all phases of the tarjanize pipeline
//!
//! Why: keeping schemas in one crate guarantees consistent serialization
//! contracts across extraction, cost, schedule, condense, and viz.

mod cost_model;
mod symbol_graph;
#[cfg(test)]
mod testutil;

#[doc(inline)]
pub use cost_model::*;
#[doc(inline)]
pub use symbol_graph::*;
