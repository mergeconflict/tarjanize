//! Schema definitions for tarjanize output formats.
//!
//! This module contains the data structures that define tarjanize's output.
//! These types are serialized to JSON and represent the symbol graph that
//! downstream phases (SCC computation, partitioning, etc.) will consume.
//!
//! The schemas are designed to be:
//! - **Self-describing**: JSON Schema is auto-generated from Rust types
//! - **Stable**: Changes require updating the golden schema file
//! - **Complete**: All information needed for analysis is captured

pub mod symbol_graph;

#[doc(inline)]
pub use symbol_graph::*;
