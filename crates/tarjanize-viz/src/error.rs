//! Error types for the visualization pipeline.
//!
//! Covers deserialization of the input `SymbolGraph` and I/O errors from
//! reading input or serving HTTP responses.

use std::{fmt, io};

/// Errors that can occur during visualization generation.
///
/// Why: centralizes error reporting for the viz CLI and server startup.
#[derive(Debug)]
pub struct VizError {
    kind: VizErrorKind,
}

/// The specific category of visualization error.
///
/// Why: keeps error classification internal while exposing helpers.
#[derive(Debug)]
enum VizErrorKind {
    /// Failed to deserialize the input `SymbolGraph` JSON.
    Deserialize(serde_json::Error),
    /// I/O error reading input or writing output.
    Io(io::Error),
}

impl fmt::Display for VizError {
    /// Formats the error for user-facing messages.
    ///
    /// Why: CLI users should see concise, actionable error text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            VizErrorKind::Deserialize(e) => {
                write!(f, "failed to deserialize input: {e}")
            }
            VizErrorKind::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for VizError {
    /// Returns the underlying source error for chaining.
    ///
    /// Why: preserves detailed errors for logs and debugging.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            VizErrorKind::Deserialize(e) => Some(e),
            VizErrorKind::Io(e) => Some(e),
        }
    }
}

impl VizError {
    /// Creates a deserialization error.
    ///
    /// Why: input parsing failures should be classified explicitly.
    pub(crate) fn deserialize(err: serde_json::Error) -> Self {
        Self {
            kind: VizErrorKind::Deserialize(err),
        }
    }

    /// Creates an I/O error.
    ///
    /// Why: consolidates I/O failures under one error type.
    pub(crate) fn io(err: io::Error) -> Self {
        Self {
            kind: VizErrorKind::Io(err),
        }
    }
}

impl From<io::Error> for VizError {
    /// Converts an I/O error into a visualization error.
    ///
    /// Why: simplifies error propagation in async setup code.
    fn from(err: io::Error) -> Self {
        Self::io(err)
    }
}
