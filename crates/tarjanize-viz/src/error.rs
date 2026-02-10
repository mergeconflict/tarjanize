//! Error types for the visualization pipeline.
//!
//! Covers deserialization of the input `SymbolGraph` and `CostModel`,
//! template rendering via askama, and I/O when writing the output HTML.

use std::{fmt, io};

/// Errors that can occur during visualization generation.
#[derive(Debug)]
pub struct VizError {
    kind: VizErrorKind,
}

/// The specific category of visualization error.
#[derive(Debug)]
enum VizErrorKind {
    /// Failed to deserialize the input `SymbolGraph` or `CostModel` JSON.
    Deserialize(serde_json::Error),
    /// Failed to serialize the schedule data to JSON for embedding.
    Serialize(serde_json::Error),
    /// Failed to render the askama HTML template.
    Template(askama::Error),
    /// I/O error reading input or writing output.
    Io(io::Error),
}

impl fmt::Display for VizError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            VizErrorKind::Deserialize(e) => {
                write!(f, "failed to deserialize input: {e}")
            }
            VizErrorKind::Serialize(e) => {
                write!(f, "failed to serialize schedule data: {e}")
            }
            VizErrorKind::Template(e) => {
                write!(f, "failed to render HTML template: {e}")
            }
            VizErrorKind::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for VizError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            VizErrorKind::Deserialize(e) | VizErrorKind::Serialize(e) => {
                Some(e)
            }
            VizErrorKind::Template(e) => Some(e),
            VizErrorKind::Io(e) => Some(e),
        }
    }
}

impl VizError {
    /// Creates a deserialization error.
    pub(crate) fn deserialize(err: serde_json::Error) -> Self {
        Self {
            kind: VizErrorKind::Deserialize(err),
        }
    }

    /// Creates a serialization error.
    pub(crate) fn serialize(err: serde_json::Error) -> Self {
        Self {
            kind: VizErrorKind::Serialize(err),
        }
    }

    /// Creates a template rendering error.
    pub(crate) fn template(err: askama::Error) -> Self {
        Self {
            kind: VizErrorKind::Template(err),
        }
    }

    /// Creates an I/O error.
    pub(crate) fn io(err: io::Error) -> Self {
        Self {
            kind: VizErrorKind::Io(err),
        }
    }
}

impl From<io::Error> for VizError {
    fn from(err: io::Error) -> Self {
        Self::io(err)
    }
}
