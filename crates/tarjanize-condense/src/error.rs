//! Error types for the tarjanize-condense crate.

use std::backtrace::Backtrace;
use std::fmt;

/// Error type for graph condensation operations.
///
/// This error captures failures that can occur during SCC computation
/// and graph condensation. Uses the canonical struct pattern with
/// backtrace capture and `is_xxx()` helper methods.
#[derive(Debug)]
pub struct CondenseError {
    kind: CondenseErrorKind,
    backtrace: Backtrace,
}

/// Internal error variants. Not exposed publicly; use `is_xxx()` methods.
#[derive(Debug)]
pub(crate) enum CondenseErrorKind {
    /// Failed to deserialize input JSON.
    Deserialization(serde_json::Error),
    /// Failed to serialize output to JSON.
    Serialization(serde_json::Error),
    /// I/O error when reading input or writing output.
    Io(std::io::Error),
}

impl CondenseError {
    /// Creates an error from an error kind, capturing a backtrace.
    pub(crate) fn new(kind: CondenseErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns true if this error is due to deserialization failure.
    pub fn is_deserialization(&self) -> bool {
        matches!(self.kind, CondenseErrorKind::Deserialization(_))
    }

    /// Returns true if this error is due to serialization failure.
    pub fn is_serialization(&self) -> bool {
        matches!(self.kind, CondenseErrorKind::Serialization(_))
    }

    /// Returns true if this error is due to I/O failure.
    pub fn is_io(&self) -> bool {
        matches!(self.kind, CondenseErrorKind::Io(_))
    }

    /// Returns the backtrace captured when this error was created.
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl fmt::Display for CondenseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CondenseErrorKind::Deserialization(err) => {
                write!(f, "failed to deserialize input: {err}")
            }
            CondenseErrorKind::Serialization(err) => {
                write!(f, "failed to serialize output: {err}")
            }
            CondenseErrorKind::Io(err) => {
                write!(f, "I/O error: {err}")
            }
        }
    }
}

impl fmt::Display for CondenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Summary of what happened.
        writeln!(f, "{}", self.kind)?;

        // Backtrace (will be empty unless RUST_BACKTRACE is set).
        write!(f, "{}", self.backtrace)
    }
}

impl std::error::Error for CondenseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            CondenseErrorKind::Deserialization(err)
            | CondenseErrorKind::Serialization(err) => Some(err),
            CondenseErrorKind::Io(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for CondenseError {
    fn from(err: std::io::Error) -> Self {
        Self {
            kind: CondenseErrorKind::Io(err),
            backtrace: Backtrace::capture(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn test_deserialization() {
        let json_err =
            serde_json::from_str::<String>("not valid json").unwrap_err();
        let err =
            CondenseError::new(CondenseErrorKind::Deserialization(json_err));

        assert!(err.is_deserialization());
        assert!(!err.is_serialization());
        assert!(!err.is_io());

        assert!(err.to_string().contains("failed to deserialize input"));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_serialization() {
        // Create an error by trying to serialize something that fails.
        // We'll just reuse a parse error since the error type is the same.
        let json_err =
            serde_json::from_str::<String>("not valid json").unwrap_err();
        let err =
            CondenseError::new(CondenseErrorKind::Serialization(json_err));

        assert!(err.is_serialization());
        assert!(!err.is_deserialization());
        assert!(!err.is_io());

        assert!(err.to_string().contains("failed to serialize output"));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_io_from() {
        let io_err =
            std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = CondenseError::from(io_err);

        assert!(err.is_io());
        assert!(!err.is_deserialization());
        assert!(!err.is_serialization());

        assert!(err.to_string().contains("I/O error"));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_backtrace_captured() {
        let json_err =
            serde_json::from_str::<String>("not valid json").unwrap_err();
        let err =
            CondenseError::new(CondenseErrorKind::Deserialization(json_err));
        // Just verify we can call backtrace() - the actual content depends
        // on RUST_BACKTRACE environment variable.
        let _ = err.backtrace();
    }

    #[test]
    fn test_debug_impl() {
        let json_err =
            serde_json::from_str::<String>("not valid json").unwrap_err();
        let err =
            CondenseError::new(CondenseErrorKind::Deserialization(json_err));
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("CondenseError"));
    }
}
