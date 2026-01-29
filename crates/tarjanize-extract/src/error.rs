//! Error types for the tarjanize-extract crate.

use std::backtrace::Backtrace;
use std::fmt;

/// Error type for symbol graph extraction operations.
///
/// This error captures failures that can occur during workspace loading
/// and output generation. Internal errors (file path resolution, crate
/// metadata issues) are handled with `anyhow` and logged, not propagated.
#[derive(Debug)]
pub struct ExtractError {
    kind: ExtractErrorKind,
    backtrace: Backtrace,
}

/// Internal error variants. Not exposed publicly; use `is_xxx()` methods instead.
#[derive(Debug)]
pub(crate) enum ExtractErrorKind {
    /// Failed to load the workspace into rust-analyzer.
    WorkspaceLoad(Box<dyn std::error::Error + Send + Sync>),
    /// Failed to serialize output to JSON.
    Serialization(serde_json::Error),
    /// I/O error when writing output.
    Io(std::io::Error),
}

impl ExtractError {
    /// Creates an error from an error kind, capturing a backtrace.
    pub(crate) fn new(kind: ExtractErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns true if this error is due to workspace loading failure.
    pub fn is_workspace_load(&self) -> bool {
        matches!(self.kind, ExtractErrorKind::WorkspaceLoad(_))
    }

    /// Returns true if this error is due to serialization failure.
    pub fn is_serialization(&self) -> bool {
        matches!(self.kind, ExtractErrorKind::Serialization(_))
    }

    /// Returns true if this error is due to I/O failure.
    pub fn is_io(&self) -> bool {
        matches!(self.kind, ExtractErrorKind::Io(_))
    }

    /// Returns the backtrace captured when this error was created.
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl fmt::Display for ExtractErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtractErrorKind::WorkspaceLoad(err) => {
                write!(f, "failed to load workspace: {err}")
            }
            ExtractErrorKind::Serialization(err) => {
                write!(f, "failed to serialize output: {err}")
            }
            ExtractErrorKind::Io(err) => {
                write!(f, "I/O error: {err}")
            }
        }
    }
}

impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Summary of what happened.
        writeln!(f, "{}", self.kind)?;

        // Backtrace (will be empty unless RUST_BACKTRACE is set).
        write!(f, "{}", self.backtrace)
    }
}

impl std::error::Error for ExtractError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ExtractErrorKind::WorkspaceLoad(err) => Some(err.as_ref()),
            ExtractErrorKind::Serialization(err) => Some(err),
            ExtractErrorKind::Io(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for ExtractError {
    fn from(err: std::io::Error) -> Self {
        Self {
            kind: ExtractErrorKind::Io(err),
            backtrace: Backtrace::capture(),
        }
    }
}

impl From<serde_json::Error> for ExtractError {
    fn from(err: serde_json::Error) -> Self {
        Self {
            kind: ExtractErrorKind::Serialization(err),
            backtrace: Backtrace::capture(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn test_workspace_load() {
        let err = ExtractError::new(ExtractErrorKind::WorkspaceLoad(
            "failed to find Cargo.toml".into(),
        ));

        assert!(err.is_workspace_load());
        assert!(!err.is_serialization());
        assert!(!err.is_io());

        assert!(err.to_string().contains("failed to load workspace"));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_serialization_from() {
        // Create an invalid JSON to trigger a parse error.
        let json_err =
            serde_json::from_str::<String>("not valid json").unwrap_err();
        let err = ExtractError::from(json_err);

        assert!(err.is_serialization());
        assert!(!err.is_workspace_load());
        assert!(!err.is_io());

        assert!(err.to_string().contains("failed to serialize output"));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_io_from() {
        let io_err =
            std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = ExtractError::from(io_err);

        assert!(err.is_io());
        assert!(!err.is_workspace_load());
        assert!(!err.is_serialization());

        assert!(err.to_string().contains("I/O error"));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_backtrace_captured() {
        let err =
            ExtractError::new(ExtractErrorKind::WorkspaceLoad("test".into()));
        // Just verify we can call backtrace() - the actual content depends
        // on RUST_BACKTRACE environment variable.
        let _ = err.backtrace();
    }

    #[test]
    fn test_debug_impl() {
        let err =
            ExtractError::new(ExtractErrorKind::WorkspaceLoad("test".into()));
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("ExtractError"));
    }
}
