//! Error types for the tarjanize-extract crate.

use std::backtrace::Backtrace;
use std::fmt;

use ra_ap_base_db::FileId;

/// Error type for symbol graph extraction operations.
///
/// This error captures failures that can occur during workspace loading,
/// file path resolution, and output generation.
#[derive(Debug)]
pub struct ExtractError {
    kind: ExtractErrorKind,
    backtrace: Backtrace,
}

/// Internal error variants. Not exposed publicly; use `is_xxx()` methods instead.
#[derive(Debug)]
enum ExtractErrorKind {
    /// A file ID could not be resolved to a path in the source roots.
    FilePathNotFound(FileId),
    /// A crate's root file has no parent directory.
    CrateRootNoParent(String),
    /// Failed to load the workspace into rust-analyzer.
    WorkspaceLoad(Box<dyn std::error::Error + Send + Sync>),
    /// Failed to serialize output to JSON.
    Serialization(serde_json::Error),
    /// I/O error when writing output.
    Io(std::io::Error),
}

impl ExtractError {
    /// Creates an error for when a file path cannot be resolved.
    pub fn file_path_not_found(file_id: FileId) -> Self {
        Self {
            kind: ExtractErrorKind::FilePathNotFound(file_id),
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates an error for when a crate's root file has no parent directory.
    pub fn crate_root_no_parent(crate_name: impl Into<String>) -> Self {
        Self {
            kind: ExtractErrorKind::CrateRootNoParent(crate_name.into()),
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates an error for workspace loading failures.
    pub fn workspace_load(
        err: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            kind: ExtractErrorKind::WorkspaceLoad(err.into()),
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates an error for serialization failures.
    pub fn serialization(err: serde_json::Error) -> Self {
        Self {
            kind: ExtractErrorKind::Serialization(err),
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates an error for I/O failures.
    pub fn io(err: std::io::Error) -> Self {
        Self {
            kind: ExtractErrorKind::Io(err),
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns true if this error is due to a file path not being found.
    pub fn is_file_path_not_found(&self) -> bool {
        matches!(self.kind, ExtractErrorKind::FilePathNotFound(_))
    }

    /// Returns true if this error is due to a crate root having no parent.
    pub fn is_crate_root_no_parent(&self) -> bool {
        matches!(self.kind, ExtractErrorKind::CrateRootNoParent(_))
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

impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExtractErrorKind::FilePathNotFound(file_id) => {
                write!(f, "file ID {file_id:?} not found in source roots")
            }
            ExtractErrorKind::CrateRootNoParent(crate_name) => {
                write!(
                    f,
                    "crate '{crate_name}' root file has no parent directory"
                )
            }
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

impl std::error::Error for ExtractError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ExtractErrorKind::FilePathNotFound(_)
            | ExtractErrorKind::CrateRootNoParent(_) => None,
            ExtractErrorKind::WorkspaceLoad(err) => Some(err.as_ref()),
            ExtractErrorKind::Serialization(err) => Some(err),
            ExtractErrorKind::Io(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for ExtractError {
    fn from(err: std::io::Error) -> Self {
        Self::io(err)
    }
}

impl From<serde_json::Error> for ExtractError {
    fn from(err: serde_json::Error) -> Self {
        Self::serialization(err)
    }
}
