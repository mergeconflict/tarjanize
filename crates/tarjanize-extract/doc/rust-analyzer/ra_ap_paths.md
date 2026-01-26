# ra_ap_paths

Thin wrappers around [`camino::Utf8PathBuf`](https://docs.rs/camino/latest/camino/struct.Utf8PathBuf.html), distinguishing between absolute and relative paths at the type level.

## Overview

This crate provides type-safe path handling by encoding whether a path is absolute or relative in the type system. This prevents common bugs where relative paths are accidentally used where absolute paths are expected (or vice versa).

The crate re-exports several types from `camino`:
- `Utf8Path` - A borrowed UTF-8 encoded path
- `Utf8PathBuf` - An owned UTF-8 encoded path
- `Utf8Component` - A component of a UTF-8 path
- `Utf8Components` - An iterator over path components
- `Utf8Prefix` - Windows path prefixes

## Type Hierarchy

```
AbsPathBuf (owned)  ─┬─ Deref ──► AbsPath (borrowed)
                     └─ From  ──► Utf8PathBuf, PathBuf

RelPathBuf (owned)  ─┬─ Deref ──► RelPath (borrowed)
                     └─ From  ──► Utf8PathBuf
```

## AbsPathBuf

An owned `Utf8PathBuf` that is guaranteed to be absolute.

### Construction

```rust
// Panics if path is not absolute
let path = AbsPathBuf::assert(Utf8PathBuf::from("/usr/bin"));

// Panics if path is not absolute or not valid UTF-8
let path = AbsPathBuf::assert_utf8(PathBuf::from("/usr/bin"));

// Fallible conversion - returns Err(path) if not absolute
let path: Result<AbsPathBuf, Utf8PathBuf> = AbsPathBuf::try_from(utf8_path_buf);
let path: Result<AbsPathBuf, Utf8PathBuf> = AbsPathBuf::try_from("/usr/bin");
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `assert` | `fn assert(path: Utf8PathBuf) -> AbsPathBuf` | Wrap an absolute path, panicking if relative |
| `assert_utf8` | `fn assert_utf8(path: PathBuf) -> AbsPathBuf` | Convert from `PathBuf`, panicking if not UTF-8 or not absolute |
| `as_path` | `fn as_path(&self) -> &AbsPath` | Coerce to borrowed `AbsPath` slice |
| `pop` | `fn pop(&mut self) -> bool` | Remove the last component (won't remove root) |
| `push` | `fn push<P: AsRef<Utf8Path>>(&mut self, suffix: P)` | Extend with a path suffix |
| `join` | `fn join(&self, path: impl AsRef<Utf8Path>) -> Self` | Create a new path by joining with another |

### Trait Implementations

- `From<AbsPathBuf> for Utf8PathBuf` - Convert to owned `Utf8PathBuf`
- `From<AbsPathBuf> for PathBuf` - Convert to owned `PathBuf`
- `Deref<Target = AbsPath>` - Coerce to borrowed `AbsPath`
- `AsRef<Utf8Path>` - Reference as `Utf8Path`
- `AsRef<OsStr>` - Reference as `OsStr`
- `AsRef<Path>` - Reference as `Path`
- `AsRef<AbsPath>` - Reference as `AbsPath`
- `Borrow<AbsPath>` - Borrow as `AbsPath`
- `TryFrom<Utf8PathBuf>` - Fallible conversion (error if not absolute)
- `TryFrom<&str>` - Fallible conversion from string slice
- `PartialEq<P: AsRef<Path>>` - Compare with any path type
- `Display` - Format for display
- `Debug`, `Clone`, `Ord`, `PartialOrd`, `Eq`, `Hash`

## AbsPath

A borrowed wrapper around an absolute `Utf8Path`. This is a `#[repr(transparent)]` newtype, allowing zero-cost conversions via pointer casts.

### Construction

```rust
// Panics if path is not absolute
let path: &AbsPath = AbsPath::assert(utf8_path);

// Fallible conversion
let path: Result<&AbsPath, &Utf8Path> = <&AbsPath>::try_from(utf8_path);
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `assert` | `fn assert(path: &Utf8Path) -> &AbsPath` | Wrap path, panicking if not absolute |
| `parent` | `fn parent(&self) -> Option<&AbsPath>` | Get the parent directory |
| `absolutize` | `fn absolutize(&self, path: impl AsRef<Utf8Path>) -> AbsPathBuf` | Join with path and normalize |
| `join` | `fn join(&self, path: impl AsRef<Utf8Path>) -> AbsPathBuf` | Join with another path |
| `normalize` | `fn normalize(&self) -> AbsPathBuf` | Remove `.`, `..`, repeated separators, trailing slashes |
| `to_path_buf` | `fn to_path_buf(&self) -> AbsPathBuf` | Convert to owned `AbsPathBuf` |
| `canonicalize` | `fn canonicalize(&self) -> !` | **Panics always** - intentionally not provided (see below) |
| `strip_prefix` | `fn strip_prefix(&self, base: &AbsPath) -> Option<&RelPath>` | Strip a prefix, returning relative path |
| `starts_with` | `fn starts_with(&self, base: &AbsPath) -> bool` | Check if path starts with another |
| `ends_with` | `fn ends_with(&self, suffix: &RelPath) -> bool` | Check if path ends with a suffix |
| `name_and_extension` | `fn name_and_extension(&self) -> Option<(&str, Option<&str>)>` | Get file stem and extension |
| `file_name` | `fn file_name(&self) -> Option<&str>` | Get the final component |
| `extension` | `fn extension(&self) -> Option<&str>` | Get the file extension |
| `file_stem` | `fn file_stem(&self) -> Option<&str>` | Get the file name without extension |
| `as_os_str` | `fn as_os_str(&self) -> &OsStr` | Get as `OsStr` |
| `as_str` | `fn as_str(&self) -> &str` | Get as string slice |
| `components` | `fn components(&self) -> Utf8Components<'_>` | Iterate over path components |

### Design Note: No `canonicalize`

The `canonicalize` method intentionally panics with a message referencing rust-analyzer issue #14430. The rationale is that canonicalization (which resolves symlinks and produces canonical paths) is "almost always a wrong solution" in the context of an IDE, where:

1. Symlinks should often be preserved
2. The canonical path may not match what the user expects
3. It requires filesystem access, violating the "POD type" design goal

### Design Note: No `Deref<Target = Utf8Path>`

Unlike `AbsPathBuf`, the `AbsPath` type deliberately does **not** implement `Deref<Target = Utf8Path>`. This is intentional because `Utf8Path` exposes convenience IO methods like `exists()` that would bypass any IO abstraction. The crate wants all IO to go through explicit `fs` calls for easier mocking.

The deprecated methods `display()` and `exists()` will panic if called, guiding users to the correct alternatives.

### Trait Implementations

- `ToOwned<Owned = AbsPathBuf>` - Clone to owned type
- `TryFrom<&Utf8Path>` - Fallible conversion (error if not absolute)
- `PartialEq<P: AsRef<Path>>` - Compare with any path type
- `AsRef<Utf8Path>` - Reference as `Utf8Path`
- `AsRef<Path>` - Reference as `Path`
- `AsRef<OsStr>` - Reference as `OsStr`
- `Display` - Format for display
- `Debug`, `Ord`, `PartialOrd`, `Eq`, `Hash`

## RelPathBuf

An owned `Utf8PathBuf` that is guaranteed to be relative.

### Construction

```rust
// Fallible conversion - returns Err(path) if absolute
let path: Result<RelPathBuf, Utf8PathBuf> = RelPathBuf::try_from(utf8_path_buf);
let path: Result<RelPathBuf, Utf8PathBuf> = RelPathBuf::try_from("src/main.rs");
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `as_path` | `fn as_path(&self) -> &RelPath` | Coerce to borrowed `RelPath` slice |

### Trait Implementations

- `From<RelPathBuf> for Utf8PathBuf` - Convert to `Utf8PathBuf`
- `Deref<Target = RelPath>` - Coerce to borrowed `RelPath`
- `AsRef<Utf8Path>` - Reference as `Utf8Path`
- `AsRef<Path>` - Reference as `Path`
- `TryFrom<Utf8PathBuf>` - Fallible conversion (error if absolute)
- `TryFrom<&str>` - Fallible conversion from string slice
- `Debug`, `Clone`, `Ord`, `PartialOrd`, `Eq`, `PartialEq`, `Hash`

## RelPath

A borrowed wrapper around a relative `Utf8Path`. This is a `#[repr(transparent)]` newtype.

### Construction

```rust
// No validation - use when you know the path is relative
let path: &RelPath = RelPath::new_unchecked(utf8_path);
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `new_unchecked` | `fn new_unchecked(path: &Utf8Path) -> &RelPath` | Create without checking if relative |
| `to_path_buf` | `fn to_path_buf(&self) -> RelPathBuf` | Convert to owned `RelPathBuf` |
| `as_utf8_path` | `fn as_utf8_path(&self) -> &Utf8Path` | Get as `Utf8Path` reference |
| `as_str` | `fn as_str(&self) -> &str` | Get as string slice |

### Trait Implementations

- `AsRef<Utf8Path>` - Reference as `Utf8Path`
- `AsRef<Path>` - Reference as `Path`
- `Debug`, `Ord`, `PartialOrd`, `Eq`, `PartialEq`, `Hash`

## Path Normalization

The `AbsPath::normalize()` method performs the following transformations:

| Input | Output |
|-------|--------|
| `/a//b` | `/a/b` (removes repeated separators) |
| `/a/./b` | `/a/b` (removes `.` components) |
| `/a/b/../c` | `/a/c` (resolves `..` components) |
| `/a/b/` | `/a/b` (removes trailing slash) |

**Note**: This is a purely lexical operation - it does not access the filesystem and will not resolve symlinks.

## Usage in tarjanize

The `AbsPathBuf` and `AbsPath` types are used throughout tarjanize-extract to ensure all paths to workspace roots, crate roots, and source files are absolute. This prevents path resolution bugs that could occur if relative paths were mixed with absolute paths.

Key usages:
- Workspace root paths from `ra_ap_project_model`
- Source file paths from rust-analyzer's VFS
- Module file paths during symbol extraction
