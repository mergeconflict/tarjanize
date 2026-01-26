# ra_ap_test_fixture

High-level utility fixture methods for writing tests. Provides a way to define multi-crate Rust projects as strings, avoiding the need for a real sysroot in tests.

## Overview

Fixtures are strings containing Rust source code with optional metadata that describe a project setup. This is the primary way to write tests for rust-analyzer without depending on the entire sysroot.

The crate provides:
1. **`WithFixture` trait** - Methods for setting up databases from fixture strings
2. **`ChangeFixture`** - Parsed fixture ready to apply to a database
3. **Test proc macros** - Predefined proc macros for testing

---

## Fixture Syntax

### Basic Structure

A fixture without metadata is parsed into a single source file (`/main.rs`). Metadata is added after a `//-` comment prefix.

```text
//- /main.rs
fn main() {
    println!("Hello");
}
```

### File Metadata

Each file can have the following metadata after `//-`:

| Metadata | Example | Description |
|----------|---------|-------------|
| **Path** (required) | `/main.rs`, `/foo/bar.rs` | Must start with `/` |
| `crate:<name>` | `crate:my_lib` | Defines a new crate with this file as root |
| `crate:<name>@<ver>,<url>` | `crate:foo@0.1.0,https://...` | With version and repo URL |
| `deps:<crate1>,<crate2>` | `deps:serde,log` | Dependencies (requires `crate:`) |
| `extern-prelude:<names>` | `extern-prelude:std,core` | Limits extern prelude |
| `edition:<year>` | `edition:2021` | Rust edition (2015, 2018, 2021, 2024) |
| `cfg:<options>` | `cfg:test,feature="foo"` | Configuration options |
| `env:<KEY>=<value>` | `env:CARGO_MANIFEST_DIR=/` | Environment variables |
| `crate-attr:<attr>` | `crate-attr:no_std` | Crate-level attributes |
| `new_source_root:<kind>` | `new_source_root:library` | Starts new source root |
| `library` | `library` | Marks crate as external library |

### Global Meta (at top, in order)

| Meta | Example | Description |
|------|---------|-------------|
| `//- toolchain:` | `//- toolchain: nightly` | Sets Rust toolchain (default: stable) |
| `//- target_data_layout:` | `//- target_data_layout: e-m:e...` | LLVM data layout string |
| `//- target_arch:` | `//- target_arch: wasm32` | Target architecture |
| `//- proc_macros:` | `//- proc_macros: identity,mirror` | Enables test proc macros |
| `//- minicore:` | `//- minicore: option, iterator` | Includes subset of libcore |

### Cursor Markers

Use `$0` to mark cursor position(s):
- Single `$0`: marks a position (use with `with_position`)
- Two `$0`: marks a range (use with `with_range`)
- Escape as `\$0` for literal `$0`

---

## WithFixture Trait

The main trait for setting up test databases.

```rust
pub trait WithFixture: Default + ExpandDatabase + SourceDatabase + 'static {
    fn with_single_file(ra_fixture: &str) -> (Self, EditionedFileId);
    fn with_many_files(ra_fixture: &str) -> (Self, Vec<EditionedFileId>);
    fn with_files(ra_fixture: &str) -> Self;
    fn with_files_extra_proc_macros(ra_fixture: &str, proc_macros: Vec<(String, ProcMacro)>) -> Self;
    fn with_position(ra_fixture: &str) -> (Self, FilePosition);
    fn with_range(ra_fixture: &str) -> (Self, FileRange);
    fn with_range_or_offset(ra_fixture: &str) -> (Self, EditionedFileId, RangeOrOffset);
    fn test_crate(&self) -> Crate;
}
```

**Automatically implemented** for any `DB: ExpandDatabase + SourceDatabase + Default + 'static`.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `with_single_file` | `(DB, EditionedFileId)` | Single file fixture |
| `with_many_files` | `(DB, Vec<EditionedFileId>)` | Multiple files, no cursor |
| `with_files` | `DB` | Multiple files, no cursor, no files returned |
| `with_files_extra_proc_macros` | `DB` | With additional custom proc macros |
| `with_position` | `(DB, FilePosition)` | Single `$0` cursor position |
| `with_range` | `(DB, FileRange)` | Two `$0` markers for range |
| `with_range_or_offset` | `(DB, EditionedFileId, RangeOrOffset)` | Either position or range |
| `test_crate` | `Crate` | Get the first non-lang crate |

---

## ChangeFixture

Parsed fixture ready to be applied.

```rust
pub struct ChangeFixture {
    pub file_position: Option<(span::EditionedFileId, RangeOrOffset)>,
    pub file_lines: Vec<usize>,       // Line numbers of each file in fixture
    pub files: Vec<span::EditionedFileId>,
    pub change: ChangeWithProcMacros,
    pub sysroot_files: Vec<FileId>,   // Core library files if minicore used
}
```

### Methods

| Method | Description |
|--------|-------------|
| `parse(ra_fixture)` | Parse fixture string |
| `parse_with_proc_macros(ra_fixture, minicore_raw, proc_macros)` | Parse with custom proc macros |

---

## Available Test Proc Macros

When using `//- proc_macros: <names>`, these are available:

| Name | Kind | Description |
|------|------|-------------|
| `identity` | Attribute | Returns input unchanged |
| `DeriveIdentity` | Derive | Returns input unchanged |
| `input_replace` | Attribute | Returns attribute input, ignores item |
| `mirror` | Bang | Reverses token stream |
| `shorten` | Bang | Shortens identifiers/literals |
| `issue_18089` | Attribute | Generates macro_rules! + pub use |
| `issue_18840` | Attribute | Tests fixup span handling |
| `issue_17479` | Bang | Reads literal as ident |
| `issue_18898` | Attribute | Generates large output |
| `disallow_cfg` | Attribute | Errors on cfg/cfg_attr tokens |
| `generate_suffixed_type` | Attribute | Generates `<Name>Suffix` struct |

### Custom Proc Macros

Use `with_files_extra_proc_macros` to add custom proc macros:

```rust
let proc_macros = vec![
    ("source code".into(), ProcMacro {
        name: Symbol::intern("my_macro"),
        kind: ProcMacroKind::Attr,
        expander: Arc::new(MyExpander),
        disabled: false,
    }),
];
let db = RootDatabase::with_files_extra_proc_macros(fixture, proc_macros);
```

---

## Constants

```rust
pub const WORKSPACE: SourceRootId = SourceRootId(0);
```

---

## Examples

### Single file with cursor position

```rust
let (db, position) = RootDatabase::with_position(r#"
fn main() {
    let x$0 = 42;
}
"#);
```

### Multiple crates with dependencies

```rust
let db = RootDatabase::with_files(r#"
//- /main.rs crate:main deps:helper
use helper::greet;
fn main() { greet(); }

//- /lib.rs crate:helper
pub fn greet() {}
"#);
```

### Using minicore for lang items

```rust
let db = RootDatabase::with_files(r#"
//- minicore: option, result, iterator
//- /main.rs
fn foo() -> Option<i32> { Some(42) }
"#);
```

Available minicore flags are listed in `test-utils/src/minicore.rs`.

### Using test proc macros

```rust
let db = RootDatabase::with_files(r#"
//- proc_macros: identity, mirror
//- /main.rs crate:main deps:proc_macros
use proc_macros::identity;

#[identity]
fn foo() {}
"#);
```

### External library crate

```rust
let db = RootDatabase::with_files(r#"
//- /main.rs crate:main deps:ext_lib
use ext_lib::util;

//- /ext.rs crate:ext_lib library
pub fn util() {}
"#);
```

### With cfg options

```rust
let db = RootDatabase::with_files(r#"
//- /lib.rs crate:test cfg:test,feature="serde"
#[cfg(feature = "serde")]
use serde::Serialize;

#[cfg(test)]
mod tests {}
"#);
```

---

## Minicore

The `minicore` feature provides a minimal subset of the standard library for testing without requiring a full sysroot. When `//- minicore: <flags>` is specified:

1. A synthetic `core` crate is created
2. Only requested features are included
3. All other crates get `core` as a dependency

This enables testing of code that uses lang items (like `Option`, `Result`, iterators) without needing the actual standard library.

---

## Usage in tarjanize

tarjanize-extract uses ra_ap_test_fixture for unit tests:

```rust
use hir::RootDatabase;
use test_fixture::WithFixture;

#[test]
fn test_extract_function() {
    let db = RootDatabase::with_files(r#"
//- /lib.rs crate:test_crate
pub fn my_function() -> i32 { 42 }
"#);

    // Use db for testing extraction logic
}
```

The fixture-based approach provides:
- Fast, in-memory testing
- No filesystem dependencies
- Precise control over project structure
- Easy multi-crate scenarios
