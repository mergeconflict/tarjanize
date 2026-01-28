//! Integration tests for tarjanize-extract.
//!
//! These tests use real Cargo workspace fixtures to test the public API.

use std::path::PathBuf;

use tarjanize_extract::{SymbolGraph, run};

/// run() should read a real Cargo project and produce reasonable output.
#[test]
fn test_run() {
    let fixture = PathBuf::from("tests/fixtures/minimal");
    let mut output = Vec::new();

    run(&fixture, &mut output).expect("run() should succeed");

    let graph: SymbolGraph = serde_json::from_slice(&output)
        .expect("run() should output valid JSON");
    assert!(
        graph.crates.contains_key("minimal_crate"),
        "JSON output should contain minimal_crate"
    );
}

/// Virtual workspaces with nested crate directories are handled correctly.
///
/// Tests that we can extract from a workspace with:
/// - No root package (pure virtual workspace)
/// - Crates nested in a `crates/` subdirectory
#[test]
fn test_virtual_workspace_nested_crates() {
    let fixture = PathBuf::from("tests/fixtures/virtual_workspace");
    let mut output = Vec::new();

    run(&fixture, &mut output).expect("run() should succeed");

    let graph: SymbolGraph = serde_json::from_slice(&output)
        .expect("run() should output valid JSON");

    // The nested member_a crate should be found
    assert!(
        graph.crates.contains_key("member_a"),
        "JSON output should contain member_a from nested crates/ directory"
    );
}

/// Derive macros in `#[derive(...)]` should be captured as dependencies.
///
/// When a struct uses `#[derive(MyDerive)]`, we want to capture the
/// dependency on `MyDerive`. This test uses a real workspace with a
/// workspace-local proc-macro crate.
///
/// Currently ignored: `sema.resolve_derive_macro()` returns None for
/// unknown reasons. See dependencies.rs for details.
#[test]
#[ignore = "derive macro resolution not yet working"]
fn test_derive_macro_captured() {
    let fixture = PathBuf::from("tests/fixtures/derive_macro");
    let mut output = Vec::new();

    run(&fixture, &mut output).expect("run() should succeed");

    let graph: SymbolGraph = serde_json::from_slice(&output)
        .expect("run() should output valid JSON");

    // The consumer crate should exist with Foo
    let consumer = graph
        .crates
        .get("consumer")
        .expect("consumer crate should exist");
    let foo = consumer
        .symbols
        .get("Foo")
        .expect("Foo symbol should exist");

    // Foo uses #[derive(MyDerive)] and should depend on it.
    assert!(
        foo.dependencies.iter().any(|dep| dep.contains("MyDerive")),
        "Foo should depend on MyDerive, but found: {:?}",
        foo.dependencies
    );
}
