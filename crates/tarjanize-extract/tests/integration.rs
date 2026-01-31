//! Integration tests for tarjanize-extract.
//!
//! These tests use real Cargo workspace fixtures to test the public API.

use std::path::PathBuf;

use tarjanize_extract::{SymbolGraph, run};

/// `run()` should read a real Cargo project and produce reasonable output.
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

/// Dev-dependency cycles are handled correctly.
///
/// When `crate_a` has a dev-dependency on `crate_b`, and `crate_b` has a normal
/// dependency on `crate_a`, this creates a cycle. rust-analyzer skips the
/// cyclic edge (`crate_a` -> `crate_b`) when building the crate graph.
///
/// This test verifies that:
/// 1. `crate_b`'s symbols depend on `crate_a` (normal dep edge exists)
/// 2. `crate_a`'s symbols do NOT depend on `crate_b` (cyclic edge was skipped)
#[test]
fn test_dev_dep_cycle_edge_skipped() {
    let fixture = PathBuf::from("tests/fixtures/dev_dep_cycle");
    let mut output = Vec::new();

    run(&fixture, &mut output).expect("run() should succeed");

    let graph: SymbolGraph = serde_json::from_slice(&output)
        .expect("run() should output valid JSON");

    // Both crates should be extracted.
    assert!(
        graph.crates.contains_key("crate_a"),
        "crate_a should be in the graph"
    );
    assert!(
        graph.crates.contains_key("crate_b"),
        "crate_b should be in the graph"
    );

    // crate_b::function_in_b should depend on crate_a::function_in_a
    // (normal dependency edge exists).
    let crate_b = graph.crates.get("crate_b").unwrap();
    let function_in_b = crate_b
        .symbols
        .get("function_in_b")
        .expect("function_in_b should exist");
    assert!(
        function_in_b
            .dependencies
            .iter()
            .any(|dep| dep.contains("crate_a")),
        "function_in_b should depend on crate_a, but found: {:?}",
        function_in_b.dependencies
    );

    // crate_a::function_in_a should NOT depend on crate_b
    // (cyclic dev-dependency edge was skipped).
    let crate_a = graph.crates.get("crate_a").unwrap();
    let function_in_a = crate_a
        .symbols
        .get("function_in_a")
        .expect("function_in_a should exist");
    assert!(
        !function_in_a
            .dependencies
            .iter()
            .any(|dep| dep.contains("crate_b")),
        "function_in_a should NOT depend on crate_b (cyclic edge should be skipped), but found: {:?}",
        function_in_a.dependencies
    );
}
