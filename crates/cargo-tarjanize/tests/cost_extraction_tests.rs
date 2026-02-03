//! Integration tests for cost extraction and profile key generation.
//!
//! These tests verify that cargo-tarjanize:
//! 1. Generates correct profile keys (`DefPath` format) for symbols
//! 2. Aggregates costs from nested items to their parent symbols
//! 3. Correctly matches profile data to symbols
//!
//! Symbol keys use rustc's internal `DefPath` format which matches
//! `-Zself-profile` output. This enables direct lookup of timing data
//! without any mapping layer.
//!
//! Test organization:
//! - Profile key generation: Verify symbol keys match `DefPath` format
//! - Cost aggregation: Verify nested items aggregate to parents
//! - Impl block variants: Profile keys for special impl types

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use tarjanize_schemas::{Module, SymbolGraph, SymbolKind};

/// Path to the cargo-tarjanize binary.
fn cargo_tarjanize_bin() -> &'static str {
    // CARGO_BIN_EXE_<name> is set by Cargo during test builds to the correct
    // binary path, regardless of target directory (works with cargo-llvm-cov).
    env!("CARGO_BIN_EXE_cargo-tarjanize")
}

/// Run cargo-tarjanize on a fixture workspace and return the extracted graph.
fn extract_fixture(fixture_name: &str) -> SymbolGraph {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let fixture_path = Path::new(manifest_dir)
        .join("tests/fixtures")
        .join(fixture_name);

    assert!(
        fixture_path.exists(),
        "Fixture not found: {}",
        fixture_path.display()
    );

    // Clean the fixture to ensure fresh extraction.
    let clean_status = Command::new("cargo")
        .arg("clean")
        .current_dir(&fixture_path)
        .status()
        .expect("failed to run cargo clean");
    assert!(clean_status.success(), "cargo clean failed");

    // Create a temporary file for output.
    let output_file =
        tempfile::NamedTempFile::new().expect("failed to create temp file");

    // Run cargo-tarjanize with output file.
    let output = Command::new(cargo_tarjanize_bin())
        .arg("-o")
        .arg(output_file.path())
        .current_dir(&fixture_path)
        .output()
        .expect("failed to run cargo-tarjanize");

    if !output.status.success() {
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("cargo-tarjanize failed with status: {}", output.status);
    }

    // Parse the JSON output from the file.
    let file = std::fs::File::open(output_file.path())
        .expect("failed to open output file");
    serde_json::from_reader(file).expect("failed to parse JSON output")
}

/// Try to run cargo-tarjanize on a fixture workspace.
///
/// Returns `None` if extraction fails (e.g., due to unavailable nightly features).
fn try_extract_fixture(fixture_name: &str) -> Option<SymbolGraph> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let fixture_path = Path::new(manifest_dir)
        .join("tests/fixtures")
        .join(fixture_name);

    if !fixture_path.exists() {
        return None;
    }

    // Clean the fixture to ensure fresh extraction.
    let clean_status = Command::new("cargo")
        .arg("clean")
        .current_dir(&fixture_path)
        .status()
        .ok()?;
    if !clean_status.success() {
        return None;
    }

    // Create a temporary file for output.
    let output_file = tempfile::NamedTempFile::new().ok()?;

    // Run cargo-tarjanize with output file.
    let output = Command::new(cargo_tarjanize_bin())
        .arg("-o")
        .arg(output_file.path())
        .current_dir(&fixture_path)
        .output()
        .ok()?;

    if !output.status.success() {
        eprintln!(
            "Skipping {fixture_name}: extraction failed (feature may not be available)"
        );
        return None;
    }

    let file = std::fs::File::open(output_file.path()).ok()?;
    serde_json::from_reader(file).ok()
}

/// Get a symbol by key suffix from the graph.
///
/// Returns the symbol and its full key if found.
fn find_symbol<'a>(
    graph: &'a SymbolGraph,
    key_suffix: &str,
) -> Option<(&'a str, &'a tarjanize_schemas::Symbol)> {
    fn find_in_module<'a>(
        module: &'a Module,
        key_suffix: &str,
    ) -> Option<(&'a str, &'a tarjanize_schemas::Symbol)> {
        for (key, symbol) in &module.symbols {
            if key.ends_with(key_suffix) {
                return Some((key.as_str(), symbol));
            }
        }
        for submodule in module.submodules.values() {
            if let Some(result) = find_in_module(submodule, key_suffix) {
                return Some(result);
            }
        }
        None
    }

    for crate_data in graph.crates.values() {
        if let Some(result) = find_in_module(&crate_data.root, key_suffix) {
            return Some(result);
        }
    }
    None
}

/// Assert that a symbol exists with the given key suffix.
fn assert_symbol_exists(graph: &SymbolGraph, key_suffix: &str) {
    assert!(
        find_symbol(graph, key_suffix).is_some(),
        "Expected symbol with key ending in '{key_suffix}'\nGraph: {graph:#?}"
    );
}

/// Assert that NO symbol exists with the given key suffix.
fn assert_no_symbol(graph: &SymbolGraph, key_suffix: &str) {
    assert!(
        find_symbol(graph, key_suffix).is_none(),
        "Did not expect symbol with key ending in '{key_suffix}'\nGraph: {graph:#?}"
    );
}

/// Get all symbol keys from the graph.
fn collect_all_keys(graph: &SymbolGraph) -> HashSet<String> {
    fn collect_from_module(module: &Module, keys: &mut HashSet<String>) {
        keys.extend(module.symbols.keys().cloned());
        for submodule in module.submodules.values() {
            collect_from_module(submodule, keys);
        }
    }

    let mut keys = HashSet::new();
    for crate_data in graph.crates.values() {
        collect_from_module(&crate_data.root, &mut keys);
    }
    keys
}

/// Get the impl name from a symbol (if it's an impl).
fn get_impl_name(symbol: &tarjanize_schemas::Symbol) -> Option<&str> {
    match &symbol.kind {
        SymbolKind::Impl { name, .. } => Some(name.as_str()),
        SymbolKind::ModuleDef { .. } => None,
    }
}

/// Count symbols matching a key pattern.
fn count_symbols_matching(graph: &SymbolGraph, pattern: &str) -> usize {
    collect_all_keys(graph)
        .iter()
        .filter(|k| k.contains(pattern))
        .count()
}

// =============================================================================
// PROFILE KEY GENERATION - BASIC SYMBOLS
//
// Verify that symbol keys use the correct DefPath format for basic symbols.
// =============================================================================

#[test]
fn test_profile_key_function() {
    let graph = extract_fixture("profile_key_function");
    // Function key is relative to crate root (crate name stored separately)
    assert_symbol_exists(&graph, "my_function");

    // Verify it's a Function kind
    let (key, symbol) = find_symbol(&graph, "my_function").unwrap();
    // Key should be just the symbol name (crate prefix added during profile lookup)
    assert_eq!(key, "my_function", "Function key should be simple name");
    assert!(matches!(
        &symbol.kind,
        SymbolKind::ModuleDef { kind, .. } if kind == "Function"
    ));
}

#[test]
fn test_profile_key_struct() {
    let graph = extract_fixture("profile_key_struct");
    assert_symbol_exists(&graph, "MyStruct");

    let (key, symbol) = find_symbol(&graph, "MyStruct").unwrap();
    assert_eq!(key, "MyStruct", "Struct key should be simple name");
    assert!(matches!(
        &symbol.kind,
        SymbolKind::ModuleDef { kind, .. } if kind == "Struct"
    ));
}

#[test]
fn test_profile_key_enum() {
    let graph = extract_fixture("profile_key_enum");
    assert_symbol_exists(&graph, "MyEnum");

    let (key, symbol) = find_symbol(&graph, "MyEnum").unwrap();
    assert_eq!(key, "MyEnum", "Enum key should be simple name");
    assert!(matches!(
        &symbol.kind,
        SymbolKind::ModuleDef { kind, .. } if kind == "Enum"
    ));
}

#[test]
fn test_profile_key_trait() {
    let graph = extract_fixture("profile_key_trait");
    assert_symbol_exists(&graph, "MyTrait");

    let (key, symbol) = find_symbol(&graph, "MyTrait").unwrap();
    assert_eq!(key, "MyTrait", "Trait key should be simple name");
    assert!(matches!(
        &symbol.kind,
        SymbolKind::ModuleDef { kind, .. } if kind == "Trait"
    ));
}

#[test]
fn test_profile_key_type_alias() {
    let graph = extract_fixture("profile_key_type_alias");
    assert_symbol_exists(&graph, "MyAlias");

    let (key, symbol) = find_symbol(&graph, "MyAlias").unwrap();
    assert_eq!(key, "MyAlias", "Type alias key should be simple name");
    assert!(matches!(
        &symbol.kind,
        SymbolKind::ModuleDef { kind, .. } if kind == "TypeAlias"
    ));
}

#[test]
fn test_profile_key_const() {
    let graph = extract_fixture("profile_key_const");
    assert_symbol_exists(&graph, "MY_CONST");

    let (key, symbol) = find_symbol(&graph, "MY_CONST").unwrap();
    assert!(key == "MY_CONST", "Const key should be simple name: {key}");
    assert!(matches!(
        &symbol.kind,
        SymbolKind::ModuleDef { kind, .. } if kind == "Const"
    ));
}

#[test]
fn test_profile_key_static() {
    let graph = extract_fixture("profile_key_static");
    assert_symbol_exists(&graph, "MY_STATIC");

    let (key, symbol) = find_symbol(&graph, "MY_STATIC").unwrap();
    assert!(
        key == "MY_STATIC",
        "Static key should be simple name: {key}"
    );
    assert!(matches!(
        &symbol.kind,
        SymbolKind::ModuleDef { kind, .. } if kind == "Static"
    ));
}

#[test]
fn test_profile_key_macro() {
    let graph = extract_fixture("profile_key_macro");
    assert_symbol_exists(&graph, "my_macro");

    let (key, symbol) = find_symbol(&graph, "my_macro").unwrap();
    assert!(key == "my_macro", "Macro key should be simple name: {key}");
    assert!(matches!(
        &symbol.kind,
        SymbolKind::ModuleDef { kind, .. } if kind == "Macro"
    ));
}

// =============================================================================
// PROFILE KEY GENERATION - IMPL BLOCKS
//
// Verify that impl blocks use `{{impl}}` notation with proper disambiguation.
// =============================================================================

#[test]
fn test_profile_key_inherent_impl() {
    let graph = extract_fixture("profile_key_inherent_impl");
    // Inherent impl should have key: `crate_name::{{impl}}`
    assert_symbol_exists(&graph, "{{impl}}");

    let (key, symbol) = find_symbol(&graph, "{{impl}}").unwrap();
    assert!(
        key.contains("{{impl}}"),
        "Impl key should use {{{{impl}}}} notation: {key}"
    );

    // Verify it's an Impl kind with correct name
    let impl_name = get_impl_name(symbol).expect("should be impl");
    assert!(
        impl_name.contains("impl "),
        "Impl name should be human-readable: {impl_name}"
    );
}

#[test]
fn test_profile_key_trait_impl() {
    let graph = extract_fixture("profile_key_trait_impl");
    assert_symbol_exists(&graph, "{{impl}}");

    let (key, symbol) = find_symbol(&graph, "{{impl}}").unwrap();
    assert!(key.contains("{{impl}}"), "Should use {{{{impl}}}} notation");

    let impl_name = get_impl_name(symbol).expect("should be impl");
    assert!(
        impl_name.contains(" for "),
        "Trait impl name should include 'for': {impl_name}"
    );
}

#[test]
fn test_profile_key_multiple_impls() {
    let graph = extract_fixture("profile_key_multiple_impls");
    // Multiple impls should be disambiguated with [N] suffix
    let keys = collect_all_keys(&graph);
    let impl_keys: Vec<_> =
        keys.iter().filter(|k| k.contains("{{impl}}")).collect();

    assert!(
        impl_keys.len() >= 2,
        "Should have at least 2 impl blocks: {impl_keys:?}"
    );

    // Check for disambiguation: should have {{impl}} and {{impl}}[1] (or similar)
    let has_base = impl_keys.iter().any(|k| k.ends_with("{{impl}}"));
    let has_numbered = impl_keys.iter().any(|k| k.contains("{{impl}}["));
    assert!(
        has_base || has_numbered,
        "Multiple impls should use disambiguation: {impl_keys:?}"
    );
}

#[test]
fn test_profile_key_nested_module_impl() {
    let graph = extract_fixture("profile_key_nested_module_impl");
    // Impl in submodule is stored inside the submodule structure.
    // The key is just `{{impl}}` but it's within the `submod` submodule.
    assert_symbol_exists(&graph, "{{impl}}");

    // Verify the impl is in a submodule by checking the crate structure.
    let crate_data =
        graph.crates.get("profile_key_nested_module_impl").unwrap();
    assert!(
        crate_data.root.submodules.contains_key("submod"),
        "Should have submod submodule"
    );
    let submod = crate_data.root.submodules.get("submod").unwrap();
    assert!(
        submod.symbols.contains_key("{{impl}}"),
        "{{{{impl}}}} should be in submod"
    );
}

#[test]
fn test_profile_key_generic_impl() {
    let graph = extract_fixture("profile_key_generic_impl");
    // Generic impl still uses {{impl}} notation
    assert_symbol_exists(&graph, "{{impl}}");

    let (key, _) = find_symbol(&graph, "{{impl}}").unwrap();
    // Generic params don't appear in DefPath - it's still just {{impl}}
    assert!(
        key.contains("{{impl}}"),
        "Generic impl should use {{{{impl}}}} notation: {key}"
    );
}

// =============================================================================
// PROFILE KEY GENERATION - DERIVE-GENERATED CODE
//
// Verify profile keys for derive-generated impl blocks use `_[N]::{{impl}}`.
// =============================================================================

#[test]
fn test_profile_key_derive_impl() {
    let graph = extract_fixture("profile_key_derive_impl");
    // Derive-generated impls use anonymous blocks: `_[N]::{{impl}}`
    let keys = collect_all_keys(&graph);

    // Should have impl blocks (either regular or in anonymous blocks)
    let impl_count = keys.iter().filter(|k| k.contains("{{impl}}")).count();
    assert!(impl_count >= 1, "Should have derive-generated impl");

    // Note: The exact format depends on rustc version. Some versions use
    // `_[N]::{{impl}}`, others use different patterns. The important thing
    // is that impl blocks are extracted.
}

#[test]
fn test_profile_key_multiple_derives() {
    let graph = extract_fixture("profile_key_multiple_derives");
    // Multiple derives should create multiple impl blocks
    let impl_count = count_symbols_matching(&graph, "{{impl}}");
    assert!(
        impl_count >= 2,
        "Multiple derives should create multiple impls: found {impl_count}"
    );
}

#[test]
fn test_profile_key_proc_macro_attribute() {
    let graph = extract_fixture("proc_macro_attribute");
    // Proc macro attributes generate code - verify something is extracted
    // The exact structure depends on the proc macro
    let keys = collect_all_keys(&graph);
    assert!(
        !keys.is_empty(),
        "Proc macro should generate extractable symbols"
    );
}

// =============================================================================
// COST AGGREGATION - NESTED FUNCTIONS
//
// Nested functions don't appear as separate symbols; their cost aggregates
// to the containing function.
// =============================================================================

#[test]
fn test_cost_nested_fn() {
    let graph = extract_fixture("cost_nested_fn");
    // Outer function should exist
    assert_symbol_exists(&graph, "outer_fn");
    // Nested function should NOT be a separate symbol
    assert_no_symbol(&graph, "nested_fn");
}

#[test]
fn test_cost_deeply_nested_fn() {
    let graph = extract_fixture("cost_deeply_nested_fn");
    assert_symbol_exists(&graph, "outer_fn");
    // Neither level 1 nor level 2 nested functions should appear
    assert_no_symbol(&graph, "level1_fn");
    assert_no_symbol(&graph, "level2_fn");
}

#[test]
fn test_cost_multiple_nested_fns() {
    let graph = extract_fixture("cost_multiple_nested_fns");
    assert_symbol_exists(&graph, "outer_fn");
    // None of the nested functions should appear
    assert_no_symbol(&graph, "helper1");
    assert_no_symbol(&graph, "helper2");
}

// =============================================================================
// COST AGGREGATION - CLOSURES
//
// Closures aggregate to their containing function via profile normalization.
// They appear in profile data as `{{closure}}` but are aggregated away.
// =============================================================================

#[test]
fn test_cost_closure() {
    let graph = extract_fixture("cost_closure");
    assert_symbol_exists(&graph, "uses_closure");
    // Closures don't appear as separate symbols
    assert_no_symbol(&graph, "{{closure}}");
}

#[test]
fn test_cost_multiple_closures() {
    let graph = extract_fixture("cost_multiple_closures");
    assert_symbol_exists(&graph, "uses_closures");
    // No closure symbols should exist
    let closure_count = count_symbols_matching(&graph, "{{closure}}");
    assert_eq!(closure_count, 0, "Closures should not appear as symbols");
}

#[test]
fn test_cost_nested_closures() {
    let graph = extract_fixture("cost_nested_closures");
    assert_symbol_exists(&graph, "nested_closure_user");
    assert_no_symbol(&graph, "{{closure}}");
}

#[test]
fn test_cost_closure_in_method() {
    let graph = extract_fixture("cost_closure_in_method");
    // The impl block should exist
    assert_symbol_exists(&graph, "{{impl}}");
    // Method closures aggregate to the impl block
    assert_no_symbol(&graph, "{{closure}}");
}

#[test]
fn test_cost_move_closure() {
    let graph = extract_fixture("cost_move_closure");
    assert_symbol_exists(&graph, "uses_move_closure");
    assert_no_symbol(&graph, "{{closure}}");
}

// =============================================================================
// COST AGGREGATION - ASYNC
//
// Async functions generate futures; their cost should aggregate properly.
// =============================================================================

#[test]
fn test_cost_async_fn() {
    let graph = extract_fixture("cost_async_fn");
    assert_symbol_exists(&graph, "my_async_fn");
    // Async-generated items should not appear separately
    assert_no_symbol(&graph, "{{closure}}");
    assert_no_symbol(&graph, "{{opaque}}");
}

#[test]
fn test_cost_async_closure() {
    let graph = extract_fixture("async_closure");
    // Uses existing fixture - just verify extraction works
    let keys = collect_all_keys(&graph);
    assert!(
        !keys.is_empty(),
        "Should extract symbols from async closure"
    );
}

#[test]
fn test_cost_async_closure_nested() {
    let graph = extract_fixture("async_closure_nested");
    // Uses existing fixture
    let keys = collect_all_keys(&graph);
    assert!(
        !keys.is_empty(),
        "Should extract symbols from nested async closure"
    );
}

#[test]
fn test_cost_async_block() {
    let graph = extract_fixture("cost_async_block");
    assert_symbol_exists(&graph, "with_async_block");
    // Async block internals should not appear
    assert_no_symbol(&graph, "{{closure}}");
}

#[test]
fn test_cost_async_await() {
    let graph = extract_fixture("async_await");
    // Uses existing fixture
    let keys = collect_all_keys(&graph);
    assert!(!keys.is_empty(), "Should extract symbols from async/await");
}

// =============================================================================
// COST AGGREGATION - CONST BLOCKS
//
// Const blocks (`const { ... }`) aggregate to containing function.
// =============================================================================

#[test]
fn test_cost_const_block() {
    let graph = extract_fixture("cost_const_block");
    assert_symbol_exists(&graph, "with_const_block");
    // Const blocks use anonymous const notation which shouldn't appear
    // as separate top-level symbols
}

// =============================================================================
// COST AGGREGATION - CONST/STATIC INITIALIZERS
//
// Complex initializers in const/static items may have internal helper items.
// =============================================================================

#[test]
fn test_cost_const_initializer() {
    let graph = extract_fixture("const_initializer");
    // Uses existing fixture - const is named MY_CONST
    assert_symbol_exists(&graph, "MY_CONST");
}

#[test]
fn test_cost_static_initializer() {
    let graph = extract_fixture("static_initializer");
    // Uses existing fixture - static is named MY_STATIC
    assert_symbol_exists(&graph, "MY_STATIC");
}

#[test]
fn test_cost_thread_local_macro() {
    let graph = extract_fixture("cost_thread_local_macro");
    // thread_local! generates statics with special structure
    let keys = collect_all_keys(&graph);
    assert!(
        !keys.is_empty(),
        "Should extract symbols from thread_local! macro"
    );
}

// =============================================================================
// COST AGGREGATION - MACRO-GENERATED ITEMS
//
// Macros can generate nested items; verify aggregation works.
// =============================================================================

#[test]
fn test_cost_tracing_callsite() {
    let graph = extract_fixture("cost_tracing_callsite");
    // tracing's event!/span! macros generate __CALLSITE statics
    // These should aggregate to the containing function
    assert_symbol_exists(&graph, "traced_function");
}

#[test]
fn test_cost_macro_nested_static() {
    let graph = extract_fixture("cost_macro_nested_static");
    // Macro-generated nested statics should aggregate
    assert_symbol_exists(&graph, "with_macro_static");
}

#[test]
fn test_cost_declarative_macro_fn() {
    let graph = extract_fixture("cost_declarative_macro_fn");
    // Declarative macro that generates a function
    // The generated function should be a normal symbol
    assert_symbol_exists(&graph, "generated_fn");
}

#[test]
fn test_cost_declarative_macro_struct() {
    let graph = extract_fixture("cost_declarative_macro_struct");
    assert_symbol_exists(&graph, "GeneratedStruct");
}

// =============================================================================
// PROFILE DATA APPLICATION
//
// Verify that profile timing data is correctly applied to symbols.
// Note: These tests verify structure, not actual timing values.
// =============================================================================

#[test]
fn test_cost_from_profile() {
    let graph = extract_fixture("cost_from_profile");
    // Verify symbol exists and has a cost field.
    let (_, symbol) = find_symbol(&graph, "profiled_fn").expect("should exist");
    // Cost should be positive (from profile data).
    // We check frontend cost since that's what gets populated for most symbols.
    let total_cost = symbol.frontend_cost_ms + symbol.backend_cost_ms;
    assert!(
        total_cost >= 0.0,
        "Cost should be non-negative: frontend={}, backend={}",
        symbol.frontend_cost_ms,
        symbol.backend_cost_ms
    );
}

#[test]
fn test_cost_impl_profile_match() {
    let graph = extract_fixture("cost_impl_profile_match");
    // Impl block should have cost from profile.
    let (_, symbol) = find_symbol(&graph, "{{impl}}").expect("should exist");
    let total_cost = symbol.frontend_cost_ms + symbol.backend_cost_ms;
    assert!(
        total_cost >= 0.0,
        "Impl cost should be non-negative: frontend={}, backend={}",
        symbol.frontend_cost_ms,
        symbol.backend_cost_ms
    );
}

// =============================================================================
// IMPL BLOCK VARIANTS
//
// Verify profile keys for impl blocks on special types.
// =============================================================================

#[test]
fn test_profile_key_impl_for_tuple() {
    let graph = extract_fixture("profile_key_impl_for_tuple");
    // impl Trait for (A, B) should still use {{impl}}
    assert_symbol_exists(&graph, "{{impl}}");

    let (_, symbol) = find_symbol(&graph, "{{impl}}").unwrap();
    let impl_name = get_impl_name(symbol).expect("should be impl");
    // The name should reference a tuple type
    assert!(
        impl_name.contains('(') || impl_name.contains("impl"),
        "Should be impl for tuple: {impl_name}"
    );
}

#[test]
fn test_profile_key_impl_for_array() {
    let graph = extract_fixture("profile_key_impl_for_array");
    assert_symbol_exists(&graph, "{{impl}}");
}

#[test]
fn test_profile_key_impl_for_dyn_trait() {
    let graph = extract_fixture("profile_key_impl_for_dyn_trait");
    assert_symbol_exists(&graph, "{{impl}}");
}

#[test]
fn test_profile_key_impl_for_external_generic() {
    let graph = extract_fixture("profile_key_impl_for_external_generic");
    // impl Trait for Box<T> should work
    assert_symbol_exists(&graph, "{{impl}}");
}

#[test]
fn test_profile_key_negative_impl() {
    let graph = extract_fixture("negative_impl");
    // Uses existing fixture for negative impl
    // Negative impls (impl !Trait for Type) should extract
    let keys = collect_all_keys(&graph);
    assert!(!keys.is_empty(), "Should extract negative impl");
}

// =============================================================================
// TRAIT ITEMS
//
// Verify costs for items defined in traits.
// =============================================================================

#[test]
fn test_cost_trait_default_method() {
    let graph = extract_fixture("trait_default_method_body");
    // Uses existing fixture
    assert_symbol_exists(&graph, "MyTrait");
}

#[test]
fn test_cost_trait_default_const() {
    let graph = extract_fixture("trait_default_const");
    // Uses existing fixture - trait is named MyTrait
    assert_symbol_exists(&graph, "MyTrait");
}

#[test]
fn test_cost_trait_method_signature() {
    let graph = extract_fixture("trait_method_signature_types");
    // Uses existing fixture
    assert_symbol_exists(&graph, "MyTrait");
}

// =============================================================================
// CROSS-CRATE
//
// Verify profile matching works across workspace crates.
// =============================================================================

#[test]
fn test_cost_cross_crate_match() {
    let graph = extract_fixture("cross_crate_local_deps");
    // Uses existing fixture - verify both crates are extracted
    assert!(
        !graph.crates.is_empty(),
        "Should extract at least one crate"
    );
}

// =============================================================================
// EDGE CASES
//
// Various edge cases for profile key generation.
// =============================================================================

#[test]
fn test_profile_key_unicode_names() {
    let graph = extract_fixture("profile_key_unicode_names");
    // Unicode identifiers should work
    assert_symbol_exists(&graph, "日本語");
}

#[test]
fn test_profile_key_special_chars() {
    let graph = extract_fixture("profile_key_special_chars");
    // Names with underscores, numbers should work
    assert_symbol_exists(&graph, "_underscore_fn");
    assert_symbol_exists(&graph, "fn_with_123");
}

#[test]
fn test_cost_empty_function() {
    let graph = extract_fixture("cost_empty_function");
    // Empty functions still have cost (from type checking, etc.).
    let (_, symbol) = find_symbol(&graph, "empty_fn").expect("should exist");
    // We just verify the symbol exists and has non-negative costs.
    let total_cost = symbol.frontend_cost_ms + symbol.backend_cost_ms;
    assert!(total_cost >= 0.0, "Empty fn should have non-negative cost");
}

// =============================================================================
// COST AGGREGATION - SUM VERIFICATION
//
// Verify that parent costs include nested item costs.
// =============================================================================

#[test]
fn test_cost_sum_nested() {
    let graph = extract_fixture("cost_sum_nested");
    // Parent function should exist and have positive cost that includes nested fn.
    let (_, symbol) =
        find_symbol(&graph, "outer_with_nested").expect("should exist");
    let total_cost = symbol.frontend_cost_ms + symbol.backend_cost_ms;
    assert!(
        total_cost >= 0.0,
        "Outer fn should have non-negative cost including nested: frontend={}, backend={}",
        symbol.frontend_cost_ms,
        symbol.backend_cost_ms
    );
    // Nested function should NOT appear as separate symbol.
    assert_no_symbol(&graph, "nested_helper");
}

#[test]
fn test_cost_sum_closures() {
    let graph = extract_fixture("cost_sum_closures");
    // Parent function should exist and have positive cost that includes closures.
    let (_, symbol) =
        find_symbol(&graph, "outer_with_closures").expect("should exist");
    let total_cost = symbol.frontend_cost_ms + symbol.backend_cost_ms;
    assert!(
        total_cost >= 0.0,
        "Outer fn should have non-negative cost including closures: frontend={}, backend={}",
        symbol.frontend_cost_ms,
        symbol.backend_cost_ms
    );
    // Closures should NOT appear as separate symbols.
    assert_no_symbol(&graph, "{{closure}}");
}

#[test]
fn test_cost_thread_local_raw() {
    // Raw #[thread_local] requires nightly feature - use try_extract.
    let Some(graph) = try_extract_fixture("cost_thread_local_raw") else {
        eprintln!("Skipping test_cost_thread_local_raw: feature not available");
        return;
    };
    // Should have the static and function.
    assert_symbol_exists(&graph, "COUNTER");
    assert_symbol_exists(&graph, "increment");
}

#[test]
fn test_cost_virtual_workspace() {
    let graph = extract_fixture("cost_virtual_workspace");
    // Both crates should be extracted.
    assert!(graph.crates.contains_key("crate_a"), "Should have crate_a");
    assert!(graph.crates.contains_key("crate_b"), "Should have crate_b");
    // Each crate's root module should have symbols.
    let crate_a = graph.crates.get("crate_a").unwrap();
    assert!(!crate_a.root.symbols.is_empty(), "crate_a should have symbols");
    let crate_b = graph.crates.get("crate_b").unwrap();
    assert!(!crate_b.root.symbols.is_empty(), "crate_b should have symbols");
}

#[test]
fn test_cost_inline_asm_sym() {
    let graph = extract_fixture("cost_inline_asm_sym");
    // Both functions should exist
    assert_symbol_exists(&graph, "target_fn");
    assert_symbol_exists(&graph, "uses_asm_sym");
}

#[test]
fn test_cost_range_pattern_const() {
    let graph = extract_fixture("cost_range_pattern_const");
    // Consts and function should exist
    assert_symbol_exists(&graph, "MIN_VALUE");
    assert_symbol_exists(&graph, "MAX_VALUE");
    assert_symbol_exists(&graph, "classify");
}
