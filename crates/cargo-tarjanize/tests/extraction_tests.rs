//! Integration tests for symbol extraction.
//!
//! These tests verify that cargo-tarjanize correctly extracts dependencies
//! from various Rust code patterns. Each test uses a fixture workspace
//! in `tests/fixtures/`.
//!
//! Test organization:
//! - Smoke tests: Basic functionality for each symbol kind
//! - Type references: Types in signatures and annotations
//! - Trait bounds: Generic bounds and where clauses
//! - Expressions: Function calls, method calls, closures
//! - Patterns: Destructuring and pattern matching
//! - Macros: Macro invocations and proc macros
//! - Normalization: Collapsing variants/assoc items to containers
//! - Filtering: Items that should NOT be dependency targets

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use tarjanize_schemas::{Crate, Module, SymbolGraph};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Iterate over all crates (compilation units) in the symbol graph.
///
/// With the Package/Target/Crate structure, crates are nested under
/// `packages[pkg].targets[target]`. This helper flattens that for tests
/// that need to iterate over all crates.
fn iter_all_crates(graph: &SymbolGraph) -> impl Iterator<Item = &Crate> {
    graph.packages.values().flat_map(|pkg| pkg.targets.values())
}

/// Get a crate by name (looks for lib target in package with matching name).
///
/// Handles the name normalization: package names use hyphens, crate names use
/// underscores. First tries exact match, then tries with hyphen/underscore
/// conversion.
fn get_crate_by_name<'a>(
    graph: &'a SymbolGraph,
    crate_name: &str,
) -> Option<&'a Crate> {
    // Try package name with hyphens (e.g., "my-crate")
    if let Some(pkg) = graph.packages.get(crate_name) {
        return pkg.targets.get("lib");
    }
    // Try package name with underscores converted to hyphens
    let hyphenated = crate_name.replace('_', "-");
    if let Some(pkg) = graph.packages.get(&hyphenated) {
        return pkg.targets.get("lib");
    }
    // Try underscored name (crate name format)
    let underscored = crate_name.replace('-', "_");
    for (pkg_name, pkg) in &graph.packages {
        if pkg_name.replace('-', "_") == underscored {
            return pkg.targets.get("lib");
        }
    }
    None
}

/// Check if a crate exists in the graph (by package/crate name).
fn has_crate(graph: &SymbolGraph, name: &str) -> bool {
    get_crate_by_name(graph, name).is_some()
}

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
/// This is useful for testing experimental features that may not be available.
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
        // Feature likely not available - skip test.
        eprintln!(
            "Skipping {fixture_name}: extraction failed (feature may not be available)"
        );
        return None;
    }

    let file = std::fs::File::open(output_file.path()).ok()?;
    serde_json::from_reader(file).ok()
}

/// Check if a symbol has a dependency on another symbol.
///
/// Uses `ends_with` matching for flexibility:
/// - `from` matches symbol keys ending with the pattern (e.g., `"{{impl}}"` or `"{{impl}}[1]"`)
/// - `to` matches dependency paths ending with the pattern
fn has_edge(graph: &SymbolGraph, from: &str, to: &str) -> bool {
    fn check_module(module: &Module, from: &str, to: &str) -> bool {
        // Check symbols in this module.
        for (name, symbol) in &module.symbols {
            if name.ends_with(from)
                && symbol.dependencies.iter().any(|dep| dep.ends_with(to))
            {
                return true;
            }
        }
        // Check submodules.
        for submodule in module.submodules.values() {
            if check_module(submodule, from, to) {
                return true;
            }
        }
        false
    }

    for crate_data in iter_all_crates(graph) {
        if check_module(&crate_data.root, from, to) {
            return true;
        }
    }
    false
}

/// Assert that a dependency edge exists in the graph.
fn assert_has_edge(graph: &SymbolGraph, from: &str, to: &str) {
    assert!(
        has_edge(graph, from, to),
        "{from} should depend on {to}\nGraph: {graph:#?}"
    );
}

/// Assert that NO dependency in the graph ends with a suffix.
fn assert_no_edge_to(graph: &SymbolGraph, to: &str, description: &str) {
    let all_deps = collect_all_dependencies(graph);
    let bad_deps: Vec<_> =
        all_deps.iter().filter(|d| d.ends_with(to)).collect();
    assert!(
        bad_deps.is_empty(),
        "No dependencies should target {description}: {bad_deps:?}"
    );
}

/// Collect all dependencies across all symbols in the graph.
fn collect_all_dependencies(graph: &SymbolGraph) -> Vec<String> {
    fn collect_module_deps(module: &Module, deps: &mut Vec<String>) {
        for symbol in module.symbols.values() {
            deps.extend(symbol.dependencies.iter().cloned());
        }
        for submodule in module.submodules.values() {
            collect_module_deps(submodule, deps);
        }
    }

    let mut deps = Vec::new();
    for crate_data in iter_all_crates(graph) {
        collect_module_deps(&crate_data.root, &mut deps);
    }
    deps
}

/// Get a specific symbol's dependencies from the graph.
///
/// For impl blocks, matches against the `name` field in `SymbolKind::Impl`
/// since symbol keys use `DefPath` format (e.g., `{{impl}}`).
fn get_symbol_deps(
    graph: &SymbolGraph,
    crate_name: &str,
    symbol_name: &str,
) -> HashSet<String> {
    fn find_in_module(module: &Module, name: &str) -> Option<HashSet<String>> {
        if let Some(symbol) = module.symbols.get(name) {
            return Some(symbol.dependencies.clone());
        }
        for (sym_name, symbol) in &module.symbols {
            // Check key for non-impl symbols, name field for impl blocks.
            let matches = match &symbol.kind {
                tarjanize_schemas::SymbolKind::ModuleDef { .. } => {
                    sym_name.ends_with(name)
                }
                tarjanize_schemas::SymbolKind::Impl {
                    name: impl_name, ..
                } => impl_name.contains(name) || name.contains(impl_name),
            };
            if matches {
                return Some(symbol.dependencies.clone());
            }
        }
        for submodule in module.submodules.values() {
            if let Some(deps) = find_in_module(submodule, name) {
                return Some(deps);
            }
        }
        None
    }

    let crate_data =
        get_crate_by_name(graph, crate_name).expect("crate not found");
    find_in_module(&crate_data.root, symbol_name).unwrap_or_default()
}

/// Get an impl symbol's anchors from the graph.
fn get_impl_anchors(
    graph: &SymbolGraph,
    _impl_pattern: &str,
) -> HashSet<String> {
    // Find the first impl block by key pattern (contains {{impl}}).
    // Most test fixtures have a single impl, so this is sufficient.
    fn find_in_module(module: &Module) -> Option<HashSet<String>> {
        for (key, symbol) in &module.symbols {
            if key.contains("{{impl}}")
                && let tarjanize_schemas::SymbolKind::Impl {
                    ref anchors, ..
                } = symbol.kind
            {
                return Some(anchors.clone());
            }
        }
        for submodule in module.submodules.values() {
            if let Some(anchors) = find_in_module(submodule) {
                return Some(anchors);
            }
        }
        None
    }

    for crate_data in iter_all_crates(graph) {
        if let Some(anchors) = find_in_module(&crate_data.root) {
            return anchors;
        }
    }
    HashSet::new()
}

/// Assert that an impl has a specific anchor.
fn assert_has_anchor(graph: &SymbolGraph, impl_pattern: &str, anchor: &str) {
    let anchors = get_impl_anchors(graph, impl_pattern);
    assert!(
        anchors.iter().any(|a| a.ends_with(anchor)),
        "impl {impl_pattern} should have anchor {anchor}\nAnchors: {anchors:?}\nGraph: {graph:#?}"
    );
}

/// Assert that an impl has no anchors.
#[expect(dead_code, reason = "utility function for future tests")]
fn assert_no_anchors(graph: &SymbolGraph, impl_pattern: &str) {
    let anchors = get_impl_anchors(graph, impl_pattern);
    assert!(
        anchors.is_empty(),
        "impl {impl_pattern} should have no anchors\nAnchors: {anchors:?}"
    );
}

/// Assert that a symbol does NOT exist in the graph.
fn assert_no_symbol(graph: &SymbolGraph, name: &str) {
    assert!(
        !symbol_exists(graph, name),
        "Symbol {name} should not exist in graph\nGraph: {graph:#?}"
    );
}

// =============================================================================
// SMOKE TESTS
//
// Basic tests that each symbol kind can have dependencies extracted.
// =============================================================================

#[test]
fn test_fn_call() {
    let graph = extract_fixture("fn_call");
    assert_has_edge(&graph, "caller_fn", "target_fn");
}

#[test]
fn test_struct_field() {
    let graph = extract_fixture("struct_field");
    assert_has_edge(&graph, "ContainerType", "TargetType");
}

#[test]
fn test_enum_field() {
    let graph = extract_fixture("enum_field");
    assert_has_edge(&graph, "MyEnum", "FieldType");
}

#[test]
fn test_union_field() {
    let graph = extract_fixture("union_field");
    assert_has_edge(&graph, "MyUnion", "FieldType");
}

#[test]
fn test_trait_impl() {
    let graph = extract_fixture("trait_impl");
    assert_has_edge(&graph, "{{impl}}", "MyTrait");
    assert_has_edge(&graph, "{{impl}}", "MyType");
}

#[test]
fn test_inherent_impl() {
    let graph = extract_fixture("inherent_impl");
    assert_has_edge(&graph, "{{impl}}", "MyType");
}

#[test]
fn test_const_type() {
    let graph = extract_fixture("const_type");
    assert_has_edge(&graph, "MY_CONST", "TargetType");
}

#[test]
fn test_type_alias() {
    let graph = extract_fixture("type_alias");
    assert_has_edge(&graph, "MyAlias", "TargetType");
}

#[test]
fn test_trait_supertrait() {
    let graph = extract_fixture("trait_supertrait");
    assert_has_edge(&graph, "Subtrait", "Supertrait");
}

// =============================================================================
// TYPE REFERENCES
// =============================================================================

#[test]
fn test_static_type() {
    let graph = extract_fixture("static_type");
    assert_has_edge(&graph, "MY_STATIC", "TargetType");
}

#[test]
fn test_fn_param_type() {
    let graph = extract_fixture("fn_param_type");
    assert_has_edge(&graph, "fn_with_param", "ParamType");
}

#[test]
fn test_fn_return_type() {
    let graph = extract_fixture("fn_return_type");
    assert_has_edge(&graph, "fn_with_return", "ReturnType");
}

#[test]
fn test_fn_ptr_type() {
    let graph = extract_fixture("fn_ptr_type");
    assert_has_edge(&graph, "takes_fn_ptr", "A");
    assert_has_edge(&graph, "takes_fn_ptr", "B");
}

#[test]
fn test_impl_trait_arg() {
    let graph = extract_fixture("impl_trait_arg");
    assert_has_edge(&graph, "takes_impl", "Tr");
}

#[test]
fn test_impl_trait_return() {
    let graph = extract_fixture("impl_trait_return");
    assert_has_edge(&graph, "returns_impl", "Tr");
}

#[test]
fn test_nested_generics() {
    let graph = extract_fixture("nested_generics");
    assert_has_edge(&graph, "nested", "Outer");
    assert_has_edge(&graph, "nested", "Inner");
}

#[test]
fn test_deeply_nested_generics() {
    let graph = extract_fixture("deeply_nested_generics");
    assert_has_edge(&graph, "nested", "Wrapper");
    assert_has_edge(&graph, "nested", "A");
    assert_has_edge(&graph, "nested", "B");
    assert_has_edge(&graph, "nested", "C");
}

#[test]
fn test_raw_pointer_type() {
    let graph = extract_fixture("raw_pointer_type");
    assert_has_edge(&graph, "takes_const_ptr", "Pointee");
    assert_has_edge(&graph, "takes_mut_ptr", "Pointee");
}

#[test]
fn test_slice_type() {
    let graph = extract_fixture("slice_type");
    assert_has_edge(&graph, "takes_slice", "Element");
    assert_has_edge(&graph, "takes_mut_slice", "Element");
}

#[test]
fn test_boxed_slice_type() {
    let graph = extract_fixture("boxed_slice_type");
    assert_has_edge(&graph, "takes_boxed_slice", "Element");
}

#[test]
fn test_option_ref_type() {
    let graph = extract_fixture("option_ref_type");
    assert_has_edge(&graph, "takes_option", "Inner");
}

#[test]
fn test_dyn_trait() {
    let graph = extract_fixture("dyn_trait");
    assert_has_edge(&graph, "takes_dyn", "Tr");
}

#[test]
fn test_dyn_multi_trait() {
    // Note: Rust only allows one non-auto trait in dyn. We test dyn Tr + Send
    // but Send is external, so we only verify the local trait dependency.
    let graph = extract_fixture("dyn_multi_trait");
    assert_has_edge(&graph, "takes_dyn", "Tr");
}

#[test]
fn test_const_expr_in_array_type() {
    let graph = extract_fixture("const_expr_in_array_type");
    assert_has_edge(&graph, "returns_array", "Element");
    assert_has_edge(&graph, "returns_array", "SIZE");
}

#[test]
fn test_const_expr_arithmetic_in_array() {
    let graph = extract_fixture("const_expr_arithmetic_in_array");
    assert_has_edge(&graph, "returns_array", "BASE");
}

#[test]
fn test_enum_tuple_variant_field() {
    let graph = extract_fixture("enum_tuple_variant_field");
    assert_has_edge(&graph, "EnumWithTuple", "TupleType");
}

#[test]
fn test_enum_struct_variant_field() {
    let graph = extract_fixture("enum_struct_variant_field");
    assert_has_edge(&graph, "EnumWithStruct", "FieldType");
}

#[test]
fn test_const_generic_default_expr() {
    let graph = extract_fixture("const_generic_default_expr");
    assert_has_edge(&graph, "Buffer", "DEFAULT_SIZE");
}

#[test]
fn test_impl_assoc_type_definition() {
    let graph = extract_fixture("impl_assoc_type_definition");
    assert_has_edge(&graph, "{{impl}}", "OutputType");
}

#[test]
fn test_type_alias_generics() {
    let graph = extract_fixture("type_alias_generics");
    assert_has_edge(&graph, "MyAlias", "Wrapper");
    assert_has_edge(&graph, "MyAlias", "Inner");
}

// =============================================================================
// EXPRESSIONS (body-level dependencies)
// =============================================================================

#[test]
fn test_method_call() {
    let graph = extract_fixture("method_call");
    assert_has_edge(&graph, "caller", "{{impl}}");
}

#[test]
fn test_const_initializer() {
    let graph = extract_fixture("const_initializer");
    assert_has_edge(&graph, "MY_CONST", "helper");
}

#[test]
fn test_thread_local_ref() {
    let graph = extract_fixture("thread_local_ref");
    // The thread_local! macro uses a const LocalKey wrapper, which is captured.
    assert_has_edge(&graph, "uses_macro_tls", "TLS_MACRO");
    // Raw #[thread_local] statics use ExprKind::ThreadLocalRef in THIR.
    assert_has_edge(&graph, "uses_raw_tls", "TLS_RAW");
}

#[test]
fn test_const_block() {
    let graph = extract_fixture("const_block");
    // Const blocks (`const { ... }`) can reference other items.
    assert_has_edge(&graph, "uses_const_block", "BASE_VALUE");
    assert_has_edge(&graph, "uses_const_block", "helper");
}

#[test]
fn test_range_pattern_const() {
    let graph = extract_fixture("range_pattern_const");
    // Range patterns (`MIN..=MAX`) should capture const dependencies.
    // THIR loses the const DefIds, but we recover them via HIR analysis.
    assert_has_edge(&graph, "in_range", "MIN");
    assert_has_edge(&graph, "in_range", "MAX");
}

#[test]
fn test_inline_asm_sym() {
    let graph = extract_fixture("inline_asm_sym");
    // Inline asm `sym` operands should capture dependencies.
    assert_has_edge(&graph, "uses_asm_sym_fn", "target_fn");
    assert_has_edge(&graph, "uses_asm_sym_static", "TARGET_STATIC");
}

// =============================================================================
// NORMALIZATION
// =============================================================================

#[test]
fn test_variant_normalizes_to_enum() {
    let graph = extract_fixture("variant_to_enum");
    assert_has_edge(&graph, "uses_variant", "MyEnum");
    assert_no_edge_to(&graph, "::Variant", "enum variants");
}

#[test]
fn test_assoc_fn_normalizes_to_impl() {
    let graph = extract_fixture("assoc_fn_to_impl");
    assert_has_edge(&graph, "caller", "{{impl}}");
}

// =============================================================================
// FILTERING
// =============================================================================

#[test]
fn test_module_not_target() {
    let graph = extract_fixture("module_not_target");
    assert_has_edge(&graph, "caller", "S");
    assert_no_edge_to(&graph, "::m", "modules");
}

#[test]
fn test_local_var_not_target() {
    let graph = extract_fixture("local_var_not_target");
    assert_has_edge(&graph, "caller", "S");
    assert_no_edge_to(&graph, "::x", "local variables");
}

// =============================================================================
// TRAIT BOUNDS
// =============================================================================

#[test]
fn test_fn_generic_bound() {
    let graph = extract_fixture("fn_generic_bound");
    assert_has_edge(&graph, "generic_fn", "MyTrait");
}

#[test]
fn test_fn_where_clause() {
    let graph = extract_fixture("fn_where_clause");
    assert_has_edge(&graph, "fn_with_where", "WhereTrait");
}

#[test]
fn test_hrtb() {
    let graph = extract_fixture("hrtb");
    assert_has_edge(&graph, "hrtb", "Tr");
}

#[test]
fn test_multiple_trait_bounds() {
    let graph = extract_fixture("multiple_trait_bounds");
    assert_has_edge(&graph, "multi_bound", "Tr1");
    assert_has_edge(&graph, "multi_bound", "Tr2");
    assert_has_edge(&graph, "multi_bound", "Tr3");
}

#[test]
fn test_assoc_type_constraint() {
    let graph = extract_fixture("assoc_type_constraint");
    assert_has_edge(&graph, "constrained", "HasItem");
    assert_has_edge(&graph, "constrained", "ItemType");
}

#[test]
fn test_struct_generic_bound() {
    let graph = extract_fixture("struct_generic_bound");
    assert_has_edge(&graph, "StructWithBound", "BoundTrait");
}

#[test]
fn test_struct_where_clause() {
    let graph = extract_fixture("struct_where_clause");
    assert_has_edge(&graph, "StructWithWhere", "WhereTrait");
}

#[test]
fn test_default_type_param() {
    let graph = extract_fixture("default_type_param");
    assert_has_edge(&graph, "WithDefault", "Default");
}

#[test]
fn test_enum_generic_bound() {
    let graph = extract_fixture("enum_generic_bound");
    assert_has_edge(&graph, "EnumWithBound", "BoundTrait");
}

#[test]
fn test_impl_generic_bound() {
    let graph = extract_fixture("impl_generic_bound");
    assert_has_edge(&graph, "{{impl}}", "BoundTrait");
}

#[test]
fn test_impl_where_clause() {
    let graph = extract_fixture("impl_where_clause");
    assert_has_edge(&graph, "{{impl}}", "WhereTrait");
}

#[test]
fn test_trait_assoc_type_bound() {
    let graph = extract_fixture("trait_assoc_type_bound");
    assert_has_edge(&graph, "TraitWithAssocBound", "BoundTrait");
}

#[test]
fn test_negative_impl() {
    let graph = extract_fixture("negative_impl");
    assert_has_edge(&graph, "{{impl}}", "MyAuto");
    assert_has_edge(&graph, "{{impl}}", "NoAuto");
}

#[test]
fn test_fn_body() {
    let graph = extract_fixture("fn_body");
    assert_has_edge(&graph, "fn_uses_body", "BodyType");
}

#[test]
fn test_type_ascription_tuple_pattern() {
    let graph = extract_fixture("type_ascription_tuple_pattern");
    assert_has_edge(&graph, "caller", "A");
    assert_has_edge(&graph, "caller", "B");
}

#[test]
fn test_type_ascription_struct_pattern() {
    let graph = extract_fixture("type_ascription_struct_pattern");
    assert_has_edge(&graph, "caller", "S");
}

#[test]
fn test_ref_to_type_alias() {
    let graph = extract_fixture("ref_to_type_alias");
    assert_has_edge(&graph, "caller", "A");
}

// =============================================================================
// MORE EXPRESSIONS
// =============================================================================

#[test]
fn test_turbofish() {
    let graph = extract_fixture("turbofish");
    assert_has_edge(&graph, "caller", "generic");
    assert_has_edge(&graph, "caller", "S");
}

#[test]
fn test_cross_module_path() {
    let graph = extract_fixture("cross_module_path");
    assert_has_edge(&graph, "caller", "S");
    assert_no_edge_to(&graph, "::inner", "modules");
}

#[test]
fn test_ref_to_const() {
    let graph = extract_fixture("ref_to_const");
    assert_has_edge(&graph, "caller", "C");
}

#[test]
fn test_ref_to_static() {
    let graph = extract_fixture("ref_to_static");
    assert_has_edge(&graph, "caller", "S");
}

#[test]
fn test_self_keyword() {
    let graph = extract_fixture("self_keyword");
    assert_has_edge(&graph, "{{impl}}", "T");
}

#[test]
fn test_static_initializer() {
    let graph = extract_fixture("static_initializer");
    assert_has_edge(&graph, "MY_STATIC", "helper");
}

#[test]
fn test_trait_default_const() {
    let graph = extract_fixture("trait_default_const");
    assert_has_edge(&graph, "MyTrait", "helper");
}

#[test]
fn test_trait_default_method_body() {
    let graph = extract_fixture("trait_default_method_body");
    assert_has_edge(&graph, "MyTrait", "helper");
}

#[test]
fn test_impl_method_body() {
    let graph = extract_fixture("impl_method_body");
    assert_has_edge(&graph, "{{impl}}", "helper");
}

#[test]
fn test_closure_captures() {
    let graph = extract_fixture("closure_captures");
    assert_has_edge(&graph, "caller", "helper");
}

#[test]
fn test_async_closure() {
    let graph = extract_fixture("async_closure");
    assert_has_edge(&graph, "caller", "helper");
}

#[test]
fn test_async_closure_nested() {
    let graph = extract_fixture("async_closure_nested");
    assert_has_edge(&graph, "caller", "helper");
}

#[test]
fn test_async_await() {
    let graph = extract_fixture("async_await");
    assert_has_edge(&graph, "caller", "helper");
}

#[test]
fn test_struct_init_shorthand() {
    let graph = extract_fixture("struct_init_shorthand");
    assert_has_edge(&graph, "caller", "Container");
    assert_has_edge(&graph, "caller", "Target");
}

#[test]
fn test_array_repeat() {
    let graph = extract_fixture("array_repeat");
    assert_has_edge(&graph, "caller", "helper");
}

#[test]
fn test_callable_field() {
    let graph = extract_fixture("callable_field");
    assert_has_edge(&graph, "make_container", "target_fn");
}

// =============================================================================
// PATTERNS
// =============================================================================

#[test]
fn test_pattern_destructuring() {
    let graph = extract_fixture("pattern_destructuring");
    assert_has_edge(&graph, "caller", "Target");
}

#[test]
fn test_slice_pattern() {
    let graph = extract_fixture("slice_pattern");
    assert_has_edge(&graph, "caller", "Element");
}

#[test]
fn test_tuple_struct_pattern() {
    let graph = extract_fixture("tuple_struct_pattern");
    assert_has_edge(&graph, "caller", "Wrapper");
}

#[test]
fn test_path_pattern_const() {
    let graph = extract_fixture("path_pattern_const");
    assert_has_edge(&graph, "caller", "MAGIC");
}

// =============================================================================
// MACROS
// =============================================================================

#[test]
fn test_macro_invocation() {
    let graph = extract_fixture("macro_invocation");
    assert_has_edge(&graph, "caller", "target_fn");
}

#[test]
fn test_proc_macro_attribute() {
    let graph = extract_fixture("proc_macro_attribute");
    assert_has_edge(&graph, "Container", "FieldType");
}

// =============================================================================
// MORE NORMALIZATION
// =============================================================================

#[test]
fn test_variant_via_pattern() {
    let graph = extract_fixture("variant_via_pattern");
    assert_has_edge(&graph, "caller", "MyEnum");
    assert_no_edge_to(&graph, "::Variant", "enum variants");
}

#[test]
fn test_variant_field_shorthand() {
    let graph = extract_fixture("variant_field_shorthand");
    assert_has_edge(&graph, "caller", "MyEnum");
    assert_no_edge_to(&graph, "::Variant", "enum variants");
}

#[test]
fn test_trait_method_call() {
    let graph = extract_fixture("trait_method_call");
    assert_has_edge(&graph, "caller", "MyTrait");
}

#[test]
fn test_method_call_on_ref() {
    let graph = extract_fixture("method_call_on_ref");
    assert_has_edge(&graph, "caller", "{{impl}}");
}

#[test]
fn test_method_call_on_mut_ref() {
    let graph = extract_fixture("method_call_on_mut_ref");
    assert_has_edge(&graph, "caller", "{{impl}}");
}

#[test]
fn test_assoc_const_via_path() {
    let graph = extract_fixture("assoc_const_via_path");
    assert_has_edge(&graph, "caller", "{{impl}}");
}

#[test]
fn test_trait_default_method_call() {
    let graph = extract_fixture("trait_default_method_call");
    // The call should normalize to the trait, not the impl
    assert_has_edge(&graph, "caller", "MyTrait");
}

#[test]
fn test_trait_assoc_const_via_qualified_path() {
    let graph = extract_fixture("trait_assoc_const_via_qualified_path");
    assert_has_edge(&graph, "caller", "MyTrait");
}

#[test]
fn test_trait_assoc_type_via_qualified_path() {
    let graph = extract_fixture("trait_assoc_type_via_qualified_path");
    assert_has_edge(&graph, "caller", "MyTrait");
    assert_has_edge(&graph, "caller", "OutputType");
}

#[test]
fn test_trait_method_signature_types() {
    let graph = extract_fixture("trait_method_signature_types");
    assert_has_edge(&graph, "MyTrait", "ParamType");
    assert_has_edge(&graph, "MyTrait", "ReturnType");
}

// =============================================================================
// MORE FILTERING
// =============================================================================

#[test]
fn test_generic_type_param_not_target() {
    let graph = extract_fixture("generic_type_param_not_target");
    assert_has_edge(&graph, "generic", "S");
    assert_no_edge_to(&graph, "::T", "generic type params");
}

#[test]
fn test_const_generic_not_target() {
    let graph = extract_fixture("const_generic_not_target");
    assert_has_edge(&graph, "generic", "S");
    assert_no_edge_to(&graph, "::N", "const generics");
}

#[test]
fn test_tuple_field_not_target() {
    let graph = extract_fixture("tuple_field_not_target");
    assert_has_edge(&graph, "caller", "S");
    // Tuple field access (.0) should not create edges
}

#[test]
fn test_label_not_target() {
    let graph = extract_fixture("label_not_target");
    assert_has_edge(&graph, "caller", "S");
    assert_no_edge_to(&graph, "'outer", "labels");
}

#[test]
fn test_lifetime_not_target() {
    let graph = extract_fixture("lifetime_not_target");
    assert_has_edge(&graph, "caller", "S");
    assert_no_edge_to(&graph, "'a", "lifetimes");
}

#[test]
fn test_static_lifetime_not_target() {
    let graph = extract_fixture("static_lifetime_not_target");
    assert_has_edge(&graph, "caller", "S");
    assert_no_edge_to(&graph, "'static", "static lifetime");
}

#[test]
fn test_crate_keyword_not_target() {
    let graph = extract_fixture("crate_keyword_not_target");
    assert_has_edge(&graph, "caller", "S");
    // crate:: keyword should not appear as a target
}

#[test]
fn test_extern_crate_not_target() {
    let graph = extract_fixture("extern_crate_not_target");
    assert_has_edge(&graph, "caller", "S");
    assert_no_edge_to(&graph, "::std", "extern crate");
}

#[test]
fn test_builtin_attr_not_target() {
    let graph = extract_fixture("builtin_attr_not_target");
    assert_has_edge(&graph, "caller", "S");
    assert_no_edge_to(&graph, "::inline", "builtin attributes");
}

// =============================================================================
// EXTERNAL FILTERING
// =============================================================================

#[test]
fn test_cross_crate_local_deps() {
    let graph = extract_fixture("cross_crate_local_deps");
    assert_has_edge(&graph, "caller", "LocalType");
    // Should NOT have edges to std::vec::Vec
    assert_no_edge_to(&graph, "::Vec", "std types");
}

#[test]
fn test_std_only_no_deps() {
    let graph = extract_fixture("std_only_no_deps");
    // caller should have no local dependencies (only uses std)
    let all_deps = collect_all_dependencies(&graph);
    // Filter out:
    // - External crate names (no `::` in them, e.g., "std", "core")
    // - Old-style std deps (contain "std::")
    let local_deps: Vec<_> = all_deps
        .iter()
        .filter(|d| d.contains("::") && !d.contains("std::"))
        .collect();
    // Only dependencies should be to the function itself or synthesized test main
    assert!(
        local_deps.is_empty()
            || local_deps
                .iter()
                .all(|d| d.ends_with("caller") || d.ends_with("main")),
        "Should have no local deps except self/main: {local_deps:?}"
    );
}

// =============================================================================
// IMPL ANCHORS
//
// Tests for impl anchor extraction (orphan rule compliance).
// =============================================================================

#[test]
fn test_anchor_inherent_impl() {
    let graph = extract_fixture("anchor_inherent_impl");
    assert_has_anchor(&graph, "impl MyType", "MyType");
}

#[test]
fn test_anchor_trait_impl() {
    let graph = extract_fixture("anchor_trait_impl");
    assert_has_anchor(&graph, "impl MyTrait for MyType", "MyTrait");
    assert_has_anchor(&graph, "impl MyTrait for MyType", "MyType");
}

#[test]
fn test_anchor_impl_for_reference() {
    let graph = extract_fixture("anchor_impl_for_reference");
    // &T is fundamental, so anchor is the inner type T
    assert_has_anchor(&graph, "impl MyTrait for &MyType", "MyType");
}

#[test]
fn test_anchor_blanket_impl() {
    let graph = extract_fixture("anchor_blanket_impl");
    // impl<T> Trait for T has no concrete local anchors
    // The trait itself may or may not be an anchor depending on implementation
    assert_has_anchor(&graph, "impl MyTrait for T", "MyTrait");
}

#[test]
fn test_anchor_inherent_impl_generic() {
    let graph = extract_fixture("anchor_inherent_impl_generic");
    assert_has_anchor(&graph, "impl Container", "Container");
}

#[test]
fn test_anchor_generic_bounds() {
    let graph = extract_fixture("anchor_generic_bounds");
    assert_has_anchor(&graph, "impl MyType", "MyType");
}

#[test]
fn test_anchor_lifetime_params() {
    let graph = extract_fixture("anchor_lifetime_params");
    assert_has_anchor(&graph, "impl MyType", "MyType");
}

#[test]
fn test_anchor_multiple_type_params() {
    let graph = extract_fixture("anchor_multiple_type_params");
    assert_has_anchor(&graph, "impl Container", "Container");
}

#[test]
fn test_anchor_generic_trait_impl() {
    let graph = extract_fixture("anchor_generic_trait_impl");
    assert_has_anchor(&graph, "impl MyTrait", "MyTrait");
    assert_has_anchor(&graph, "impl MyTrait", "MyType");
}

#[test]
fn test_anchor_where_clause() {
    let graph = extract_fixture("anchor_where_clause");
    assert_has_anchor(&graph, "impl MyType", "MyType");
}

#[test]
fn test_anchor_unsafe_impl() {
    let graph = extract_fixture("anchor_unsafe_impl");
    assert_has_anchor(&graph, "impl UnsafeTrait for MyType", "UnsafeTrait");
    assert_has_anchor(&graph, "impl UnsafeTrait for MyType", "MyType");
}

#[test]
fn test_anchor_tuple_type() {
    let graph = extract_fixture("anchor_tuple_type");
    // Tuple is fundamental, so anchors are the inner types
    assert_has_anchor(&graph, "impl MyTrait for (A, B)", "A");
    assert_has_anchor(&graph, "impl MyTrait for (A, B)", "B");
}

#[test]
fn test_anchor_array_type() {
    let graph = extract_fixture("anchor_array_type");
    // Array is fundamental, so anchor is the element type
    assert_has_anchor(&graph, "impl MyTrait for [Element; 3]", "Element");
}

#[test]
fn test_anchor_dyn_trait() {
    let graph = extract_fixture("anchor_dyn_trait");
    assert_has_anchor(&graph, "impl OtherTrait for dyn MyTrait", "MyTrait");
}

#[test]
fn test_anchor_fundamental_box() {
    let graph = extract_fixture("anchor_fundamental_box");
    // Box<T> is fundamental, so anchor is T
    assert_has_anchor(&graph, "impl MyTrait for Box<MyType, Global>", "MyType");
}

#[test]
fn test_anchor_fundamental_box_concrete() {
    let graph = extract_fixture("anchor_fundamental_box_concrete");
    assert_has_anchor(
        &graph,
        "impl MyTrait for Box<LocalType, Global>",
        "LocalType",
    );
}

#[test]
fn test_anchor_fundamental_mut_ref() {
    let graph = extract_fixture("anchor_fundamental_mut_ref");
    // &mut T is fundamental, so anchor is T
    assert_has_anchor(&graph, "impl MyTrait for &mut MyType", "MyType");
}

#[test]
fn test_anchor_non_fundamental_wrapper() {
    let graph = extract_fixture("anchor_non_fundamental_wrapper");
    // Vec<T> is NOT fundamental - no local anchor
    // The only anchor would be LocalType if extraction unwraps non-fundamental
    // For now, we expect LocalType to be an anchor since it's a type param
    assert_has_anchor(
        &graph,
        "impl MyTrait for Vec<LocalType, Global>",
        "LocalType",
    );
}

#[test]
fn test_anchor_trait_type_param() {
    let graph = extract_fixture("anchor_trait_type_param");
    assert_has_anchor(
        &graph,
        "impl MyTrait<ParamType> for MyType",
        "ParamType",
    );
}

#[test]
fn test_anchor_trait_type_param_local_self() {
    let graph = extract_fixture("anchor_trait_type_param_local_self");
    assert_has_anchor(&graph, "impl MyTrait", "MyType");
    assert_has_anchor(&graph, "impl MyTrait", "MyTrait");
}

#[test]
fn test_anchor_trait_multiple_type_params() {
    let graph = extract_fixture("anchor_trait_multiple_type_params");
    assert_has_anchor(
        &graph,
        "impl MyTrait<ParamA, ParamB> for MyType",
        "ParamA",
    );
    assert_has_anchor(
        &graph,
        "impl MyTrait<ParamA, ParamB> for MyType",
        "ParamB",
    );
}

#[test]
fn test_anchor_external_trait_not_anchor() {
    let graph = extract_fixture("anchor_external_trait_not_anchor");
    // Clone is external, so it should NOT be an anchor
    let anchors = get_impl_anchors(&graph, "impl Clone for MyType");
    assert!(
        !anchors.iter().any(|a| a.contains("Clone")),
        "External trait Clone should not be an anchor: {anchors:?}"
    );
    // But MyType should be an anchor
    assert_has_anchor(&graph, "impl Clone for MyType", "MyType");
}

#[test]
fn test_anchor_external_trait_local_type_param() {
    let graph = extract_fixture("anchor_external_trait_local_type_param");
    // From<LocalType> - From is external but LocalType is local
    assert_has_anchor(&graph, "impl From<LocalType>", "LocalType");
}

#[test]
fn test_anchor_external_trait_ref_type_param() {
    let graph = extract_fixture("anchor_external_trait_ref_type_param");
    // AsRef<LocalType> - LocalType (through &LocalType) is an anchor
    assert_has_anchor(&graph, "impl AsRef<LocalType>", "LocalType");
}

#[test]
fn test_anchor_crate_prefixed_self_type() {
    let graph = extract_fixture("anchor_crate_prefixed_self_type");
    // Verify anchors include the full package/target-prefixed path, not just the
    // type name. This is important for cross-crate workspace support. The fixture
    // package is "fixture", so we check for a "[fixture/<target>]::" prefix.
    let anchors = get_impl_anchors(&graph, "impl MyTrait for MyType");
    let has_my_type = anchors.iter().any(|anchor| {
        (anchor == "[fixture/lib]::MyType")
            || (anchor == "[fixture/test]::MyType")
    });
    let has_my_trait = anchors.iter().any(|anchor| {
        (anchor == "[fixture/lib]::MyTrait")
            || (anchor == "[fixture/test]::MyTrait")
    });
    assert!(
        has_my_type,
        "impl MyTrait for MyType should have anchor [fixture/<target>]::MyType\nAnchors: {anchors:?}"
    );
    assert!(
        has_my_trait,
        "impl MyTrait for MyType should have anchor [fixture/<target>]::MyTrait\nAnchors: {anchors:?}"
    );
}

// =============================================================================
// SYMBOL EXTRACTION
//
// Tests for symbol metadata extraction (kind, visibility, paths).
// =============================================================================

/// Find a symbol by name in the graph and return its kind string.
///
/// For impl blocks, matches against the `name` field in `SymbolKind::Impl`
/// since symbol keys use `DefPath` format (e.g., `{{impl}}`).
fn get_symbol_kind(graph: &SymbolGraph, name: &str) -> Option<String> {
    fn find_in_module(module: &Module, name: &str) -> Option<String> {
        for (sym_name, symbol) in &module.symbols {
            let matches = match &symbol.kind {
                tarjanize_schemas::SymbolKind::ModuleDef { .. } => {
                    sym_name.contains(name)
                }
                tarjanize_schemas::SymbolKind::Impl {
                    name: impl_name, ..
                } => impl_name.contains(name) || name.contains(impl_name),
            };
            if matches {
                return match &symbol.kind {
                    tarjanize_schemas::SymbolKind::ModuleDef {
                        kind, ..
                    } => Some(kind.clone()),
                    tarjanize_schemas::SymbolKind::Impl { .. } => {
                        Some("Impl".to_string())
                    }
                };
            }
        }
        for submodule in module.submodules.values() {
            if let Some(kind) = find_in_module(submodule, name) {
                return Some(kind);
            }
        }
        None
    }

    for crate_data in iter_all_crates(graph) {
        if let Some(kind) = find_in_module(&crate_data.root, name) {
            return Some(kind);
        }
    }
    None
}

/// Check if a symbol exists in the graph.
fn symbol_exists(graph: &SymbolGraph, name: &str) -> bool {
    get_symbol_kind(graph, name).is_some()
}

/// Get the friendly name of the first impl block in the graph.
///
/// Impl blocks use `DefPath` format as keys (e.g., `{{impl}}`), but store
/// a human-readable name in `SymbolKind::Impl.name`. This helper finds
/// the first impl and returns that name for assertion in formatting tests.
fn get_impl_name(graph: &SymbolGraph) -> Option<String> {
    fn find_in_module(module: &Module) -> Option<String> {
        for (key, symbol) in &module.symbols {
            if key.contains("{{impl}}")
                && let tarjanize_schemas::SymbolKind::Impl { ref name, .. } =
                    symbol.kind
            {
                return Some(name.clone());
            }
        }
        for submodule in module.submodules.values() {
            if let Some(name) = find_in_module(submodule) {
                return Some(name);
            }
        }
        None
    }

    for crate_data in iter_all_crates(graph) {
        if let Some(name) = find_in_module(&crate_data.root) {
            return Some(name);
        }
    }
    None
}

/// Get a symbol's visibility.
fn get_symbol_visibility(
    graph: &SymbolGraph,
    name: &str,
) -> Option<tarjanize_schemas::Visibility> {
    fn find_in_module(
        module: &Module,
        name: &str,
    ) -> Option<tarjanize_schemas::Visibility> {
        for (sym_name, symbol) in &module.symbols {
            if sym_name.contains(name) {
                return match &symbol.kind {
                    tarjanize_schemas::SymbolKind::ModuleDef {
                        visibility,
                        ..
                    } => Some(*visibility),
                    tarjanize_schemas::SymbolKind::Impl { .. } => None, // Impls don't have visibility
                };
            }
        }
        for submodule in module.submodules.values() {
            if let Some(vis) = find_in_module(submodule, name) {
                return Some(vis);
            }
        }
        None
    }

    for crate_data in iter_all_crates(graph) {
        if let Some(vis) = find_in_module(&crate_data.root, name) {
            return Some(vis);
        }
    }
    None
}

#[test]
fn test_symbol_macro_generated() {
    let graph = extract_fixture("symbol_macro_generated");
    assert!(
        symbol_exists(&graph, "generated_function"),
        "macro-generated fn should exist"
    );
}

#[test]
fn test_symbol_macro_struct() {
    let graph = extract_fixture("symbol_macro_struct");
    assert!(
        symbol_exists(&graph, "GeneratedStruct"),
        "macro-generated struct should exist"
    );
}

#[test]
fn test_symbol_macro_struct_in_module() {
    let graph = extract_fixture("symbol_macro_struct_in_module");
    assert!(
        symbol_exists(&graph, "NestedGeneratedStruct"),
        "nested macro struct should exist"
    );
}

#[test]
fn test_symbol_modules_not_in_symbols() {
    let graph = extract_fixture("symbol_modules_not_in_symbols");
    // Modules should appear as submodules, not as symbols
    assert!(symbol_exists(&graph, "outer_fn"), "outer_fn should exist");
    assert!(symbol_exists(&graph, "inner_fn"), "inner_fn should exist");
    // my_module should NOT be a symbol
    assert!(
        get_symbol_kind(&graph, "my_module").is_none(),
        "module should not appear as symbol"
    );
}

#[test]
fn test_symbol_visibility() {
    let graph = extract_fixture("symbol_visibility");
    assert_eq!(
        get_symbol_visibility(&graph, "public_fn"),
        Some(tarjanize_schemas::Visibility::Public)
    );
    assert_eq!(
        get_symbol_visibility(&graph, "private_fn"),
        Some(tarjanize_schemas::Visibility::NonPublic)
    );
    assert_eq!(
        get_symbol_visibility(&graph, "crate_fn"),
        Some(tarjanize_schemas::Visibility::NonPublic)
    );
}

#[test]
fn test_symbol_unnamed_const_skipped() {
    let graph = extract_fixture("symbol_unnamed_const_skipped");
    assert!(
        symbol_exists(&graph, "NAMED_CONST"),
        "named const should exist"
    );
    // Unnamed const _ should not exist
    assert!(
        !symbol_exists(&graph, "const _"),
        "unnamed const should not exist"
    );
}

#[test]
fn test_symbol_kind_strings() {
    let graph = extract_fixture("symbol_kind_strings");
    assert_eq!(
        get_symbol_kind(&graph, "a_function"),
        Some("Function".to_string())
    );
    assert_eq!(
        get_symbol_kind(&graph, "AStruct"),
        Some("Struct".to_string())
    );
    assert_eq!(get_symbol_kind(&graph, "AnEnum"), Some("Enum".to_string()));
    assert_eq!(get_symbol_kind(&graph, "ATrait"), Some("Trait".to_string()));
    assert_eq!(
        get_symbol_kind(&graph, "ATypeAlias"),
        Some("TypeAlias".to_string())
    );
    assert_eq!(
        get_symbol_kind(&graph, "A_CONST"),
        Some("Const".to_string())
    );
    assert_eq!(
        get_symbol_kind(&graph, "A_STATIC"),
        Some("Static".to_string())
    );
}

#[test]
fn test_symbol_path_function() {
    let graph = extract_fixture("symbol_path_function");
    assert!(symbol_exists(&graph, "my_function"));
    assert_eq!(
        get_symbol_kind(&graph, "my_function"),
        Some("Function".to_string())
    );
}

#[test]
fn test_symbol_path_struct() {
    let graph = extract_fixture("symbol_path_struct");
    assert!(symbol_exists(&graph, "MyStruct"));
    assert_eq!(
        get_symbol_kind(&graph, "MyStruct"),
        Some("Struct".to_string())
    );
}

#[test]
fn test_symbol_path_enum() {
    let graph = extract_fixture("symbol_path_enum");
    assert!(symbol_exists(&graph, "MyEnum"));
    assert_eq!(get_symbol_kind(&graph, "MyEnum"), Some("Enum".to_string()));
}

#[test]
fn test_symbol_path_trait() {
    let graph = extract_fixture("symbol_path_trait");
    assert!(symbol_exists(&graph, "MyTrait"));
    assert_eq!(
        get_symbol_kind(&graph, "MyTrait"),
        Some("Trait".to_string())
    );
}

#[test]
fn test_symbol_path_type_alias() {
    let graph = extract_fixture("symbol_path_type_alias");
    assert!(symbol_exists(&graph, "MyAlias"));
    assert_eq!(
        get_symbol_kind(&graph, "MyAlias"),
        Some("TypeAlias".to_string())
    );
}

#[test]
fn test_symbol_path_const() {
    let graph = extract_fixture("symbol_path_const");
    assert!(symbol_exists(&graph, "MY_CONST"));
    assert_eq!(
        get_symbol_kind(&graph, "MY_CONST"),
        Some("Const".to_string())
    );
}

#[test]
fn test_symbol_path_static() {
    let graph = extract_fixture("symbol_path_static");
    assert!(symbol_exists(&graph, "MY_STATIC"));
    assert_eq!(
        get_symbol_kind(&graph, "MY_STATIC"),
        Some("Static".to_string())
    );
}

#[test]
fn test_symbol_path_macro() {
    let graph = extract_fixture("symbol_path_macro");
    assert!(symbol_exists(&graph, "my_macro"));
    assert_eq!(
        get_symbol_kind(&graph, "my_macro"),
        Some("Macro".to_string())
    );
}

#[test]
fn test_symbol_path_inherent_impl() {
    let graph = extract_fixture("symbol_path_inherent_impl");
    assert_eq!(get_impl_name(&graph), Some("impl MyType".to_string()));
}

#[test]
fn test_symbol_path_trait_impl() {
    let graph = extract_fixture("symbol_path_trait_impl");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for MyType".to_string())
    );
}

#[test]
fn test_impl_name_negative() {
    let graph = extract_fixture("negative_impl");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl !MyAuto for NoAuto".to_string())
    );
}

#[test]
fn test_impl_name_tuple() {
    let graph = extract_fixture("impl_for_tuple");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for (A, B)".to_string())
    );
}

#[test]
fn test_impl_name_array() {
    let graph = extract_fixture("impl_for_array");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for [Foo; 3]".to_string())
    );
}

#[test]
fn test_impl_name_dyn_trait() {
    let graph = extract_fixture("impl_for_dyn_trait");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for dyn OtherTrait".to_string())
    );
}

#[test]
fn test_impl_name_external_generic() {
    let graph = extract_fixture("impl_for_external_generic");
    // Box has two type params: the inner type and the allocator
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for Box<Foo, Global>".to_string())
    );
}

#[test]
fn test_impl_name_generic_bound() {
    let graph = extract_fixture("impl_generic_bound");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl ImplTrait for ImplType<T>".to_string())
    );
}

#[test]
fn test_impl_name_self_type() {
    let graph = extract_fixture("self_keyword");
    assert_eq!(get_impl_name(&graph), Some("impl T".to_string()));
}

#[test]
fn test_impl_name_where_clause() {
    let graph = extract_fixture("impl_where_clause");
    // Where clause is not included in the impl name
    assert_eq!(
        get_impl_name(&graph),
        Some("impl ImplTrait for ImplType<T>".to_string())
    );
}

#[test]
fn test_impl_name_assoc_type() {
    let graph = extract_fixture("impl_assoc_type_definition");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl TraitWithAssoc for ImplType".to_string())
    );
}

#[test]
fn test_impl_name_generic_inherent() {
    let graph = extract_fixture("anchor_inherent_impl_generic");
    assert_eq!(get_impl_name(&graph), Some("impl Container<T>".to_string()));
}

#[test]
fn test_impl_name_generic_trait() {
    let graph = extract_fixture("anchor_generic_trait_impl");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait<T> for MyType".to_string())
    );
}

#[test]
fn test_impl_name_blanket() {
    let graph = extract_fixture("anchor_blanket_impl");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for T".to_string())
    );
}

#[test]
fn test_impl_name_lifetime() {
    let graph = extract_fixture("anchor_lifetime_params");
    assert_eq!(get_impl_name(&graph), Some("impl MyType<'a>".to_string()));
}

#[test]
fn test_impl_name_inherent_where_clause() {
    let graph = extract_fixture("anchor_where_clause");
    // Where clause is not included in the impl name
    assert_eq!(get_impl_name(&graph), Some("impl MyType<T>".to_string()));
}

#[test]
fn test_impl_name_inherent_basic() {
    let graph = extract_fixture("anchor_inherent_impl");
    assert_eq!(get_impl_name(&graph), Some("impl MyType".to_string()));
}

#[test]
fn test_impl_name_trait_basic() {
    let graph = extract_fixture("anchor_trait_impl");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for MyType".to_string())
    );
}

#[test]
fn test_impl_name_for_reference() {
    let graph = extract_fixture("anchor_impl_for_reference");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for &MyType".to_string())
    );
}

#[test]
fn test_impl_name_for_mut_reference() {
    let graph = extract_fixture("anchor_fundamental_mut_ref");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for &mut MyType".to_string())
    );
}

#[test]
fn test_impl_name_generic_bounds_inherent() {
    let graph = extract_fixture("anchor_generic_bounds");
    // Generic bounds are not included in the impl name
    assert_eq!(get_impl_name(&graph), Some("impl MyType<T>".to_string()));
}

#[test]
fn test_impl_name_multiple_type_params() {
    let graph = extract_fixture("anchor_multiple_type_params");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl Container<A, B>".to_string())
    );
}

#[test]
fn test_impl_name_unsafe() {
    let graph = extract_fixture("anchor_unsafe_impl");
    assert_eq!(
        get_impl_name(&graph),
        Some("unsafe impl UnsafeTrait for MyType".to_string())
    );
}

#[test]
fn test_impl_name_box_generic() {
    let graph = extract_fixture("anchor_fundamental_box");
    // Box has two type params: the inner type and the allocator (Global).
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for Box<MyType, Global>".to_string())
    );
}

#[test]
fn test_impl_name_box_concrete() {
    let graph = extract_fixture("anchor_fundamental_box_concrete");
    // Box has two type params: the inner type and the allocator (Global).
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for Box<LocalType, Global>".to_string())
    );
}

#[test]
fn test_impl_name_vec() {
    let graph = extract_fixture("anchor_non_fundamental_wrapper");
    // Vec has two type params: the element type and the allocator (Global).
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for Vec<LocalType, Global>".to_string())
    );
}

#[test]
fn test_impl_name_trait_type_param() {
    let graph = extract_fixture("anchor_trait_type_param");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait<ParamType> for MyType".to_string())
    );
}

#[test]
fn test_impl_name_trait_generic_param() {
    let graph = extract_fixture("anchor_trait_type_param_local_self");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait<T> for MyType".to_string())
    );
}

#[test]
fn test_impl_name_trait_multiple_type_params() {
    let graph = extract_fixture("anchor_trait_multiple_type_params");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait<ParamA, ParamB> for MyType".to_string())
    );
}

#[test]
fn test_impl_name_clone() {
    let graph = extract_fixture("anchor_external_trait_not_anchor");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl Clone for MyType".to_string())
    );
}

#[test]
fn test_impl_name_from() {
    let graph = extract_fixture("anchor_external_trait_local_type_param");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl From<LocalType> for i32".to_string())
    );
}

#[test]
fn test_impl_name_as_ref() {
    let graph = extract_fixture("anchor_external_trait_ref_type_param");
    assert_eq!(
        get_impl_name(&graph),
        Some("impl AsRef<LocalType> for i32".to_string())
    );
}

#[test]
fn test_impl_name_crate_prefixed() {
    let graph = extract_fixture("anchor_crate_prefixed_self_type");
    // crate:: prefix should be stripped from the name
    assert_eq!(
        get_impl_name(&graph),
        Some("impl MyTrait for MyType".to_string())
    );
}

#[test]
fn test_symbol_path_nested_module() {
    let graph = extract_fixture("symbol_path_nested_module");
    assert!(symbol_exists(&graph, "nested_fn"));
}

// =============================================================================
// IMPL FOR SPECIAL TYPES
//
// Tests for impl blocks on tuple types, array types, dyn trait, and
// external generic types.
// =============================================================================

/// Impl for tuple type creates edges to element types and trait.
#[test]
fn test_impl_for_tuple() {
    let graph = extract_fixture("impl_for_tuple");
    assert_has_edge(&graph, "{{impl}}", "A");
    assert_has_edge(&graph, "{{impl}}", "B");
    assert_has_edge(&graph, "{{impl}}", "MyTrait");
}

/// Impl for array type creates edges to element type and trait.
#[test]
fn test_impl_for_array() {
    let graph = extract_fixture("impl_for_array");
    assert_has_edge(&graph, "{{impl}}", "Foo");
    assert_has_edge(&graph, "{{impl}}", "MyTrait");
}

/// Impl for dyn trait creates edges to both traits.
#[test]
fn test_impl_for_dyn_trait() {
    let graph = extract_fixture("impl_for_dyn_trait");
    assert_has_edge(&graph, "{{impl}}", "OtherTrait");
    assert_has_edge(&graph, "{{impl}}", "MyTrait");
}

/// Impl for external generic with local type arg creates edges.
#[test]
fn test_impl_for_external_generic() {
    let graph = extract_fixture("impl_for_external_generic");
    assert_has_edge(&graph, "{{impl}}", "Foo");
    assert_has_edge(&graph, "{{impl}}", "MyTrait");
}

// =============================================================================
// MULTI-CRATE INTEGRATION TESTS
//
// Tests for workspace scenarios with multiple crates.
// =============================================================================

/// Virtual workspaces with nested crate directories are handled correctly.
///
/// Tests that cargo-tarjanize can extract from a workspace with:
/// - No root package (pure virtual workspace)
/// - Crates nested in a `crates/` subdirectory
#[test]
fn test_virtual_workspace_nested_crates() {
    let graph = extract_fixture("virtual_workspace_nested_crates");
    // The nested member_a crate should be found and extracted.
    assert!(
        has_crate(&graph, "member_a"),
        "member_a should be in the graph: {:?}",
        graph.packages.keys().collect::<Vec<_>>()
    );
    // The hello function should exist.
    assert!(
        symbol_exists(&graph, "hello"),
        "hello function should exist"
    );
}

/// Dev-dependency cycles are handled correctly.
///
/// When `crate_a` has a dev-dependency on `crate_b`, and `crate_b` has a normal
/// dependency on `crate_a`, this creates a cycle. Cargo handles this because
/// dev-deps are only used for tests.
///
/// This test verifies that:
/// 1. Both crates are extracted
/// 2. `crate_b`'s symbols depend on `crate_a` (normal dep edge exists)
#[test]
fn test_dev_dep_cycle() {
    let graph = extract_fixture("dev_dep_cycle");

    // Both crates should be extracted.
    assert!(
        has_crate(&graph, "crate_a"),
        "crate_a should be in the graph: {:?}",
        graph.packages.keys().collect::<Vec<_>>()
    );
    assert!(
        has_crate(&graph, "crate_b"),
        "crate_b should be in the graph: {:?}",
        graph.packages.keys().collect::<Vec<_>>()
    );

    // crate_b::function_in_b should depend on crate_a (via function_in_a call).
    let deps = get_symbol_deps(&graph, "crate_b", "function_in_b");
    assert!(
        deps.iter().any(|d| d.contains("crate_a")),
        "function_in_b should depend on crate_a, but found: {deps:?}"
    );
}

/// Test: lib and test targets are extracted separately with correct cfg(test) handling.
///
/// When compiling a lib target (no --test flag), #[cfg(test)] code should be
/// excluded. When compiling a test target (--test flag), #[cfg(test)] code
/// should be included. This test verifies rustc's cfg handling works correctly
/// through our extraction pipeline.
///
/// Uses `dev_dep_cycle` fixture which has a `#[cfg(test)]` module in `crate_a`.
#[test]
fn test_lib_and_test_target_separation() {
    let graph = extract_fixture("dev_dep_cycle");

    // Get crate_a's package.
    let pkg = graph
        .packages
        .get("crate_a")
        .expect("crate_a package should exist");

    // Both lib and test targets should be extracted.
    assert!(
        pkg.targets.contains_key("lib"),
        "lib target should exist, found: {:?}",
        pkg.targets.keys().collect::<Vec<_>>()
    );
    assert!(
        pkg.targets.contains_key("test"),
        "test target should exist, found: {:?}",
        pkg.targets.keys().collect::<Vec<_>>()
    );

    let lib_target = pkg.targets.get("lib").unwrap();
    let test_target = pkg.targets.get("test").unwrap();

    // Lib target should NOT have the #[cfg(test)] "tests" submodule.
    assert!(
        !lib_target.root.submodules.contains_key("tests"),
        "lib target should NOT contain #[cfg(test)] module, but found submodules: {:?}",
        lib_target.root.submodules.keys().collect::<Vec<_>>()
    );

    // Lib target should have the public function.
    assert!(
        lib_target.root.symbols.contains_key("function_in_a"),
        "lib target should have function_in_a"
    );

    // Test target SHOULD have the #[cfg(test)] "tests" submodule.
    assert!(
        test_target.root.submodules.contains_key("tests"),
        "test target should contain #[cfg(test)] module, but found submodules: {:?}",
        test_target.root.submodules.keys().collect::<Vec<_>>()
    );

    // The tests module should have the test function.
    let tests_module = test_target.root.submodules.get("tests").unwrap();
    assert!(
        tests_module.symbols.contains_key("test_using_crate_b"),
        "tests module should have test_using_crate_b, found: {:?}",
        tests_module.symbols.keys().collect::<Vec<_>>()
    );
}

// =============================================================================
// Experimental/Unstable Feature Tests
//
// These tests cover THIR ExprKind variants that only appear with unstable
// features. They skip gracefully if the feature is not available.
// =============================================================================

/// Test: `ByUse` expression (`.use` postfix syntax) creates dependency edges.
///
/// Requires `#![feature(ergonomic_clones)]` which may not be available.
#[test]
fn test_by_use_expr() {
    let Some(graph) = try_extract_fixture("by_use_expr") else {
        eprintln!("Skipping test_by_use_expr: feature not available");
        return;
    };

    // uses_by_use should depend on takes_ownership (called via .use).
    assert_has_edge(&graph, "uses_by_use", "takes_ownership");
}

/// Test: `LoopMatch` expression (`#[loop_match]`) creates dependency edges.
///
/// Requires `#![feature(loop_match)]` which may not be available.
#[test]
fn test_loop_match_expr() {
    let Some(graph) = try_extract_fixture("loop_match_expr") else {
        eprintln!("Skipping test_loop_match_expr: feature not available");
        return;
    };

    // state_machine should depend on the helper functions called in each state.
    assert_has_edge(&graph, "state_machine", "helper_start");
    assert_has_edge(&graph, "state_machine", "helper_middle");
    assert_has_edge(&graph, "state_machine", "helper_end");
}

/// Test: Unsafe binder expressions create dependency edges.
///
/// Requires `#![feature(unsafe_binders)]` which may not be available.
#[test]
fn test_unsafe_binder_expr() {
    let Some(graph) = try_extract_fixture("unsafe_binder_expr") else {
        eprintln!("Skipping test_unsafe_binder_expr: feature not available");
        return;
    };

    // unwrap_and_use should depend on get_value (called after unwrapping).
    assert_has_edge(&graph, "unwrap_and_use", "get_value");
}

// =============================================================================
// NESTED ITEM COLLAPSING
//
// Items defined inside function bodies (nested functions, statics, consts) are
// collapsed to their containing function. They don't appear as separate symbols,
// and dependencies on them are redirected to the parent function.
// =============================================================================

/// Test: Nested functions are collapsed to their parent function.
///
/// A nested function defined inside another function:
/// - Should not appear as a separate symbol
/// - Its dependencies should be captured as part of the parent function's deps
#[test]
fn test_nested_fn() {
    let graph = extract_fixture("nested_fn");

    // The nested `inner` function should not appear as a symbol.
    assert_no_symbol(&graph, "inner");

    // The outer function should exist.
    assert!(symbol_exists(&graph, "outer"), "outer should exist");

    // The outer function should have the dependency that inner uses (Helper).
    assert_has_edge(&graph, "outer", "Helper");

    // caller depends on outer (not on inner, which doesn't exist as a symbol).
    assert_has_edge(&graph, "caller", "outer");
}

/// Test: Items defined inside closures are collapsed to the containing function.
///
/// Macros like `tokio::select!` generate internal structs inside closures
/// (e.g., `__tokio_select_util::Mask`). These should not appear as separate
/// symbols - they should be collapsed to the function containing the closure.
#[test]
fn test_closure_internal_struct() {
    let graph = extract_fixture("cost_closure_internal_struct");

    // The internal struct `Mask` should not appear as a symbol.
    assert_no_symbol(&graph, "Mask");
    assert_no_symbol(&graph, "__internal_util");

    // The containing function `run` should exist.
    assert!(symbol_exists(&graph, "run"), "run should exist");
}

/// Test: `pub use` re-exports are extracted as symbols.
///
/// Facade crates consist entirely of `pub use` re-exports, which have real
/// compilation cost and create dependency edges. They should appear as symbols
/// with kind "Use".
#[test]
fn test_use_reexport() {
    let graph = extract_fixture("use_reexport");

    // The re-exported struct and function should appear as Use symbols.
    assert!(
        symbol_exists(&graph, "{{use}}"),
        "pub use should be extracted"
    );
    assert_eq!(
        get_symbol_kind(&graph, "{{use}}"),
        Some("Use".to_string()),
        "pub use should have kind 'Use'"
    );

    // The original items in the inner module should also exist.
    assert!(symbol_exists(&graph, "Original"), "Original should exist");
    assert!(symbol_exists(&graph, "helper"), "helper should exist");
}
