# Code Review Checklist

## Follow-up Tasks

- [ ] Audit logging levels: downgrade INFO logs to DEBUG where appropriate
- [ ] Implement proptest strategies for generating structurally valid SymbolGraph instances (dependencies and Impl paths reference actual symbols). Requires two-phase generation: create symbol structure first, then populate dependencies from the set of valid symbol paths.

## tarjanize

### main.rs

- [x] `Cli` struct
- [x] `Commands` enum
- [x] `main()` function

## tarjanize-schemas

### lib.rs

- [x] module docs

### symbol_graph.rs

#### Production Code

- [x] `SymbolGraph` struct
- [x] `Module` struct
- [x] `Symbol` struct
- [x] `SymbolKind` enum
- [x] `Edge` struct

#### Tests

- [x] `test_symbol_graph_roundtrip`

## tarjanize-extract

### lib.rs

#### Production Code

- [x] `file_path()`
- [x] `extract_symbol_graph()`
- [x] `run()`

#### Tests

- [x] `test_extract_symbol_graph_basic`

### error.rs

#### Production Code

- [ ] `ExtractError` + `ExtractErrorKind`
- [ ] `ExtractError` methods
- [ ] `Display`/`Error`/`From` impls

#### Tests

- [ ] `test_file_path_not_found`
- [ ] `test_crate_name_missing`
- [ ] `test_crate_root_no_parent`
- [ ] `test_workspace_load`
- [ ] `test_serialization`
- [ ] `test_io`
- [ ] `test_backtrace`
- [ ] `test_debug_impl`

### workspaces.rs

#### Production Code

- [x] `load_workspace()`

#### Tests

- [x] `test_load_workspace_succeeds`
- [x] `test_load_workspace_fails_for_nonexistent_path`

### crates.rs

#### Production Code

- [x] `extract_crate()`

#### Tests

- [x] `test_extract_crate_returns_name_and_symbols`

### modules.rs

#### Production Code

- [x] `extract_module()`
- [x] `extract_module_symbols()`

### module_defs.rs

#### Production Code

- [x] `extract_module_def()`

#### Tests

- [ ] `test_macro_generated_symbol_has_file_path`
- [ ] `test_modules_not_in_symbols`
- [ ] `test_visibility`
- [ ] `test_unnamed_const_skipped`
- [ ] `test_kind_strings`

### impls.rs

#### Production Code

- [ ] `extract_impl()`
- [ ] `find_impl_dependencies()`
- [ ] `ImplDependencies` struct

### paths.rs

#### Production Code

- [x] `build_module_path()`
- [ ] `module_def_path()`
- [ ] `definition_path()`
- [ ] `impl_path()`
- [ ] `impl_name()`
- [ ] `compute_relative_file_path()`

### dependencies.rs

#### Production Code

- [ ] `is_local_def()`
- [ ] `is_local()`
- [ ] `find_dependencies()`
- [ ] `find_node_in_file()`
- [ ] `collect_deps_from()`
- [ ] `collect_path_deps()`
- [ ] `normalize_module_def()`
- [ ] `collapse_assoc_item()`
- [ ] `variant_def_to_adt_def()`
- [ ] `normalize_definition()`
- [ ] `collect_expr_deps()`

#### Tests

- [ ] `test_fixture_fn_call`
- [ ] `test_fixture_struct_field`
- [ ] `test_fixture_trait_bound`
- [ ] `test_fixture_impl_block`
- [ ] `test_fixture_cross_crate`
- [ ] `test_fixture_const_static`
- [ ] `test_fixture_type_alias`
- [ ] `test_fixture_submodule`
- [ ] `test_fixture_inherent_impl`
- [ ] `test_fixture_fn_param_type`
- [ ] `test_fixture_fn_return_type`
- [ ] `test_fixture_fn_local_var`
- [ ] `test_fixture_fn_multiple_deps`
- [ ] `test_fixture_fn_generic_bound`
- [ ] `test_fixture_fn_where_clause`
- [ ] `test_fixture_struct_generic_field`
- [ ] `test_fixture_struct_tuple`
- [ ] `test_fixture_struct_unit`
- [ ] `test_fixture_enum_variant_struct`
- [ ] `test_fixture_enum_variant_tuple`
- [ ] `test_fixture_trait_method_param`
- [ ] `test_fixture_trait_method_return`
- [ ] `test_fixture_trait_assoc_type_bound`
- [ ] `test_fixture_trait_supertrait`
- [ ] `test_fixture_impl_method_body`
- [ ] `test_fixture_impl_generic_constraint`
- [ ] `test_fixture_impl_trait_for_type`
- [ ] `test_fixture_method_call_inherent`
- [ ] `test_fixture_method_call_trait`
- [ ] `test_fixture_qualified_path`
- [ ] `test_fixture_use_statement`
- [ ] `test_fixture_macro_use`
- [ ] `test_fixture_type_alias_generic`
- [ ] `test_fixture_const_type`
- [ ] `test_fixture_static_type`
- [ ] `test_fixture_closure_capture`
- [ ] `test_fixture_async_fn`
- [ ] `test_fixture_impl_deref`
- [ ] `test_fixture_self_referential`
