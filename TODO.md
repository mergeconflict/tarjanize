# Code Review Checklist

## Follow-up Tasks

- [ ] Audit logging levels: downgrade INFO logs to DEBUG where appropriate
- [ ] Implement proptest strategies for generating structurally valid SymbolGraph instances (paths in Edge/Impl reference actual symbols)

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

- [ ] `test_load_nonexistent_path`
- [ ] `test_load_workspace_and_extract`

### crates.rs

#### Production Code

- [ ] `extract_crate()`

### modules.rs

#### Production Code

- [ ] `module_def_module()`
- [ ] `extract_module()`
- [ ] `build_module_path()`
- [ ] `extract_symbols()`
- [ ] `extract_module_def()`
- [ ] `extract_impl()`
- [ ] `impl_name()`
- [ ] `module_def_kind_str()`
- [ ] `module_def_path()`
- [ ] `dependency_path()`
- [ ] `impl_path()`
- [ ] `compute_relative_file_path()`
- [ ] `extract_visibility()`

#### Tests

- [ ] `test_macro_generated_symbol_has_file_path`

### dependencies.rs

#### Production Code

- [ ] `Dependency` enum + `From` impls
- [ ] `is_local_dep()`
- [ ] `is_local()`
- [ ] `find_dependencies()`
- [ ] `find_node_in_file()`
- [ ] `collect_deps_from()`
- [ ] `collect_path_deps()`
- [ ] `normalize_module_def()`
- [ ] `collapse_assoc_item()`
- [ ] `collect_expr_deps()`
- [ ] `find_impl_dependencies()`
- [ ] `ImplDependencies` struct

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
- [ ] `test_fixture_visibility`
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
