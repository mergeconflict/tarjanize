# Code Review Checklist

## Follow-up Tasks

- [x] Audit logging levels: downgrade INFO logs to DEBUG where appropriate
- [ ] Implement proptest strategies for generating structurally valid SymbolGraph instances (dependencies and Impl paths reference actual symbols). Requires two-phase generation: create symbol structure first, then populate dependencies from the set of valid symbol paths.
- [ ] Experiment with further parallelization in `extract_module_symbols` using rayon. Currently crates are processed in parallel, but symbol extraction within modules is sequential. Considerations: (1) `Semantics` must be `Send + Sync` to use `par_iter()`, (2) may cause lock contention on the database if already parallel at crate level, (3) overhead may outweigh benefit for small modules.

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
- [x] `find_dependencies()`

#### Tests

- [ ] `test_macro_generated_symbol_has_file_path`
- [ ] `test_modules_not_in_symbols`
- [ ] `test_visibility`
- [ ] `test_unnamed_const_skipped`
- [ ] `test_kind_strings`

### impls.rs

#### Production Code

- [x] `extract_impl()`
- [x] `find_dependencies()`

#### Tests

- [x] `test_inherent_impl`
- [x] `test_trait_impl`
- [x] `test_impl_for_reference`
- [x] `test_blanket_impl`
- [x] `test_impl_merging`

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
- [ ] `find_node_in_file()`
- [ ] `collect_deps_from()`
- [ ] `collect_path_deps()`
- [ ] `normalize_module_def()`
- [ ] `collapse_assoc_item()`
- [ ] `variant_def_to_adt_def()`
- [ ] `normalize_definition()`
- [ ] `collect_expr_dep()`

#### Tests

- [ ] `test_function_call`
- [ ] `test_struct_field`
- [ ] `test_trait_bound`
- [ ] `test_impl_block`
- [ ] `test_cross_crate`
- [ ] `test_const_static`
- [ ] `test_type_alias`
- [ ] `test_submodule`
- [ ] `test_inherent_impl`
- [ ] `test_fn_param_type`
- [ ] `test_fn_return_type`
- [ ] `test_fn_body_type`
- [ ] `test_fn_where_clause`
- [ ] `test_struct_trait_bound`
- [ ] `test_struct_where_clause`
- [ ] `test_enum_tuple_variant`
- [ ] `test_enum_struct_variant`
- [ ] `test_enum_trait_bound`
- [ ] `test_union_field`
- [ ] `test_trait_supertrait`
- [ ] `test_trait_assoc_type_bound`
- [ ] `test_trait_default_method`
- [ ] `test_trait_default_const`
- [ ] `test_impl_method_body_deps`
- [ ] `test_impl_assoc_type`
- [ ] `test_method_call_inherent`
- [ ] `test_method_call_trait`
- [ ] `test_enum_variant_collapses`
- [ ] `test_module_not_edge_target`
- [ ] `test_trait_assoc_const_collapses`
- [ ] `test_impl_for_reference`
- [ ] `test_impl_for_mut_reference`
- [ ] `test_std_only_no_local_deps`
- [ ] `test_const_with_initializer_deps`
- [ ] `test_static_with_initializer_deps`
- [ ] `test_type_alias_with_generic_deps`
- [ ] `test_trait_default_method_call`
- [ ] `test_impl_assoc_const_as_dependency`
- [ ] `test_impl_assoc_type_as_dependency`
- [ ] `test_trait_with_assoc_const_default`
- [ ] `test_dependency_to_const`
- [ ] `test_dependency_to_static`
- [ ] `test_dependency_to_type_alias`
- [ ] `test_assoc_fn_via_path`
- [ ] `test_callable_field`
