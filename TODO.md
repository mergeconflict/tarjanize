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

- [x] `is_local_def()`
- [x] `collect_path_deps()`
- [x] `normalize_definition()`
- [x] `collapse_if_assoc()`
- [x] `variant_def_to_adt()`

#### Tests

Tests are organized into 9 sections covering dependency sources, locations,
normalization, resolution mechanisms, and filtering. All 78 tests reviewed.

- [x] BASIC DEPENDENCY TESTS (11 tests)
- [x] FUNCTION DEPENDENCY LOCATIONS (11 tests)
- [x] ADT DEPENDENCY LOCATIONS (6 tests)
- [x] TRAIT DEPENDENCY LOCATIONS (3 tests)
- [x] IMPL AND ITEM DEPENDENCY LOCATIONS (5 tests)
- [x] EDGE TARGET NORMALIZATION (12 tests)
- [x] OTHER RESOLUTION MECHANISMS (17 tests)
- [x] EDGE TARGET FILTERING (11 tests)
- [x] EXTERNAL DEPENDENCY FILTERING (2 tests)
