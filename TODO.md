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

### impls.rs

#### Production Code

- [x] `impl_name()` - moved from paths.rs
- [x] `extract_impl()`
- [x] `find_dependencies()`

#### Tests

- [x] IMPL EXTRACTION TESTS (15 tests) - tests name generation and symbol extraction

### modules.rs

#### Tests

- [x] `test_impl_merging` - tests that multiple impl blocks with same signature get merged

### paths.rs

#### Production Code

- [x] `module_def_path()` - uses rust-analyzer's `canonical_path`
- [x] `impl_path()`
- [ ] `compute_relative_file_path()`

### module_defs.rs

#### Production Code

- [x] `extract_module_def()`
- [x] `find_dependencies()`
- [x] `definition_path()` - moved from paths.rs, only used here

#### Tests

- [x] `test_macro_generated_symbol_has_file_path`
- [x] `test_modules_not_in_symbols`
- [x] `test_visibility`
- [x] `test_unnamed_const_skipped`
- [x] `test_kind_strings`
- [x] DEFINITION PATH TESTS (12 tests)

### dependencies.rs

#### Production Code

- [x] `is_local_def()`
- [x] `collect_path_deps()`
- [x] `normalize_definition()`
- [x] `collapse_if_assoc()`
- [x] `variant_def_to_adt()`

#### Tests

Tests are organized into 9 sections covering dependency sources, locations,
normalization, resolution mechanisms, and filtering. All 84 tests reviewed.

- [x] BASIC DEPENDENCY TESTS (11 tests)
- [x] FUNCTION DEPENDENCY LOCATIONS (11 tests)
- [x] ADT DEPENDENCY LOCATIONS (6 tests)
- [x] TRAIT DEPENDENCY LOCATIONS (3 tests)
- [x] IMPL AND ITEM DEPENDENCY LOCATIONS (11 tests)
- [x] EDGE TARGET NORMALIZATION (12 tests)
- [x] OTHER RESOLUTION MECHANISMS (17 tests)
- [x] EDGE TARGET FILTERING (11 tests)
- [x] EXTERNAL DEPENDENCY FILTERING (2 tests)
