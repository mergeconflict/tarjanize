# Plan: Separate Lib/Test/Bin Targets

This document describes the refactor to separate compilation targets (lib, test, bin, etc.) in the symbol graph, rather than merging them into a single entry.

## Motivation

Currently, we merge lib and test compilations into a single `Crate` entry. This causes problems:

1. **Dev-dependency cycles**: When crate A's tests depend on crate B, and crate B depends on crate A, we see an apparent cycle. But in reality there's no cycle - tests depend on libs, not vice versa.

2. **Inaccurate costs**: We take `max(lib_cost, test_cost)` which loses information.

3. **Inaccurate critical path**: The lib→test dependency isn't modeled.

By separating targets, we get:
- Natural resolution of dev-dependency "cycles"
- Accurate per-target costs
- More accurate critical path analysis (includes lib→test edges)

This directly addresses TODO B.2 in `docs/cost-model-validation.md`: "Fix Critical Path Computation for Dev-Dependency Cycles." Currently, critical path computation fails when dev-dependencies create apparent cycles. With target separation, there are no cycles - tests depend on libs, not vice versa.

## Terminology

We use Cargo and rustc terminology precisely:

| Term | Definition | Example |
|------|------------|---------|
| **Package** | A Cargo.toml and its contents. Has a unique name in the workspace. | `my-package` |
| **Target** | A compilation unit within a package. Cargo's term. | lib, bin, test, example, bench |
| **Crate** | What rustc compiles. Each target compiles to a crate. | `my_package` (lib), `cli` (bin) |
| **Crate name** | The name rustc uses, with underscores. | `my_package` |
| **Package name** | The name Cargo uses, may have hyphens. | `my-package` |

Key insight: A **Package** contains multiple **Targets**, and each **Target** compiles to a **Crate**.

## Schema Changes

### Current Schema

```rust
pub struct SymbolGraph {
    pub crates: HashMap<String, Crate>,  // keyed by crate name (underscores)
}

pub struct Crate {
    pub linking_ms: f64,
    pub metadata_ms: f64,
    pub dependencies: HashSet<String>,
    pub dev_dependencies: HashSet<String>,
    pub root: Module,
}
```

### New Schema

```rust
pub struct SymbolGraph {
    /// All packages in the workspace, keyed by package name (hyphens preserved).
    pub packages: HashMap<String, Package>,
}

pub struct Package {
    /// Compilation targets for this package.
    /// Keys: "lib", "test", "bin/{name}", "example/{name}", "bench/{name}"
    pub targets: HashMap<String, Crate>,
}

pub struct Crate {
    pub linking_ms: f64,
    pub metadata_ms: f64,
    /// Dependencies on other crates, e.g., "other-package/lib", "my-package/lib"
    pub dependencies: HashSet<String>,
    pub root: Module,
}
```

**Key changes:**
- Top-level is `packages` (keyed by package name with hyphens)
- `Package` contains a map of targets
- Each target value is a `Crate` (what rustc compiles)
- `dev_dependencies` field removed (absorbed into test target's `dependencies`)

### Target Keys

| Target Type | Key Format | Example |
|-------------|------------|---------|
| Library | `"lib"` | `"lib"` |
| Unit tests | `"test"` | `"test"` |
| Binary | `"bin/{name}"` | `"bin/cli"` |
| Example | `"example/{name}"` | `"example/basic"` |
| Benchmark | `"bench/{name}"` | `"bench/perf"` |

Integration tests (files in `tests/`) are separate packages from Cargo's perspective and get their own entry in the top-level `packages` map.

### Path Notation

**Crate references** (in `Crate.dependencies`):
```
my-package/lib
other-package/lib
my-package/bin/cli
```

**Symbol references** (in `Symbol.dependencies`):
```
[my-package/lib]::module::symbol
[other-package/lib]::SomeType
```

Brackets `[]` delimit the package/target portion in symbol paths. Package names preserve hyphens.

### Example Output

For a package `my-package` with lib, unit tests, and binary `cli`:

```json
{
  "packages": {
    "my-package": {
      "targets": {
        "lib": {
          "linking_ms": 100.0,
          "metadata_ms": 50.0,
          "dependencies": ["other-package/lib"],
          "root": { "symbols": { ... } }
        },
        "test": {
          "linking_ms": 80.0,
          "metadata_ms": 40.0,
          "dependencies": ["my-package/lib", "test-utils/lib"],
          "root": { "symbols": { ... } }
        },
        "bin/cli": {
          "linking_ms": 60.0,
          "metadata_ms": 30.0,
          "dependencies": ["my-package/lib"],
          "root": { "symbols": { ... } }
        }
      }
    }
  }
}
```

## Architecture / Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Orchestrator                                                             │
│                                                                          │
│  1. Read cargo metadata                                                  │
│     - Get packages, targets, dependencies                                │
│     - Write mapping file for driver (see below)                          │
│                                                                          │
│  2. Run cargo build with RUSTC_WRAPPER                                   │
│     │                                                                    │
│     ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Driver (invoked per target by cargo)                                │ │
│  │                                                                     │ │
│  │  - Reads mapping file to find its package/target from src_path      │ │
│  │  - Extracts symbols (paths in crate-name format)                    │ │
│  │  - Applies costs from profiler (also crate-name format)             │ │
│  │  - Writes to {output_dir}/{package}/{target_key}.json               │ │
│  │                                                                     │ │
│  │  Driver works entirely in crate-name space (rustc's native format)  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│     │                                                                    │
│     ▼                                                                    │
│  3. Read driver output files                                             │
│     - Directory structure identifies package/target                      │
│     - Build crate_name → (package, target) mapping from cargo metadata   │
│                                                                          │
│  4. Transform symbol paths                                               │
│     - Driver outputs: `other_crate::SomeType`                            │
│     - Transform to: `[other-package/lib]::SomeType`                      │
│     - Uses crate_name → package mapping from cargo metadata              │
│                                                                          │
│  5. Populate target dependencies from cargo metadata                     │
│     - Lib: normal deps → `"{dep-package}/lib"`                           │
│     - Test: normal + dev deps + `"{self-package}/lib"`                   │
│     - Bin: normal deps + `"{self-package}/lib"`                          │
│                                                                          │
│  6. Write symbol_graph.json                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Mapping File

The orchestrator writes a mapping file before running cargo build. The driver reads this to know which package/target it's compiling and where to write output.

**Why this approach?**
- The driver is invoked by cargo via RUSTC_WRAPPER, not by the orchestrator directly
- All driver invocations see the same environment variables
- The driver needs to disambiguate same-named binaries from different packages
- The source file path (from rustc args) uniquely identifies each target

**Mapping file format** (`{output_dir}/target_mapping.json`):
```json
{
  "/workspace/crates/my-package/src/lib.rs": {
    "package": "my-package",
    "target_key": "lib"
  },
  "/workspace/crates/my-package/src/bin/cli.rs": {
    "package": "my-package",
    "target_key": "bin/cli"
  },
  "/workspace/crates/other-package/src/bin/cli.rs": {
    "package": "other-package",
    "target_key": "bin/cli"
  }
}
```

**Driver behavior:**
1. Extract source file path from rustc args
2. Look up in mapping file to get package and target_key
3. Create output directory: `{output_dir}/{package}/`
4. Write output: `{output_dir}/{package}/{target_key}.json`
   - Note: `target_key` like `bin/cli` creates nested directories

## Path Transformation

**Why the orchestrator transforms paths (not the driver):**

The driver only needs to know where to write its own output. To transform dependency paths like `other_crate::Foo` → `[other-package/lib]::Foo`, the driver would need a mapping from ALL workspace crate names to their packages.

The orchestrator already has cargo metadata with this complete picture. Centralizing the transformation there:
- Keeps the driver simpler (it works in rustc's native crate-name format)
- Avoids passing redundant information to the driver
- Ensures consistent transformation using a single source of truth

**Transformation process:**

The orchestrator builds a `crate_name → (package, target)` mapping from cargo metadata:
```
my_package → (my-package, lib)
cli → (my-package, bin/cli)      # assuming my-package has this bin
cli → (other-package, bin/cli)   # collision! see below
```

For binaries, the crate name is the binary name, which can collide across packages. The orchestrator handles this by:
1. Processing each driver output file
2. The file path tells us the package: `{output_dir}/{package}/{target_key}.json`
3. When transforming dependency paths in that file, we know the context

For cross-package dependencies (e.g., `other_crate::Foo`), we assume lib target since that's what rustc resolves to for `use other_crate::Foo`.

## Implementation Plan

### Phase 1: Schema Changes (`tarjanize-schemas`)

**Files:** `crates/tarjanize-schemas/src/symbol_graph.rs`

- [ ] Rename `Crate` struct to `Package`
- [ ] Add new `Crate` struct (was `Target`) with: `linking_ms`, `metadata_ms`, `dependencies`, `root`
- [ ] Change `Package` (was `Crate`) to contain `targets: HashMap<String, Crate>`
- [ ] Rename `SymbolGraph.crates` to `SymbolGraph.packages`
- [ ] Update proptest strategies for new structure

### Phase 2: Orchestrator - Mapping File (`cargo-tarjanize`)

**Files:** `crates/cargo-tarjanize/src/orchestrator.rs`

- [ ] After reading cargo metadata, write mapping file:
  - Iterate all packages and their targets
  - Map each target's `src_path` → `{package, target_key}`
  - Write to `{output_dir}/target_mapping.json`
- [ ] Add `ENV_MAPPING_FILE` environment variable pointing to the mapping file

### Phase 3: Driver Changes (`cargo-tarjanize`)

**Files:** `crates/cargo-tarjanize/src/driver.rs`

- [ ] Read mapping file from `ENV_MAPPING_FILE`
- [ ] Extract source file path from rustc args (the non-flag .rs argument)
- [ ] Look up in mapping to get package and target_key
- [ ] If not found in mapping, pass through to rustc without extraction (external crate)
- [ ] Update output path: `{output_dir}/{package}/{target_key}.json`
- [ ] Update `CrateResult` to include package and target_key

### Phase 4: Extraction Changes (`cargo-tarjanize`)

**Files:** `crates/cargo-tarjanize/src/extract.rs`

- [ ] Add `is_test_code()` function to detect test-specific code:
  - Check if the symbol is inside a `#[cfg(test)]` module
  - This is sufficient; `#[test]` functions are typically inside `#[cfg(test)]` anyway
- [ ] Add parameter to `extract_crate()` for target kind
- [ ] When extracting:
  - Lib target: exclude symbols inside `#[cfg(test)]`
  - Test target: only include symbols inside `#[cfg(test)]`
  - Bin: include all symbols (separate compilation unit)

### Phase 5: Orchestrator - Aggregation (`cargo-tarjanize`)

**Files:** `crates/cargo-tarjanize/src/orchestrator.rs`

- [ ] Build `crate_name → (package, target)` mapping from cargo metadata
- [ ] Remove `merge_crate()` and `merge_module()` functions (no longer needed)
- [ ] Update `aggregate_results()`:
  - Walk `{output_dir}/{package}/{target_key}.json` files
  - Directory structure tells us package/target
  - Build `Package.targets` structure directly (no merging)
- [ ] Add path transformation:
  - For each symbol's dependencies, transform `crate_name::path` → `[package/target]::path`
  - Use the crate_name → package mapping
  - Assume `lib` target for cross-package dependencies
- [ ] Update `populate_dependencies()`:
  - Lib target: normal deps → `"{dep-package}/lib"`
  - Test target: normal + dev deps + `"{self-package}/lib"`
  - Bin target: normal deps + `"{self-package}/lib"`

### Phase 6: Condense Changes (`tarjanize-condense`)

**Files:** `crates/tarjanize-condense/src/scc.rs`, `lib.rs`

- [ ] Update to iterate `packages` instead of `crates`
- [ ] Update `SymbolIndex` to handle new structure:
  - Iterate all targets in all packages
  - Symbol paths now include `[package/target]` prefix
- [ ] Update path parsing/construction throughout
- [ ] SCC computation logic remains the same (no changes needed)

### Phase 7: Cost Changes (`tarjanize-cost`)

**Files:** `crates/tarjanize-cost/src/lib.rs`

This phase enables the key goal: accurate critical path computation without dev-dependency cycle issues (see `docs/cost-model-validation.md` TODO B.2).

- [ ] Update `critical_path()` to handle new structure:
  - Build dependency graph from all targets across all packages
  - Graph is now acyclic: `dep-package/lib → my-package/lib → my-package/test`
  - No more "apparent cycles" from dev-dependencies
  - Compute critical path across targets
- [ ] Update cost computation helpers
- [ ] Update tests
- [ ] Re-validate against real build times (TODO B.3 in cost-model-validation.md)

### Phase 8: Condense Target Type Preservation (`tarjanize-condense`)

**Files:** `crates/tarjanize-condense/src/scc.rs`

**Problem:** Currently, the condense phase outputs all synthetic packages with `lib` as the target type. This loses binary/test identity and breaks critical path analysis (binaries should be leaf nodes).

**Key insight:** Symbols from different target types (lib, bin, test) cannot form cycles:
- Lib code cannot depend on bin/test code
- Bin/test code depends on lib code (one direction only)
- Therefore, symbols from different target types will always end up in different synthetic packages

**Solution:** Extract the target type from symbols' original paths and preserve it in output.

1. Symbols have paths like `[pkg/lib]::foo`, `[pkg/bin/main]::bar`, `[pkg/test]::test_x`
2. All symbols in a synthetic package will share the same original target type
3. When outputting, extract target type from any symbol's path and use it

**Implementation:**

- [ ] In `compute_condensed_graph()`, track original target type per symbol:
  - Parse `[package/TARGET]::...` to extract TARGET
  - Store alongside symbol in index or compute when building output

- [ ] When building output packages, determine target type:
  - Look at symbols in the synthetic package
  - Extract target type from first symbol's path (all will match)
  - Use that as the target key instead of hardcoded `"lib"`

- [ ] Handle edge cases:
  - `bin/X` targets: preserve the binary name (`X`)
  - `example/X`, `bench/X`: same treatment as `bin/X`
  - Mixed target types in one package: should not happen (assert/warn if it does)

**Example transformation:**

Before (current):
```
cargo-tarjanize-42/lib  (contains lib symbols)
cargo-tarjanize-43/lib  (contains test symbols)
cargo-tarjanize-44/lib  (contains bin/main symbols)
```

After (fixed):
```
cargo-tarjanize-42/lib
cargo-tarjanize-43/test
cargo-tarjanize-44/bin/cargo_tarjanize
```

**Note on extracted code:** Helper code from a binary that gets split out becomes a lib (it's library code once extracted). Only the package containing `main()` remains a binary. This happens naturally because `main()` has no incoming dependencies.

## Testing Strategy

- [ ] **Unit tests**: Update all existing tests to use new schema
- [ ] **Integration tests**: Verify real workspace extraction produces correct targets
- [ ] **Roundtrip tests**: Ensure JSON serialization/deserialization works
- [ ] **Path transformation tests**: Verify crate_name → package/target transformation
- [ ] **Validation**: Run on omicron workspace and verify:
  - Lib and test targets are separate
  - Dependencies are correct
  - No artificial cycles from dev-dependencies

## Migration

- **Backward compatibility**: Not maintained (acceptable per discussion)
- Existing `symbol_graph.json` files will not parse with new schema
- Users must re-run `cargo tarjanize` to generate new format

## Open Questions

None remaining - all resolved in discussion.

## Future Work

- May want to add target-level metadata (source files, etc.)
