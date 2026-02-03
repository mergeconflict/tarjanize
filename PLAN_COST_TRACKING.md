# Implementation Plan: Compilation Cost Tracking

Implement detailed compilation cost tracking as specified in COMPILATION_COSTS.md, separating frontend and backend costs for accurate wall-clock time prediction.

## Summary

Replace the single `Symbol.cost` field with `frontend_cost_ms` and `backend_cost_ms`, add crate-level overhead tracking (`linking_ms`, `metadata_ms`), and implement the machinery to populate these from rustc's self-profile and mono-items output.

## Files to Modify

### Phase 1: Schema Changes (foundation)

**tarjanize-schemas/src/symbol_graph.rs**
- Add `Crate` struct with `linking_ms: f64`, `metadata_ms: f64`, `root: Module`
- Replace `Symbol.cost: f64` with `frontend_cost_ms: f64` and `backend_cost_ms: f64`
- Change `SymbolGraph.crates` from `HashMap<String, Module>` to `HashMap<String, Crate>`
- Update proptest strategies (`arb_symbol`, `arb_symbol_graph`)
- Add `is_zero()` helper for serde skip_serializing_if

### Phase 2: Fix Downstream Consumers

**tarjanize-condense/src/scc.rs**
- Update `SymbolIndex::build()` to access `crate_data.root` instead of `root_module` (line 40-42)
- Update `make_symbol()` helper: `(frontend, backend, deps)` instead of `(cost, deps)` (line 199-209)
- Update all test helper calls

**tarjanize-condense/src/lib.rs**
- Update `make_symbol()` helper in tests (line 109-119)
- Update test that creates `SymbolGraph { crates }` to use `Crate` wrapper (line 126-135)

### Phase 3: Enhanced Profile Parsing

**cargo-tarjanize/src/profile.rs** (major restructure)
- Add `EventCategory` enum: `Frontend`, `Backend`, `Overhead`
- Add `categorize_event(label: &str) -> EventCategory` function
- Add `CrateOverhead` struct: `{ linking_ms: f64, metadata_ms: f64 }`
- Restructure `ProfileData`:
  ```rust
  pub struct ProfileData {
      frontend_costs: HashMap<String, Duration>,  // by DefPath
      cgu_costs: HashMap<String, Duration>,       // by CGU name
      crate_overhead: HashMap<String, CrateOverhead>,
  }
  ```
- Update `aggregate_profile()` to categorize events and route to appropriate map
- Add new API: `get_frontend_cost_ms()`, `get_cgu_costs()`, `get_crate_overhead()`
- Update tests

### Phase 4: Mono-Items Parsing (new module)

**cargo-tarjanize/src/mono_items.rs** (new file)
- `MonoItemsMap` struct: `cgu_to_items: HashMap<String, Vec<String>>`
- `parse(reader, crate_name) -> MonoItemsMap` - parse `-Zprint-mono-items` output
- `normalize_mono_item(path) -> Option<String>` - strip generics, closures, handle shims
- Regex for parsing: `MONO_ITEM (fn|static) (.+) @@ (.+)`
- Handle items in multiple CGUs (for inlining)

**cargo-tarjanize/src/main.rs**
- Add `mod mono_items;`

### Phase 5: Driver Changes

**cargo-tarjanize/src/driver.rs**
- Line 93: Change `-Zself-profile-events=default,args` to `default,llvm,args`
- Add new flag: `-Zprint-mono-items=yes` (line ~94)

### Phase 6: Orchestrator Integration

**cargo-tarjanize/src/orchestrator.rs**
- Change from `cmd.status()` to `cmd.output()` to capture stderr
- Parse mono-items from captured stderr
- Add `distribute_backend_costs(profile, mono_items) -> HashMap<String, f64>`
- Split `apply_profile_costs()` into `apply_frontend_costs()` and `apply_backend_costs()`
- Update `aggregate_results()`:
  - Create `HashMap<String, Crate>` instead of `HashMap<String, Module>`
  - Apply crate overhead from profile data
- Update `PartialResult` struct to work with new schema

**cargo-tarjanize/src/extract.rs**
- Update all symbol creation sites (lines 219, 249, 271, 295, 331, 351, 385) to use:
  ```rust
  frontend_cost_ms: 0.0,  // populated later by orchestrator
  backend_cost_ms: 0.0,
  ```

## Event Categorization Rules

```rust
// Backend events (parallel via CGUs)
const BACKEND_PREFIXES: &[&str] = &["LLVM", "codegen_module", "link_", "target_machine"];

// Overhead events (per-crate fixed costs)
const OVERHEAD_PREFIXES: &[&str] = &["incr_comp_", "self_profile", "metadata_", "copy_all_cgu"];

// Everything else is frontend (serial)
```

Special handling for crate overhead:
- `link_crate`, `link_binary`, `link_rlib` -> `CrateOverhead.linking_ms`
- `generate_crate_metadata` -> `CrateOverhead.metadata_ms`

## Data Flow

```
rustc (driver mode)
    │
    ├── .mm_profdata files ──► profile.rs
    │                              │
    │                              ├── frontend_costs (by DefPath)
    │                              ├── cgu_costs (by CGU name)
    │                              └── crate_overhead
    │
    └── stderr (mono items) ──► mono_items.rs
                                   │
                                   └── cgu_to_items mapping

orchestrator.rs
    │
    ├── distribute_backend_costs()
    │       Join cgu_costs with cgu_to_items
    │       Normalize mono items to symbol paths
    │       Distribute CGU cost equally across items
    │
    └── aggregate_results()
            Apply frontend_cost_ms from profile
            Apply backend_cost_ms from distribution
            Apply linking_ms, metadata_ms at crate level
```

## Wall-Clock Prediction Formula

For reference (downstream consumers):
```
wall_clock = frontend_time + backend_time + overhead

where:
  frontend_time = Σ symbol.frontend_cost_ms           (serial)
  backend_time  = max(module.backend_cost_ms())       (parallel)
  overhead      = crate.linking_ms + crate.metadata_ms
```

Module backend cost is derived: sum of all symbols' backend costs recursively.

## Implementation Order

1. **Schema** (tarjanize-schemas) - foundation
2. **Condense fixes** (tarjanize-condense) - unblock cargo check
3. **Profile restructure** (profile.rs) - categorized costs
4. **Mono-items module** (mono_items.rs) - CGU mapping
5. **Driver flags** (driver.rs) - enable data collection
6. **Orchestrator integration** (orchestrator.rs) - tie it together
7. **Extract updates** (extract.rs) - use new schema

## Verification

After implementation:

1. Run `cargo nextest run` - all tests should pass
2. Run `cargo clippy --all-targets` - no warnings
3. Run `cargo-tarjanize` on a test fixture and verify:
   - Output JSON has `frontend_cost_ms` and `backend_cost_ms` per symbol
   - Output JSON has `linking_ms` and `metadata_ms` per crate
   - Values are non-zero (profiling is working)
4. Run `tarjanize condense` on the output - should work with new schema
