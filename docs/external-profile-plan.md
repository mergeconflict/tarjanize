# Two-Pass Profiling Plan

## Problem

Self-profile data captured during `cargo tarjanize` is inflated 10-15x because our extraction callbacks trigger compiler queries that get profiled. The profile doesn't distinguish between real compilation work and query overhead from extraction.

**Evidence:**
- Tokio: 62s model frontend vs 4.6s actual rmeta time (13x inflation)
- Omicron: varies 5x-15x by crate

## Solution

Run two passes internally, transparent to the user:

```bash
# User just runs:
cargo tarjanize -o symbols.json

# Internally, tarjanize runs:
# Pass 1: cargo build --all-targets with -Zself-profile (no extraction)
# Pass 2: cargo build --all-targets with RUSTC_WRAPPER for extraction (no profiling)
# Merge: Load profile from pass 1, symbols from pass 2
```

## Implementation

### Changes to `cargo-tarjanize`

**CLI changes** (`main.rs`)
```rust
/// Skip profiling pass (faster, but costs will be inflated by extraction overhead)
#[arg(long)]
no_profile: bool,
```

**Orchestrator changes** (`orchestrator.rs`)

Replace single build with two builds:

```rust
pub fn run(config: &Config) -> Result<SymbolGraph> {
    let profile_dir = tempdir()?;
    let extract_dir = tempdir()?;

    // Pass 1: Profile only (no extraction)
    run_profiled_build(config, &profile_dir)?;

    // Pass 2: Extract only (no profiling)
    let results = run_extraction_build(config, &extract_dir)?;

    // Merge: Apply profile costs to extracted symbols
    let profile_data = ProfileData::load_from_dir(&profile_dir)?;
    apply_costs(&mut results, &profile_data)?;

    Ok(results)
}
```

**Pass 1: Profiled build**
- Run `cargo build --all-targets` with:
  - `CARGO_INCREMENTAL=0`
  - `RUSTFLAGS=-Zself-profile=<profile_dir>`
- No `RUSTC_WRAPPER` (normal rustc, no extraction)
- Output: profile data in `profile_dir`

**Pass 2: Extraction build**
- Run `cargo build --all-targets` with:
  - `RUSTC_WRAPPER=cargo-tarjanize`
  - `CARGO_INCREMENTAL=0`
  - No `-Zself-profile` (or ignore the output)
- Output: extracted symbols via driver callbacks

**Driver changes** (`driver.rs`)
- Remove profile loading from driver (moved to orchestrator)
- Driver only does extraction, returns Module without costs
- Orchestrator applies costs after both passes complete

**Profile verification** (`orchestrator.rs` or new `merge.rs`)
- After merging, count symbols with/without profile data
- Log error for each symbol missing profile data
- Summary: "X of Y symbols (Z%) missing profile data"

### File changes

| File | Change |
|------|--------|
| `main.rs` | Add `--no-profile` CLI flag |
| `orchestrator.rs` | Two-pass build logic, merge profile with symbols, verification |
| `driver.rs` | Remove profile loading, extraction only |
| `profile.rs` | No changes needed |

### Workflow

**Current (single-pass, inflated costs):**
```
cargo tarjanize
  └─→ cargo build (profile + extract combined)
  └─→ symbols.json (inflated costs)
```

**New (two-pass, accurate costs):**
```
cargo tarjanize
  ├─→ Pass 1: cargo build (profile only, no RUSTC_WRAPPER)
  ├─→ Pass 2: cargo build (extract only, with RUSTC_WRAPPER)
  └─→ Merge profile + symbols
  └─→ symbols.json (accurate costs)
```

## Validation

After implementation:
1. Run on tokio, compare model costs to `--timings` data
2. Expected: inflation reduced from 13x to ~1-2x
3. Verify per-symbol cost distribution still shows expected skew (top 1% = 75% of cost)

## Trade-offs

**Pros:**
- Accurate costs without user intervention
- Same CLI, just works better
- Profile data uncontaminated by extraction overhead

**Cons:**
- Two full builds instead of one (2x build time)
- More complex orchestration logic
- Larger temp directory usage (two target dirs)

**Mitigation for build time:**
- Share target directory between passes
- External dependencies compiled in Pass 1 are reused in Pass 2
- Only workspace crates need recompilation (due to RUSTC_WRAPPER fingerprint change)
- Net overhead: ~1.1-1.3x instead of 2x for large workspaces with many external deps

## Scope

- ~100-150 lines of code changes
- Refactor orchestrator to two-pass
- Simplify driver (extraction only)
- No changes to profile parsing or cost model

## Decisions

1. **`--no-profile` flag**: Add this to skip Pass 1 and use inflated costs (useful for quick iteration during development).

2. **Partial failures**: Use existing strategy - if a crate/target fails in Pass 2, we don't get data for it but continue with others. Not a total failure.

3. **Profile verification**: Verify that extracted symbols have matching profile data. Log errors for symbols without profile data so users know if something is wrong.
