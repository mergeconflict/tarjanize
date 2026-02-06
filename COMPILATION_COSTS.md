# Rustc Compilation Cost Analysis

> **Current status**: This document describes how rustc compilation works internally and remains a useful reference on rustc's frontend/backend phases, CGUs, self-profile data, and mono items. However, **tarjanize no longer tracks backend costs**. Validation showed that backend cost modeling (CGU→symbol attribution via mono items) is unreliable and irrelevant for crate splitting — frontend time is serial, determines rmeta readiness, and gates downstream crates, while backend time runs in parallel via CGUs and doesn't meaningfully affect the critical path. The current cost model uses only `frontend_wall_ms` per target. The "Schema Design for Cost Storage" and "Cost Modeling" sections below reflect the original design; see the plan file for what was actually removed.

This document describes how rustc compiles code, what profiling data is available, and how to store and use that data for accurate build time prediction.

## Problem Statement

**Goal**: Store compilation costs in the `SymbolGraph` such that we can predict the wall-clock time of compiling a crate with reasonable accuracy.

**Why this is hard**: Rustc's compilation has multiple phases with different parallelism characteristics:

- **Frontend** (parsing, type checking, borrow checking) — largely serial, costs attributable to individual symbols via DefPath
- **Backend** (LLVM optimization, codegen) — parallel across Codegen Units (CGUs), costs attributable only to CGUs, not symbols
- **Overhead** (linking, metadata, incremental) — per-crate fixed costs

The naive approach of summing symbol costs dramatically overestimates compile time because it ignores backend parallelism. A crate with 10 CGUs might have 1000ms of total LLVM work, but only 150ms of wall-clock backend time (the slowest CGU).

**What we need**:
1. Attribute frontend costs to symbols (for critical path analysis)
2. Preserve CGU-level backend costs (for accurate wall-clock prediction)
3. Record crate-level overhead (linking, metadata)
4. Enable prediction of compile time after crate splitting

## Why This Matters for Tarjanize

Tarjanize splits crates to improve build parallelism. To predict whether a split helps, we need accurate cost models.

**The key insight**: Crate compilation is NOT simply the sum of symbol costs. Rustc already parallelizes backend work across Codegen Units (CGUs). Splitting a crate into two only helps if:

1. The split creates inter-crate parallelism that exceeds the intra-crate CGU parallelism, OR
2. The crate is frontend-dominated (type-heavy), where CGU parallelism doesn't help

For code-heavy crates with good CGU distribution, splitting may provide minimal benefit or even hurt (due to linking overhead).

## Table of Contents

1. [Compilation Phases](#compilation-phases)
2. [Codegen Units (CGUs)](#codegen-units-cgus)
3. [Self-Profile Data](#self-profile-data)
4. [Print Mono Items](#print-mono-items)
5. [Joining the Data Sources](#joining-the-data-sources)
6. [Observed Ratios](#observed-ratios)
7. [Cost Modeling](#cost-modeling)
8. [Practical Commands](#practical-commands)
9. [Schema Design for Cost Storage](#schema-design-for-cost-storage)

---

## Compilation Phases

Rustc compilation proceeds through multiple phases with different parallelism characteristics:

### Frontend (Largely Serial)

The frontend performs parsing, type checking, and MIR generation. These phases are largely serial within a crate, though the experimental parallel frontend (`-Zthreads=N`) can parallelize some work.

| Event Label | Description |
|-------------|-------------|
| `expand_crate` | Macro expansion |
| `macro_expand_crate` | Procedural macro expansion |
| `resolver_for_lowering_raw` | Name resolution |
| `late_resolve_crate` | Late name resolution |
| `typeck` | Type checking (per-item) |
| `type_check_crate` | Type checking (whole crate) |
| `mir_borrowck` | Borrow checking (per-item) |
| `MIR_borrow_checking` | Borrow checking (whole crate) |
| `optimized_mir` | MIR optimization |
| `mir_built` | MIR construction |

**Key property**: Frontend events have **DefPath** in their `additional_data`, allowing direct attribution to symbols:
```
typeck
  additional_data: ["my_crate::module::Type::{{impl}}::method"]
  duration: 1.23ms
```

### DefPath Formats

DefPaths from self-profile use rustc's internal naming conventions:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `{{impl}}` | An impl block | `my_crate::Type::{{impl}}` |
| `{{impl}}[N]` | Nth impl block on same type | `my_crate::Type::{{impl}}[1]` |
| `{{closure}}` | A closure | `my_crate::foo::{{closure}}` |
| `{{closure}}[N]` | Nth closure in same scope | `my_crate::foo::{{closure}}[0]` |
| `_[N]` | Anonymous/derive-generated item | `my_crate::module::_[7]::{{impl}}` |

**Normalization for symbol matching**: To match DefPaths to our symbol graph:
1. Truncate after `{{impl}}` or `{{impl}}[N]` — impl methods aggregate to impl block
2. Truncate at first `{{closure}}` — closures aggregate to containing function
3. Filter out `std::`, `core::`, `alloc::` paths — external crate costs

Example normalizations:
```
my_crate::Type::{{impl}}::method       → my_crate::Type::{{impl}}
my_crate::foo::{{closure}}::{{closure}} → my_crate::foo
my_crate::module::_[7]::{{impl}}::deserialize → my_crate::module::_[7]::{{impl}}
```

### Backend/Codegen (Parallel via CGUs)

The backend compiles MIR to machine code via LLVM. This work is parallelized across **Codegen Units (CGUs)**.

| Event Label | Description |
|-------------|-------------|
| `cgu_partitioning` | Partitioning mono items into CGUs |
| `collect_and_partition_mono_items` | Monomorphization collection |
| `codegen_module` | Codegen for one CGU |
| `LLVM_module_optimize` | LLVM optimization passes |
| `LLVM_module_codegen` | LLVM code generation |
| `LLVM_module_codegen_emit_obj` | LLVM object file emission |
| `LLVM_passes` | LLVM pass manager |

**Key property**: Backend events have **CGU names** (not DefPaths) in their `additional_data`:
```
LLVM_module_codegen_emit_obj
  additional_data: ["my_crate.ab33bbc52732c0f1-cgu.01"]
  duration: 45.67ms
```

**Note**: The `codegen_module` event has **two** values in `additional_data`:
```
codegen_module
  additional_data: ["my_crate.ab33bbc52732c0f1-cgu.01", "1090"]
  duration: 4.61ms
```
The second value appears to be an instruction/item count for the CGU.

**Note**: The `cgu_instructions` event (with `event_kind: ArtifactSize`) provides instruction counts:
```
cgu_instructions
  event_kind: ArtifactSize
  additional_data: ["my_crate.ab33bbc52732c0f1-cgu.01"]
```

**Warning**: Don't confuse `codegen_fn_attrs` with backend events. Despite the name, it's a **query** about function attributes, not actual codegen work:
```
codegen_fn_attrs          ← This is a FRONTEND query!
  event_kind: Query
  additional_data: []
```

### Linking (Serial)

| Event Label | Description |
|-------------|-------------|
| `link_crate` | Crate linking |
| `link_binary` | Binary linking |
| `link_rlib` | Rlib creation |

---

## Codegen Units (CGUs)

### What Are CGUs?

A Codegen Unit is a chunk of code that LLVM compiles independently. Multiple CGUs can be compiled in parallel by separate LLVM instances.

### Partitioning Strategy

Rustc partitions monomorphized code into CGUs using this strategy:

1. **Two CGUs per source-level module**:
   - "Stable" CGU: non-generic code (changes less frequently)
   - "Volatile" CGU: monomorphized generic instances (changes when generic usage changes)

2. **Default limits**:
   - Release builds: up to 16 CGUs
   - Debug builds: up to 256 CGUs (more incremental granularity)

3. **Controllable via**: `-Ccodegen-units=N`

### CGU Naming

CGU names follow the pattern: `{crate_name}.{hash}-cgu.{index}`

Example: `tarjanize_schemas.ab33bbc52732c0f1-cgu.01`

- `tarjanize_schemas`: crate name
- `ab33bbc52732c0f1`: hash for determinism
- `cgu.01`: CGU index within the crate

### Parallelism Implications

- CGUs compile in parallel (up to `codegen-units` limit)
- The **slowest CGU is the bottleneck** for backend time
- More CGUs = better parallelism but potentially worse optimization (LLVM can't inline across CGU boundaries)
- Setting `codegen-units=1` gives best optimization but no backend parallelism

---

## Self-Profile Data

### Enabling Self-Profile

```bash
RUSTFLAGS="-Zself-profile=/path/to/output" cargo +nightly build
```

This produces `.mm_profdata` files in the specified directory. Files are named:
```
{crate_name}-{pid}.mm_profdata
```

For example: `tarjanize_schemas-0065639.mm_profdata`

These can be analyzed with tools from the [measureme](https://github.com/rust-lang/measureme) repository. When loading with `analyzeme`, pass the path **without** the `.mm_profdata` extension:

```rust
// Correct: pass the stem
ProfilingData::new(Path::new("/tmp/output/tarjanize_schemas-0065639"))

// Wrong: don't include the extension
ProfilingData::new(Path::new("/tmp/output/tarjanize_schemas-0065639.mm_profdata"))
```

### Event Categories

Control what's recorded with `-Zself-profile-events=...`:

| Category | What it records | In `default`? |
|----------|-----------------|---------------|
| `query-provider` | Each internal query invocation | Yes |
| `generic-activity` | Non-query compiler work | Yes |
| `incr-cache-load` | Incremental cache loading | Yes |
| `query-cache-hit` | Query cache hits | No |
| `query-blocked` | Query blocking on other threads | No |
| `llvm` | LLVM passes and codegen | No |
| `query-keys` | Query arguments (DefPaths) | No |
| `function-args` | Generic activity arguments | No |
| `args` | Alias for `query-keys` + `function-args` | No |

**Critical**: To get useful data for cost attribution, use:
```bash
RUSTFLAGS="-Zself-profile=/path -Zself-profile-events=default,llvm,args"
```

The `args` flag is essential. Here's what you get with and without it:

**Without `args`** (just `default,llvm`):
```
LLVM_module_codegen_emit_obj
  additional_data: []          ← empty!
  duration: 45.67ms
```

**With `args`** (`default,llvm,args`):
```
LLVM_module_codegen_emit_obj
  additional_data: ["tarjanize_schemas.ab33bbc52732c0f1-cgu.01"]  ← CGU name!
  duration: 45.67ms
```

Without `args`, you cannot attribute backend costs to CGUs or frontend costs to symbols.

### Event Structure

Each event in the profile has:

```rust
struct Event {
    event_kind: String,      // "Query", "GenericActivity", etc.
    label: String,           // "typeck", "LLVM_module_codegen_emit_obj", etc.
    additional_data: Vec<String>,  // DefPath or CGU name (if `args` enabled)
    duration: Option<Duration>,
    thread_id: u32,
}
```

### Parsing Profile Data

Use the `analyzeme` crate:

```rust
use analyzeme::ProfilingData;
use std::path::Path;

let data = ProfilingData::new(Path::new("/path/to/crate-12345")).unwrap();

for event in data.iter_full() {
    println!("{}: {:?} ({:?})",
             event.label,
             event.additional_data,
             event.duration());
}
```

### Analysis Tools

Install from measureme repository:
```bash
cargo install --git https://github.com/rust-lang/measureme summarize
cargo install --git https://github.com/rust-lang/measureme crox
cargo install --git https://github.com/rust-lang/measureme flamegraph
```

- `summarize summarize /path/to/profile`: Human-readable summary
- `crox /path/to/profile`: Chrome trace format (view in chrome://tracing)
- `flamegraph /path/to/profile`: Flame graph SVG

---

## Print Mono Items

### What Are Mono Items?

A "mono item" is a monomorphized function or static that will become actual machine code. Generic functions like `fn foo<T>()` become multiple mono items: `foo::<i32>`, `foo::<String>`, etc.

### Enabling Print Mono Items

```bash
RUSTFLAGS="-Zprint-mono-items=yes" cargo +nightly build 2>&1 | tee mono_items.txt
```

### Output Format

```
MONO_ITEM fn my_crate::foo::bar @@ my_crate.ab33bbc52732c0f1-cgu.01[External]
MONO_ITEM fn my_crate::foo::baz @@ my_crate.ab33bbc52732c0f1-cgu.01[Internal]
MONO_ITEM static my_crate::CONSTANT @@ my_crate.ab33bbc52732c0f1-cgu.02[External]
```

Format: `MONO_ITEM {kind} {path} @@ {cgu_name}[{visibility}]`

- `kind`: `fn`, `static`, or `fn ... - shim(...)`
- `path`: Full path to the monomorphized item
- `cgu_name`: Which CGU this item is placed in
- `visibility`: `External` (exported) or `Internal` (local to CGU)

### Mono Item Path Formats

Mono item paths have several special formats you'll encounter:

**Generic instantiations** — the `<Type>` suffix shows concrete types:
```
MONO_ITEM fn std::option::Option::<my_crate::Foo>::map::<Bar, ...>
```

**Closures** — include source location:
```
MONO_ITEM fn std::option::Option::<T>::map::<U, {closure@src/lib.rs:131:59: 131:69}>
```

**Shims** — compiler-generated adapters:
```
MONO_ITEM fn <fn() as std::ops::FnOnce<()>>::call_once - shim(fn())
MONO_ITEM fn <MyType as Trait>::method - shim(vtable)
```

**Items in multiple CGUs** — some items appear in more than one CGU (for inlining):
```
MONO_ITEM fn my_crate::frequently_inlined @@ crate.hash-cgu.05[Internal] crate.hash-cgu.09[Internal]
```
Note the two CGU assignments on the same line.

### What This Tells Us

1. **Which functions exist after monomorphization**: A generic function may have 0, 1, or many mono items depending on how it's used.

2. **Which CGU each function is in**: This is the link between symbols and backend costs.

3. **Code distribution**: How evenly code is distributed across CGUs affects parallelism.

---

## Joining the Data Sources

### The Problem

- **Frontend costs** (from self-profile) have DefPath → directly attributable to symbols
- **Backend costs** (from self-profile) have CGU name → not directly attributable to symbols

### The Solution

Run both flags in the same compilation:

```bash
RUSTFLAGS="-Zself-profile=/tmp/profile -Zself-profile-events=default,llvm,args -Zprint-mono-items=yes" \
  cargo +nightly build 2>&1 | tee /tmp/mono_items.txt
```

### Join Process

1. **Parse self-profile** to build: `CGU name → total LLVM duration`

```rust
let mut cgu_times: HashMap<String, Duration> = HashMap::new();
for event in data.iter_full() {
    if event.label.starts_with("LLVM") {
        if let (Some(cgu), Some(dur)) = (event.additional_data.first(), event.duration()) {
            *cgu_times.entry(cgu.to_string()).or_default() += dur;
        }
    }
}
```

2. **Parse mono items** to build: `CGU name → [mono item paths]`

```bash
grep "MONO_ITEM" mono_items.txt | \
  sed 's/MONO_ITEM \(fn\|static\) \(.*\) @@ \(.*\)\[.*/\3 \2/' | \
  sort
```

3. **Join on CGU name**: Now you know which mono items contributed to each CGU's LLVM time.

4. **Distribute costs**: Allocate each CGU's time across its mono items. Options:
   - Equal distribution: `item_cost = cgu_cost / num_items`
   - Weighted by code size (requires additional data)
   - Weighted by MIR statement count

### Complications

1. **Mono items ≠ symbols**: Generic `fn foo<T>()` becomes multiple mono items. You need to aggregate back to the original symbol.

2. **Same build required**: CGU names include a hash that changes between builds. Both outputs must come from the same compilation.

3. **Closures and shims**: Mono items include closures (`{closure@file.rs:123:45}`) and shims (`fn ... - shim(...)`) that need special handling.

---

## Observed Ratios

The frontend/backend ratio varies dramatically by crate characteristics:

| Crate Type | Example | Frontend | Backend | Overhead |
|------------|---------|----------|---------|----------|
| Type-heavy library | serde_core | 88% | 2.5% | 10% |
| Code-heavy library | syn | 67% | 30% | 4% |
| Small with derives | tarjanize-schemas | 40% | 47% | 13% |

**Key insights**:

1. **Type-heavy crates** (lots of structs, traits, generics, derive macros) spend most time in frontend. Backend is minimal because there's little actual code to generate.

2. **Code-heavy crates** (lots of function bodies, algorithms) spend significant time in backend. LLVM has real work to do.

3. **Small crates with derives** can be backend-dominated because derive macros generate significant code from small type definitions.

### Example: Real CGU Breakdown

For `tarjanize-schemas` (release build, 10 CGUs):

```
CGU LLVM times:
  388.65ms  cgu.01  ← bottleneck
  325.56ms  cgu.00
  191.92ms  cgu.02
  172.83ms  cgu.03
  131.03ms  cgu.04
  124.10ms  cgu.06
  123.79ms  cgu.07
  120.32ms  cgu.09
   79.29ms  cgu.08
   52.91ms  cgu.05

Total LLVM time: 1710.41ms (if serial)
Max CGU time:     388.65ms (actual with parallelism)
Speedup:          4.4x
```

The slowest CGU determines the backend time, regardless of how many cores are available.

---

## Cost Modeling

### Why This Matters

When analyzing build parallelism (e.g., for crate splitting), the cost model dramatically affects conclusions.

We tested on a large workspace (Omicron) and found:

| Configuration | Critical Path | Crates | Parallelism |
|--------------|---------------|--------|-------------|
| Original | 225,413 ms | 161 | 3.89x |
| After splitting (naive model) | 94,307 ms | 126,876 | 9.29x |

The 9.29x parallelism is **illusory**—it assumes each tiny crate compiles serially. In reality, rustc parallelizes work within crates via CGUs, so the actual speedup would be much less.

### Naive Model (Inaccurate)

```
Cost(crate) = Σ Cost(symbols)
```

**Problem**: Assumes all work is serial. Dramatically overestimates for crates with significant backend work, because it ignores CGU parallelism.

### Better Model

```
Cost(crate) = Frontend + max(CGU_costs) + Linking
```

Where:
- `Frontend` = sum of all frontend event durations
- `max(CGU_costs)` = duration of the slowest CGU (the bottleneck)
- `Linking` = sum of linking event durations

### Even Better Model (with Parallel Frontend)

```
Cost(crate) = Frontend / min(threads, frontend_parallelism) + max(CGU_costs) + Linking
```

The parallel frontend (`-Zthreads=N`) can reduce frontend time by ~40% with 8 threads (observed in Rust blog benchmarks).

### Attributing Backend Costs to Symbols

For critical path analysis when splitting crates, you need symbol-level costs:

1. **Frontend costs**: Use DefPath from self-profile directly.

2. **Backend costs**:
   - Join self-profile with mono-items to get CGU→symbols mapping
   - Distribute each CGU's cost to its symbols
   - Aggregate mono-item costs back to original symbols (for generics)

3. **Per-symbol cost**:
   ```
   symbol_cost = frontend_cost(symbol) + backend_cost(symbol)
   ```

4. **Crate cost from symbols**:
   ```
   crate_cost = Σ frontend_cost(symbols) + max(Σ backend_cost(symbols in CGU) for each CGU)
   ```

---

## Practical Commands

### Full Profiling Build

```bash
# Create output directory
mkdir -p /tmp/profile_output

# Build with all profiling flags
RUSTFLAGS="-Zself-profile=/tmp/profile_output \
           -Zself-profile-events=default,llvm,args \
           -Zprint-mono-items=yes" \
  cargo +nightly build --release 2>&1 | tee /tmp/mono_items.txt

# Analyze with summarize
summarize summarize /tmp/profile_output/my_crate-*
```

### Quick Frontend/Backend Split

```bash
summarize summarize /path/to/profile | head -30
```

Look for:
- Frontend: `typeck`, `mir_borrowck`, `expand_crate`, etc.
- Backend: `LLVM_*`, `codegen_module`
- Overhead: `incr_comp_*`, `metadata_*`

### Categorizing Events Programmatically

We found these prefix rules work well for categorization:

```rust
fn categorize_event(label: &str) -> Category {
    // Backend: LLVM and actual codegen work
    let backend_prefixes = ["LLVM", "codegen_module", "link_", "target_machine"];

    // Overhead: incremental compilation, metadata, profiling
    let overhead_prefixes = ["incr_comp_", "self_profile", "metadata_", "copy_all_cgu"];

    if backend_prefixes.iter().any(|p| label.starts_with(p)) {
        Category::Backend
    } else if overhead_prefixes.iter().any(|p| label.starts_with(p)) {
        Category::Overhead
    } else {
        Category::Frontend
    }
}
```

**Important**: `codegen_fn_attrs` starts with "codegen" but is NOT a backend event—it's a query. The prefix `codegen_module` specifically identifies actual codegen work.

### Extract CGU Timing

```rust
use analyzeme::ProfilingData;
use std::collections::HashMap;
use std::time::Duration;

fn cgu_times(profile_path: &str) -> HashMap<String, Duration> {
    let data = ProfilingData::new(profile_path.as_ref()).unwrap();
    let mut times = HashMap::new();

    for event in data.iter_full() {
        if event.label.starts_with("LLVM") {
            if let (Some(cgu), Some(dur)) =
                (event.additional_data.first(), event.duration())
            {
                *times.entry(cgu.to_string()).or_default() += dur;
            }
        }
    }
    times
}
```

### Parse Mono Items

```bash
# Extract CGU → mono item mapping
grep "MONO_ITEM" mono_items.txt | \
  grep "my_crate" | \
  awk '{
    # Extract CGU name (between @@ and [)
    match($0, /@@ ([^\[]+)/, cgu);
    # Extract item path (between MONO_ITEM and @@)
    match($0, /MONO_ITEM (fn|static) (.+) @@/, item);
    print cgu[1], item[2]
  }'
```

---

## Schema Design for Cost Storage

This section describes how to store compilation costs in the SymbolGraph to enable accurate wall-clock time prediction.

**Scope**: We model **clean builds only** (no incremental compilation). Incremental builds are too variable to predict reliably.

### Cost Attribution Summary

All costs can be attributed to either symbols or the crate level:

| Cost Type | Attribution | Storage |
|-----------|-------------|---------|
| Frontend (typeck, mir, borrow check) | Symbol (via DefPath) | `Symbol.frontend_cost_ms` |
| Backend (LLVM) | Symbol (via mono-items) | `Symbol.backend_cost_ms` |
| Linking | Crate | `CrateCosts.linking_ms` |
| Metadata generation | Crate | `CrateCosts.metadata_ms` |

**Key insight**: Backend costs ARE attributable to symbols. Even compiler-generated shims can be traced back:

| Shim Type | Attribute To |
|-----------|--------------|
| `<MyType as Trait>::method - shim(vtable)` | The impl block (`MyType::{{impl}}`) |
| `<fn() as FnOnce<()>>::call_once - shim(fn())` | The function being wrapped |
| `drop_in_place::<MyType>` | The type (`MyType`) |
| `clone - shim` | The type's Clone impl |

### The Wall-Clock Prediction Formula

```
wall_clock = frontend_time + backend_time + overhead

where:
  frontend_time = Σ symbol.frontend_cost_ms           # sum ALL symbols (serial)
  backend_time  = max(module.backend_cost_ms())       # max across modules (parallel!)
  overhead      = linking_ms + metadata_ms            # serial
```

**Critical insight**: Frontend and backend costs are aggregated differently:
- **Frontend**: Sum across ALL symbols. Every symbol's frontend cost adds to wall-clock time.
- **Backend**: Sum within each module, then take MAX across modules. Only the slowest module's backend cost affects wall-clock time—other modules compile in parallel.

This is why we need separate `frontend_cost_ms` and `backend_cost_ms` fields. A combined `cost_ms` field would make accurate prediction impossible.

**Example**: Crate with modules A and B

| Module | Frontend | Backend |
|--------|----------|---------|
| A | 50ms | 100ms |
| B | 30ms | 50ms |

```
frontend_time = 50 + 30 = 80ms      # sum all
backend_time  = max(100, 50) = 100ms # max across modules
wall_clock    = 80 + 100 = 180ms
```

If we moved 30ms of backend work from A to B:
```
backend_time = max(70, 80) = 80ms   # B is now bottleneck, but faster overall!
wall_clock   = 80 + 80 = 160ms      # 20ms improvement
```

The split between frontend/backend lets us model how code reorganization affects parallelism.

Module backend cost is derived (not stored):
```rust
impl Module {
    fn backend_cost_ms(&self) -> f64 {
        self.symbols.values().map(|s| s.backend_cost_ms).sum::<f64>()
            + self.submodules.values().map(|m| m.backend_cost_ms()).sum::<f64>()
    }
}
```

### Schema

#### Symbol

```rust
struct Symbol {
    // ... existing fields ...

    /// Frontend compilation cost in milliseconds.
    /// Includes: typeck, borrow checking, MIR optimization.
    /// Directly attributable via DefPath from self-profile.
    frontend_cost_ms: f64,

    /// Backend compilation cost in milliseconds.
    /// Distributed from CGU costs via mono-items mapping.
    /// Includes LLVM optimization and codegen.
    backend_cost_ms: f64,
}
```

**How to populate**:
- `frontend_cost_ms`: Sum durations for self-profile events with this symbol's DefPath
- `backend_cost_ms`: Distribute CGU costs to symbols via mono-items join (see below)

#### Module

No cost fields—backend cost is derived by summing symbols recursively.

#### Crate

Introduce a `Crate` struct to hold crate-level costs alongside the root module:

```rust
pub struct SymbolGraph {
    pub crates: HashMap<String, Crate>,  // Changed from HashMap<String, Module>
}

pub struct Crate {
    /// Time spent linking this crate (combining CGU outputs).
    pub linking_ms: f64,

    /// Time spent generating crate metadata.
    pub metadata_ms: f64,

    /// The root module containing all symbols.
    pub root: Module,
}
```

JSON format changes from:
```json
{ "crates": { "my_crate": { "symbols": {...} } } }
```

To:
```json
{ "crates": { "my_crate": { "linking_ms": 2.5, "metadata_ms": 7.2, "root": { "symbols": {...} } } } }
```

Linking and metadata are truly crate-level costs that don't decompose to symbols. They scale roughly with crate size but are paid once per crate.

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐
│  self-profile   │     │ -Zprint-mono-items│
│  (with args)    │     │                  │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│ DefPath → cost  │     │ mono item → CGU  │
│ CGU → LLVM cost │     │                  │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │ JOIN on CGU name
                     ▼
         ┌───────────────────────┐
         │ Symbol:               │
         │   frontend_cost_ms    │◄── from DefPath events
         │   backend_cost_ms     │◄── distributed from CGU
         │                       │
         │ Crate:                │
         │   linking_ms          │◄── from link_* events
         │   metadata_ms         │◄── from generate_crate_metadata
         │   root: Module        │
         └───────────────────────┘
```

### Backend Cost Distribution

To distribute CGU costs to symbols:

1. **Parse mono-items** to build `CGU → [mono item paths]`
2. **For each CGU** with cost `C` and `N` mono items:
   - Each mono item gets `C / N` (equal distribution)
3. **Normalize mono items to symbols**:
   - Strip generic parameters: `foo::<i32>` → `foo`
   - Strip closure suffixes: `foo::{{closure}}` → `foo`
   - Map shims to their source: `<T as Trait>::method - shim(vtable)` → `T::{{impl}}`
4. **Aggregate**: Sum costs for mono items that map to the same symbol

```rust
fn distribute_backend_costs(
    cgu_costs: &HashMap<String, f64>,
    cgu_to_mono_items: &HashMap<String, Vec<String>>,
) -> HashMap<String, f64> {
    let mut symbol_costs: HashMap<String, f64> = HashMap::new();

    for (cgu, cost) in cgu_costs {
        if let Some(items) = cgu_to_mono_items.get(cgu) {
            let per_item = cost / items.len() as f64;
            for item in items {
                let symbol = normalize_mono_item_to_symbol(item);
                *symbol_costs.entry(symbol).or_default() += per_item;
            }
        }
    }

    symbol_costs
}
```

### Cost Prediction Scenarios

#### Scenario 1: Predict Existing Crate

For a crate we've profiled:

```
frontend = Σ symbol.frontend_cost_ms
backend  = max(module.backend_cost_ms() for module in crate.root)
overhead = crate.linking_ms + crate.metadata_ms

wall_clock = frontend + backend + overhead
```

#### Scenario 2: Predict After Crate Split

When splitting crate A into A1 and A2:

**Frontend**: Sum `frontend_cost_ms` for symbols in each new crate.

**Backend**: Use module structure as proxy for CGU structure:
```
backend_time = max(module.backend_cost_ms() for module in new_crate)
```

This works because CGU partitioning is module-based (~2 CGUs per module).

**Overhead**: Estimate as `base_overhead + factor * symbol_count`. Splitting increases total overhead (two crates instead of one) but may reduce per-crate overhead.

### Decisions Made

1. **Where to store CrateCosts?**

   **Decision**: Introduce a `Crate` struct. Change `SymbolGraph.crates` from `HashMap<String, Module>` to `HashMap<String, Crate>`:
   ```rust
   pub struct Crate {
       pub linking_ms: f64,
       pub metadata_ms: f64,
       pub root: Module,
   }
   ```
   This gives crate-level costs a natural home and is more semantically accurate.

### Open Questions

1. **How to handle symbols in multiple CGUs?**

   Some mono items appear in multiple CGUs (for inlining). When distributing CGU costs to symbols, should we:
   - **Sum** all contributions? (symbol in CGU-01 at 10ms + CGU-02 at 5ms = 15ms)
   - **Take max**? (symbol cost = 10ms, the larger contribution)
   - **Average**? (symbol cost = 7.5ms)

   Sum seems most accurate for "total backend work caused by this symbol", but need to verify this makes sense for prediction.

2. **How to predict linking/metadata costs for split crates?**

   After condensation, we create new crates that weren't profiled. We need to estimate their overhead. Options:
   - Scale linearly from original crate based on backend cost (for linking) or public symbol count (for metadata)
   - Learn per-symbol factors from profiling data and apply to new crates
   - Use fixed base + scaling factor

   Sub-question: What exactly determines metadata cost? Public symbols only, or impls too? Is the relationship linear?

3. **Generic instantiation costs after splitting?**

   A generic function's backend cost depends on which concrete types it's instantiated with. After splitting:
   - The function might be instantiated with different types
   - Some instantiations might disappear, others might appear
   - Current approach doesn't track per-instantiation costs

   May need to revisit if predictions are inaccurate.

---

## References

- [Intro to rustc's self profiler](https://blog.rust-lang.org/inside-rust/2020/02/25/intro-rustc-self-profile/) - Inside Rust Blog
- [Parallel compilation - Rust Compiler Development Guide](https://rustc-dev-guide.rust-lang.org/parallel-rustc.html)
- [Monomorphization - Rust Compiler Development Guide](https://rustc-dev-guide.rust-lang.org/backend/monomorph.html)
- [measureme repository](https://github.com/rust-lang/measureme) - Profiling tools
- [Back-end parallelism in the Rust compiler](https://nnethercote.github.io/2023/07/11/back-end-parallelism-in-the-rust-compiler.html) - Nicholas Nethercote
- [Faster compilation with the parallel front-end](https://blog.rust-lang.org/2023/11/09/parallel-rustc/) - Rust Blog
