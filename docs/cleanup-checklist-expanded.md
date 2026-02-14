# Expanded Code Smell Checklist

Expands every item from `docs/cleanup-checklist.md` and the smell categories
from `docs/plans/2026-02-13-codebase-cleanup-design.md` into concrete,
actionable instructions with what to look for, how to detect it, and how to
fix it.

Applies to **all production and test code** (Rust and TypeScript).

**Inviolable constraint:** Tests encode behavioral contracts. No cleanup may
lose the semantic information captured in existing tests. Tests may be
refactored (e.g., extracted into helpers, restructured for clarity) or updated
to reflect type changes (e.g., a newtype replacing a raw `String`), but the
behavioral coverage they represent must be preserved. If a smell fix changes a
public API, update the tests to exercise the same contract through the new API.
If a test seems redundant, verify that it doesn't cover a distinct edge case
before removing it. When in doubt, keep the test.

---

## A. Bloaters

Things that have grown too large to work with effectively.

### A1. Long Functions

**What to look for:** The goal is readability: can a person read the function,
understand it, and confirm on inspection that it obviously does what it's
meant to do? Line count is one indicator, but complexity is often better —
deep nesting, many local variables in scope, complex control flow graphs,
or operating at multiple levels of abstraction within a single function.
Functions with `#[expect(clippy::too_many_lines)]` are pre-flagged, but
short functions can also be hard to follow if they're dense.

**How to detect:**
- `grep -n '#\[expect.*too_many_lines' crates/` for pre-flagged Rust functions
- Use Serena `get_symbols_overview` to find large functions
- For TS: inspect functions manually in larger files (`renderer.ts`, `tree.ts`)
- Beyond length, look for: deep nesting (3+ levels), many local variables,
  interleaved concerns, or functions you can't summarize in one sentence

**How to fix:** Extract coherent blocks into named helper functions. Each
function should do one thing at one level of abstraction.

**Known findings:** `scc.rs::condense_and_partition` has an `#[expect]` for
this. `scc.rs::build_output_graph` also has one.

### A2. Large Modules / God Files

**What to look for:** Single files over ~500 lines that combine unrelated
concerns. A file is a "god module" if you'd describe it with "and" more than
once ("it does X and Y and Z").

**How to detect:** A high line count is a fair initial indicator, but
cohesiveness — grouping of related concerns — is the real measure. Ask:
"Would I describe this file's purpose with 'and' more than once?" Use
`wc -l` on `.rs` and `.ts` source files as a starting point, then evaluate
whether the code in large files is cohesive or just colocated.

**How to fix:** Extract cohesive subsets into separate modules/files. Use
`mod` in Rust or separate `.ts` files for frontend.

### A3. Long Parameter Lists

**What to look for:** Functions with 5+ parameters. Especially functions where
several parameters are always passed together (data clumps).

**How to detect:** Use Serena `find_symbol` with `include_body=True` to
inspect function signatures — this handles multi-line parameter lists
correctly, unlike grep which breaks on line-wrapped signatures.

**How to fix:** Group related parameters into a struct. If 3+ parameters always
travel together, that's a data clump — make it a type.

### A4. Primitive Obsession

**What to look for:** Using `String` where a newtype or enum would add type
safety. Common symptoms:
- Target keys as `String` (e.g., `"pkg/lib"`) passed around without validation
- Symbol paths as raw `String` instead of a `SymbolPath` newtype
- Duration stored as `f64` milliseconds instead of `Duration`

**How to detect:** Look at function signatures that accept/return `String` or
`f64` for domain concepts. Check if the same parsing/formatting logic (e.g.,
`split_once('/')`) is repeated at multiple call sites.

**How to fix:** Introduce newtypes with `From`/`Display` impls. Only worth doing
if the same string format is parsed/validated in 3+ places. Also consider
whether `Duration` values actually represent durations or instants (points
in time) — if some `Duration` fields semantically represent timestamps or
offsets from an epoch, a newtype like `Instant` or `Timestamp` makes the
distinction explicit and prevents accidentally mixing the two.

### A5. Data Clumps

**What to look for:** Groups of variables that always appear together. Classic
example: `(crate_name, package_name, target_key)` passed as 3 separate `&str`
params to many functions.

**How to detect:** Look for functions that share 3+ parameters with the same
names. The driver.rs span-helper functions are a known cluster.

**How to fix:** Bundle into a struct. Only if it appears in 3+ function
signatures. Also look for tuples `(A, B, C)` used as return types or local
variables that would be clearer as structs with named fields — named fields
make both the construction site and the access site self-documenting.

---

## B. Dispensables

Things that add no value and can be removed or consolidated.

### B1. Duplicate Code

**What to look for:** Identical or near-identical functions in multiple crates.
Copy-pasted logic that must be kept in sync manually.

**How to detect:** Grep for function names that appear in multiple files.
Search memory `code-smell-audit-findings.md` for known findings.

**Known findings:**
- `duration_to_ms_f64()` — 7 copies across 6 files (identical one-liner)
- `count_symbols()` — 3 copies (2 production + 1 renamed variant)
- `collect_frontend_cost()` — 2 copies (schedule + cost crates)

**How to fix:** Move shared helpers to `tarjanize-schemas` (it's already a
dependency of all crates that use these functions). Re-export and delete copies.

### B2. Dead Code

**What to look for:** Functions, structs, imports, or feature-gated code that
is never called or compiled.

**How to detect:**
- `cargo udeps` (if available) for unused dependencies
- `#[allow(dead_code)]` or `#[expect(dead_code)]` suppressions
- Grep for `pub` functions and check if they have callers outside tests

**How to fix:** Delete it. Version control has the history.

### B3. Speculative Generality

**What to look for:** Generics, trait abstractions, or configuration options
that have exactly one concrete usage. Code that was written "in case we need
it later."

**How to detect:** Look for generic type parameters with only one concrete
instantiation. Look for trait objects (`dyn Trait`) with only one implementor.
Look for config fields that are always set to the same value.

**How to fix:** Replace with the concrete type. Re-introduce generality when
a second use case actually arrives.

### B4. Lazy Modules / Thin Wrappers

**What to look for:** Modules that just re-export or trivially delegate to
another module. Files under ~20 lines that only contain `pub use` or a single
thin wrapper function.

**How to detect:** Look at small files (under 30 lines). Check if `data.rs`
or similar files just re-export types.

**How to fix:** There's nothing inherently wrong with small modules. Only
consider merging if (a) the module has exactly one dependent, (b) merging
wouldn't hurt cohesiveness of the dependent, and (c) the module doesn't
serve as a meaningful abstraction boundary. This is essentially what
`tarjanize-condense` helps evaluate for crate-level splits — apply the
same reasoning at the module level.

---

## C. Change Preventers

Patterns that make future modifications painful.

### C1. Shotgun Surgery

**What to look for:** A single logical change requires touching many files.
Example: adding a new field to `TargetTimings` requires changes in schemas,
cost, schedule, condense, and viz.

**How to detect:** Think about common change scenarios (adding a metric,
adding a symbol kind, adding a target type). Trace how many files each
change touches.

**How to fix:** Centralize the concept. If a type is the source of truth,
put the behavior next to the type.

**Note:** This smell is hard to detect mechanically and requires judgment
about how the codebase evolves over time. Flag potential instances for human
review rather than attempting autonomous fixes. Concrete cases (like the
`TargetTimings` example) can be identified by tracing all files that import
a type and checking if they all need updating for a single field addition.

### C2. Divergent Change

**What to look for:** A single module changes for many unrelated reasons.
Example: if `scc.rs` changes both when the SCC algorithm changes AND when
the cost model changes AND when the output format changes.

**How to detect:** Look at `jj log` for files that appear in many unrelated
commits.

**How to fix:** Split the module along responsibility boundaries.

---

## D. Couplers

Excessive dependencies between modules.

### D1. Feature Envy

**What to look for:** A function that uses more data/methods from another
module's types than from its own. Example: a function in `condense` that
mostly calls methods on `schedule` types.

**How to detect:** For each non-trivial function, check whether the types it
operates on mostly belong to its own crate or another crate.

**How to fix:** Move the function to the crate whose types it uses most, or
extract the shared concept into the schema crate.

### D2. Inappropriate Intimacy

**What to look for:** Modules that reach into another module's internal
structure. Example: code that manually walks `Module.submodules` and
`Module.symbols` instead of using a provided traversal method.

**How to detect:** Look for repeated patterns of manual module-tree traversal
across crates. The `count_symbols` and `collect_frontend_cost` duplications
are symptoms of this.

**How to fix:** Add traversal methods to the schema types themselves, then
use them everywhere.

---

## E. Misapplied Abstractions

### E1. Unused Generality

See B3 (Speculative Generality). Same thing, different name.

### E2. Temporary Fields

**What to look for:** Struct fields that are `Option<T>` not because the
value is semantically optional, but because it gets set at a different time
than construction. Fields that are `None` during one phase and `Some` during
another.

**How to detect:** Look for `Option` fields where every access does
`.unwrap()` or `.expect()`.

**How to fix:** Restructure into two types (builder -> built), or use a
separate struct for the phase where the field is always present. More
broadly, a type's inherent impl should reflect its intended use: if a type
is effectively immutable after construction, its methods should take `&self`,
not `&mut self`. The method signatures are the type's contract — they should
match how the type is meant to be used.

---

## F. Rust-Specific Smells

### F1. Production `#[expect]` / `#[allow]`

**What to look for:** Any lint suppression in production code (not test
fixtures). Per the design doc: "No `#[expect]` in production code."

**How to detect:** `grep -rn '#\[expect\|#\[allow' crates/*/src/`

**How to fix:** Fix the underlying lint issue. If suppression is truly
necessary, it must have a `reason = "..."` explaining why.

**Exception:** The `#[expect]` on `is_non_public` in `impl Visibility` is
intentional and should remain — we should not delete the `derive(Copy)` for
`Visibility` just to silence the lint.

### F2. Production `expect()` / `unwrap()`

**What to look for:** `.expect()` and `.unwrap()` calls in non-test code.
Per M-PANIC-IS-STOP, panics are for programming errors only.

**How to detect:** Search saved findings in
`code-smell-audit-findings.md` (27 production call sites found).

**Rules:**
1. **Never use `unwrap()`.** Always use `expect()` or `?`.
2. **`expect()` is acceptable when failure represents a bug** — i.e., the
   invariant should always hold and its violation means there's a programming
   error somewhere.
3. The `expect` message must state the invariant that was expected to hold,
   and suggest what the bug might be if violated. Example:
   `expect("schedule must have at least one target; empty graph should have been rejected earlier")`
4. For RwLock `.expect("poisoned: ...")` — this is idiomatic (poison = bug
   in another thread).
5. You won't always be able to tell statically whether the caller guarantees
   the condition (it can require global reasoning). When in doubt, prefer `?`
   to propagate the error.

**How to audit each site:** For each call, answer:
1. Is this a true invariant? (Failure = programming bug, not bad input.)
2. Does the `expect` message state the invariant and suggest what the bug
   might be?
3. Could a caller ever violate the invariant through normal (non-buggy)
   use? If yes -> convert to `?` with a proper error type.

### F3. Excessive `.clone()`

**What to look for:** Cloning `String`, `Vec`, `HashMap`, or other
heap-allocated types where a borrow would work. Especially in hot loops
or recursive traversals.

**How to detect:** `grep -n '\.clone()' crates/*/src/*.rs` — then check
if the owned value is actually needed or if `&` would suffice.

**How to fix:** Use view/borrow types instead of owned types where possible:
- `&str` instead of `String`
- `&[T]` instead of `Vec<T>`
- `&Path` instead of `PathBuf`
- `Cow<str>` or `Cow<[T]>` when you sometimes need owned and sometimes borrowed
- Restructure data flow to avoid the need for cloning altogether

### F4. Missing `Debug` on Public Types

**What to look for:** Per M-PUBLIC-DEBUG, all public types must implement
`Debug`.

**Known findings:** `CrateResult`, `ExtractionResult`, `AppState` are
missing `Debug`.

**How to fix:** Add `#[derive(Debug)]`. For types containing non-Debug
fields (like `RwLock`), implement Debug manually.

### F5. Missing Documentation on Public Items

**What to look for:** Per M-CANONICAL-DOCS:
- Every `pub fn` needs a `///` doc comment with summary
- Every `pub struct` / `pub enum` needs a `///` doc comment
- Functions that return `Result` should document `# Errors`
- Functions that panic should document `# Panics`
- `unsafe` functions must document `# Safety`

**How to detect:** `cargo doc --document-private-items 2>&1 | grep warning`
(missing_docs lint). Or enable `warn(missing_docs)` temporarily.

### F6. Missing Module-Level Documentation

**What to look for:** Per M-MODULE-DOCS, every public module needs `//!`
inner doc comments.

**How to detect:** Check each `lib.rs` and public `mod` file for `//!`
at the top.

### F7. Re-exports Without `#[doc(inline)]`

**What to look for:** Per M-DOC-INLINE, `pub use` re-exports should have
`#[doc(inline)]` so items appear naturally in docs.

**How to detect:** `grep -n 'pub use' crates/*/src/*.rs` and check for
missing `#[doc(inline)]`.

### F8. Magic Values

**What to look for:** Per M-DOCUMENTED-MAGIC, hardcoded numeric literals
(not 0 or 1) must have a comment explaining why that value was chosen.

**How to detect:** Grep for numeric literals in conditionals, thresholds,
and constants: `grep -n '[2-9][0-9]*\|0\.\|1\.' crates/*/src/*.rs`
(filter out line numbers, indices, and obvious cases).

### F9. Leaked External Types in Public APIs

**What to look for:** Per M-DONT-LEAK-TYPES, public function signatures
should prefer std types over external crate types. External types in public
APIs are acceptable only when they provide substantial ecosystem value
(e.g., `serde::Serialize`).

**How to detect:** Check public function signatures for types from external
crates (petgraph, indexmap, analyzeme, etc.).

### F10. Boolean Parameters

**What to look for:** Functions that take `bool` parameters where the call
site reads as `foo(true)` or `foo(false)` with no indication of what the
bool means. Particularly bad with 2+ bools.

**How to detect:** `grep -n 'fn.*bool' crates/*/src/*.rs` — check if call
sites are readable.

**How to fix:** Replace with a two-variant enum. Example:
`is_test: bool` -> `TargetKind::Lib | TargetKind::Test`

---

## G. TypeScript / Frontend Smells

Applies to the 8 TS files under `crates/tarjanize-viz/templates/`.

### G1. Long Functions

**What to look for:** Functions over ~40 lines. TS functions should be short
and focused.

**How to detect:** Read each file; most are small (<100 lines). Focus on
`renderer.ts` (399 lines) and `tree.ts` (329 lines).

### G2. Magic Numbers

**What to look for:** Hardcoded pixel values, colors, thresholds, or timing
values without named constants or comments.

**How to detect:** Look for numeric literals in `renderer.ts`, `sidebar.ts`,
`tree.ts`. Check if they're defined in `constants.ts` or inline.

**How to fix:** Move to `constants.ts` with a descriptive name, or add a
comment explaining the value.

### G3. Type Safety Gaps

**What to look for:**
- `any` type annotations (bypasses type checking entirely)
- Type assertions (`as Foo`) that could be replaced with type guards
- Missing return type annotations on exported functions
- Optional chaining (`?.`) used where the value should never be null

**How to detect:** ESLint with `@typescript-eslint/no-explicit-any` and
`@typescript-eslint/no-unsafe-*` rules. Grep for `as ` and `: any`.

### G4. DOM Manipulation Smells

**What to look for:** Since this is vanilla TS (no React):
- `innerHTML` assignments (XSS risk, destroys event listeners)
- `document.getElementById` without null checks
- String concatenation for building HTML instead of DOM API or templates
- Event listeners not cleaned up

**How to detect:** Grep for `innerHTML`, `getElementById`, string HTML
construction patterns.

### G5. Imperative Loops Where Functional Would Be Clearer

**What to look for:** `for` loops over arrays where `.map()`, `.filter()`,
or `.reduce()` would be more expressive.

**How to detect:** Manual inspection of loop bodies in larger files.

### G6. Duplicated Frontend Logic

**What to look for:** Similar formatting, calculation, or DOM construction
logic repeated across TS files.

**How to detect:** Compare functions across `sidebar.ts`, `tree.ts`,
`renderer.ts` for overlapping patterns.

### G7. Missing JSDoc on Exported Functions

**What to look for:** Exported functions without `/** */` doc comments.

**How to detect:** Check each `export function` for a preceding JSDoc block.

---

## H. Cross-Cutting Concerns

### H1. Inconsistent Error Handling

**What to look for:** Mix of `anyhow` and custom errors within the same
call chain. Error types that don't implement `Display` or `Error`.

### H2. Catch-All Match Arms

**What to look for:** `_ => {}` or `_ => unreachable!()` in match
expressions. These hide new variants added later.

**How to detect:** `grep -n '_ =>' crates/*/src/*.rs` — check if the
matched type is an enum that could grow.

### H3. Oddball Solutions

**What to look for:** Similar problems solved differently in different
places. Example: one crate walks the module tree imperatively, another
uses recursion with `map/sum`, another uses a different traversal order.

**How to detect:** Compare the implementations of duplicated functions
(see B1). Check if they use different idioms for the same task.

---

## Verification

After applying fixes from this checklist:
- `cargo fmt`
- `cargo clippy --all-targets` — 0 warnings
- `cargo nextest run` — all tests pass (no test removals without justification)
- `cargo test --doc` — doc tests pass
- `cargo llvm-cov nextest` — coverage has not decreased; any test changes must
  preserve the behavioral contracts of the original tests
- Grep for remaining `#[expect]` in production code — 0 hits (or all have reasons)
- ESLint clean on TS files
