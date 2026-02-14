# Codebase Cleanup Design

**Goal:** Restore code quality across Rust and TypeScript by improving comments,
structure, error handling, linting, tests, logging, and docs, without changing
JSON schemas or file formats.

## Context
The codebase is functional and tests pass, but the implementation is messy
after low-supervision "vibe coding." The cleanup emphasizes maintainability
and correctness, with special focus on comments that justify the existence
of code (what it does and why it is needed).

## Constraints and Non-Goals
- No changes to JSON schema or file formats.
- Public APIs may change; REST API changes are allowed if the TS client is
  updated in lockstep.
- Do not reduce existing test coverage or change test semantics. Tests encode
  invariants and must remain valid.
- Test cleanups must be behavior-preserving only (e.g., refactors for clarity,
  fixture organization, or duplication removal). No weakening of assertions,
  skipping, or deletion of coverage.
- No `#[expect]` in production code. Allowed only in test fixtures when
  unavoidable, always with a reason.
- Avoid noisy or redundant comments. Comments must explain what and why.

## Change Management (Jujutsu)
- This repo uses `jj` for changes and descriptions.
- Use `jj new` and `jj describe` for each chunk of work.
- Keep changes small and reviewable, and keep the working copy always green.

## Skill Requirements
- Use `rust-development` for all Rust review and refactor work.
- Use `using-superpowers` and any other relevant skills as required by the
  skill system before responding or acting.

## Primary Quality Standard: Comments
- Applies to Rust and TypeScript.
- Every function, struct, enum, class, and non-trivial block must include a
  comment that explains:
  - What it does.
  - Why it is necessary (context, constraints, what breaks if it is removed).
- Comments should be concise, avoid restating obvious code, and focus on
  rationale and invariants.

## Smell Checklist (Applied During Review)
- Bloaters: long functions, large modules, long parameter lists, data clumps, primitive obsession.
- Change preventers: divergent change, shotgun surgery, parallel inheritance patterns.
- Misapplied abstractions: switch-as-type-system, unused generality, temporary fields.
- Coupling issues: feature envy, inappropriate intimacy, message chains, middleman.
- Dispensables: duplication, dead code, speculative generality, lazy modules, data-only types.
- Constant-parameter usage: parameters that are always passed the same value.
- Lint suppressions: any `#[expect]` in production code is a smell.
- "Weird code": collapse or refactor ad-hoc or inconsistent helpers into
  clearer, shared implementations.

## Checklist Discipline (Meticulous, Every Crate)
- For every crate sweep, explicitly and meticulously walk:
  - `docs/cleanup-checklist.md`
  - This document's Smell Checklist and Rust Guideline Smells
- Do not skip items even if no changes are needed; record "checked/no findings."
- Record completion notes per checklist section in `docs/cleanup-ledger.md`.
## Rust Guideline Smells (from `rust-development`)
Treat these as cleanup signals and align code with the guideline intent:
- Documentation gaps on public items: missing summary, examples, or Errors/Panics/Safety.
- Missing module-level docs on public modules.
- Re-exports without `#[doc(inline)]`.
- First doc sentences that are too long or unclear.
- Public types missing `Debug` / `Display` as appropriate.
- Weasel-word type names (Service/Manager/Factory).
- Hardcoded values without documented rationale.
- Leaking external types in public APIs without justification.
- Non-Send public types or futures where Send is expected.
- Unnecessary `unsafe`, or `unsafe` without explicit safety docs.
- Panics used for recoverable conditions.
- Overly nested abstractions or visible smart pointers in public APIs.
- Overuse of generics where concrete types would suffice.
- Large crates that should be split into smaller, cohesive units.

## Logging and Observability
- Remove printf-style debugging and stale info logs.
- Keep logs only when they provide durable operational value.
- Use structured logging with named fields.
- When retaining a log, add a brief comment explaining why it is useful.

## Linting and Tooling
- Rust: `cargo fmt` and `cargo clippy --all-targets` must remain clean with
  no production `#[expect]`.
- TS: introduce or tighten ESLint + TypeScript-ESLint rules and add lint scripts.
- Prefer best-practice rules aligned with the existing codebase style.

## Testing (Coverage and Semantics)
- Preserve all existing tests and their behavioral intent.
- If a test must be rewritten for cleanup, ensure it asserts the same behavior
  with no reduction in coverage or strictness.
- Any test change should be explainable as a refactor, not a behavior change.

## Documentation
- Update docs as part of each crate sweep while context is fresh.
- Correct outdated or inaccurate docs that no longer match implementation.
- Keep docs aligned with code changes to avoid drift.

## Process Overview (Staged, Always-Green)
Each crate sweep follows the same checklist, in this order:
1. Inventory (Serena-based map of symbols, TODO clusters, duplication
   candidates, and panic/expect hotspots).
2. Comment-first pass (what + why comments on all items).
3. Structural cleanup (remove dead code, split god modules, consolidate
   duplication, fix constant-parameter usage).
4. Error handling and API hygiene (remove inappropriate panics/expect,
   improve errors).
5. Logging cleanup (reduce noise, keep structured logs only where justified).
6. Lint and tests (clippy/fmt, TS lint, targeted tests).
7. Docs update for touched areas.
8. Ledger update for the crate.

## Serena MCP Usage
- Use symbol-level queries to map large modules, public APIs, and TODO clusters.
- Use pattern search to locate duplication, always-constant parameters, and
  panic/expect usage.
- Avoid full-file reads unless needed; operate at symbol granularity first.

## Phasing and Order
1. `tarjanize-schemas`
2. `tarjanize-schedule`
3. `cargo-tarjanize`
4. `tarjanize-condense` + `tarjanize-viz` (paired sweep for shared algorithms)
5. `tarjanize-cost` + `tarjanize-magsac`
6. `tarjanize` CLI

## Shared Algorithm Handling
For overlapping logic between `tarjanize-condense` and `tarjanize-viz`:
- Prefer extracting shared helpers into a common module (likely `tarjanize-schedule` or a small new crate).
- If extraction is not justified, keep mirrored implementations with aligned tests to prevent drift.

## Error Handling and API Expectations
- Panics are reserved for programming errors or invariant violations.
- When a panic or expect is truly warranted, include a comment stating the
  invariant and why failure is unrecoverable.
- Public APIs may be reshaped to simplify call sites, remove duplication, or
  eliminate always-constant parameters, as long as behavior and tests remain
  unchanged and the TS client is updated for REST changes.

## Risks and Mitigations
- Risk: large refactors break behavior. Mitigation: staged sweeps, tests per crate, preserve invariants.
- Risk: comment churn becomes noisy. Mitigation: enforce what/why, avoid obvious restatements.
- Risk: logging regression removes useful signals. Mitigation: require justification on retained logs.

## Acceptance Criteria
- Comment standard applied across all Rust and TS code.
- No production `#[expect]`; clippy clean; TS lint clean.
- Tests remain green with no semantic changes.
- Docs aligned with implementation.
- No JSON schema or file format changes.
