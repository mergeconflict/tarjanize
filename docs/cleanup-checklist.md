# Cleanup Checklist

This checklist applies to all Rust and TypeScript code.

> **Expanded version:** See [cleanup-checklist-expanded.md](cleanup-checklist-expanded.md)
> for concrete detection methods, fix strategies, and known findings for each
> smell category.

## Comments (What + Why)
- Every function, struct, enum, class, and non-trivial block has a comment.
- Each comment explains what the code does and why it exists.
- Avoid redundant comments that restate obvious code.

## Smell Sweep
- Eliminate duplicate code and dead code.
- Remove always-constant parameters or inline them.
- Split god modules and consolidate ad-hoc helpers.
- Replace unclear abstractions with simpler, explicit ones.

## Rust Guideline Smells
- Add missing docs on public items and modules.
- Ensure public types implement Debug and Display where appropriate.
- Avoid weasel-word type names (Service/Manager/Factory).
- Document magic values and invariants.
- Avoid unsafe or explain it with Safety docs.
- Avoid panics for recoverable errors.

## Logging
- Remove stale debug/printf logs.
- Keep only structured logs with durable operational value.

## Lint and Tests
- `cargo fmt` and `cargo clippy --all-targets` are clean.
- ESLint is clean for TS.
- Do not change test semantics or reduce coverage.

## Docs
- Update any doc that becomes stale while touching code.
