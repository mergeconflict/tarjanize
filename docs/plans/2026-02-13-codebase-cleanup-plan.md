# Codebase Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute the staged cleanup across Rust and TypeScript with comment-first quality, without changing JSON schemas or test semantics.

**Architecture:** Follow a crate-by-crate sweep with a fixed checklist: inventory, comment pass, structural cleanup, error/logging hygiene, lint/tests, doc updates, ledger update. Keep the working copy always green and use `jj new` + `jj describe` per chunk.

**Tech Stack:** Rust, Cargo, clippy, rustfmt, TypeScript, ESLint, Vitest, Axum, PixiJS, Jujutsu (jj)

---

### Task 1: Create Cleanup Checklist Doc

**Files:**
- Create: `docs/cleanup-checklist.md`

**Step 1: Write the checklist doc**

```markdown
# Cleanup Checklist

This checklist applies to all Rust and TypeScript code.

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
```

**Step 2: Save file**

Run: no command.
Expected: file created.

**Step 3: Commit**

Run: `jj describe -m "docs: add cleanup checklist"`
Expected: description updated for current working copy.

### Task 2: Create Cleanup Ledger Doc

**Files:**
- Create: `docs/cleanup-ledger.md`

**Step 1: Write the ledger doc**

```markdown
# Cleanup Ledger

Status legend: Not started, In progress, Done

| Crate | Comments | Structure | Error/Logging | Lint/Tests | Docs | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| tarjanize-schemas | Not started | Not started | Not started | Not started | Not started | |
| tarjanize-schedule | Not started | Not started | Not started | Not started | Not started | |
| cargo-tarjanize | Not started | Not started | Not started | Not started | Not started | |
| tarjanize-condense | Not started | Not started | Not started | Not started | Not started | |
| tarjanize-viz | Not started | Not started | Not started | Not started | Not started | |
| tarjanize-cost | Not started | Not started | Not started | Not started | Not started | |
| tarjanize-magsac | Not started | Not started | Not started | Not started | Not started | |
| tarjanize | Not started | Not started | Not started | Not started | Not started | |
```

**Step 2: Save file**

Run: no command.
Expected: file created.

**Step 3: Commit**

Run: `jj describe -m "docs: add cleanup ledger"`
Expected: description updated for current working copy.

### Task 3: Add TypeScript Linting

**Files:**
- Modify: `package.json`
- Create: `eslint.config.js`

**Step 1: Add ESLint dev dependencies and lint script**

```json
{
  "devDependencies": {
    "eslint": "^9.0.0",
    "@typescript-eslint/parser": "^8.0.0",
    "@typescript-eslint/eslint-plugin": "^8.0.0"
  },
  "scripts": {
    "lint": "eslint . --max-warnings=0"
  }
}
```

**Step 2: Add ESLint config**

```js
import js from "@eslint/js";
import tseslint from "@typescript-eslint/eslint-plugin";
import tsparser from "@typescript-eslint/parser";

export default [
  {
    ignores: [
      "node_modules/**",
      "build/**",
      "dist/**",
      "target/**",
      "**/*.min.js"
    ]
  },
  js.configs.recommended,
  {
    files: ["**/*.ts"],
    languageOptions: {
      parser: tsparser,
      parserOptions: {
        sourceType: "module",
        ecmaVersion: "latest",
        project: "./tsconfig.json"
      }
    },
    plugins: {
      "@typescript-eslint": tseslint
    },
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
      "@typescript-eslint/explicit-function-return-type": "off",
      "@typescript-eslint/no-explicit-any": "error",
      "no-console": "error"
    }
  }
];
```

**Step 3: Run lint**

Run: `npm run lint`
Expected: exit 0 with no errors.

**Step 4: Commit**

Run: `jj describe -m "chore: add ESLint for TypeScript"`
Expected: description updated for current working copy.

### Task 4: tarjanize-schemas Inventory

**Files:**
- Modify: `docs/cleanup-ledger.md`

**Step 1: Inventory symbols and TODOs with Serena**

Run: use Serena `get_symbols_overview` for `crates/tarjanize-schemas/src/*.rs` and log hotspots.
Expected: quick map of public APIs and TODO clusters.

**Step 2: Update ledger notes**

Add a brief note in `docs/cleanup-ledger.md` for tarjanize-schemas.

**Step 3: Commit**

Run: `jj describe -m "docs: note tarjanize-schemas inventory"`
Expected: description updated for current working copy.

### Task 5: tarjanize-schemas Comment Pass (symbol_graph)

**Files:**
- Modify: `crates/tarjanize-schemas/src/symbol_graph.rs`

**Step 1: Add what+why comments to all public types and key helpers**

Example comment pattern:

```rust
/// Describes a compilation target and its symbol tree.
///
/// Why: The schedule and condense phases need a stable target-level view
/// that can be serialized without rustc internals. Without this type, the
/// pipeline cannot persist dependency structure between phases.
```

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment tarjanize-schemas symbol_graph"`
Expected: description updated for current working copy.

### Task 6: tarjanize-schemas Comment Pass (cost_model, testutil, lib)

**Files:**
- Modify: `crates/tarjanize-schemas/src/cost_model.rs`
- Modify: `crates/tarjanize-schemas/src/testutil.rs`
- Modify: `crates/tarjanize-schemas/src/lib.rs`

**Step 1: Add what+why comments and doc sections for public items**

Use the same what+why standard and ensure public module docs are present.

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment tarjanize-schemas modules"`
Expected: description updated for current working copy.

### Task 7: tarjanize-schemas Structural and Error Cleanup

**Files:**
- Modify: `crates/tarjanize-schemas/src/*.rs`
- Modify: `docs/cleanup-ledger.md`

**Step 1: Remove duplication and clarify APIs**

Collapse redundant helpers, eliminate always-constant parameters, and align
public APIs with rust-development guidelines.

**Step 2: Run clippy and tests**

Run: `cargo clippy -p tarjanize-schemas --all-targets`
Expected: exit 0 with no warnings.

Run: `cargo test -p tarjanize-schemas`
Expected: exit 0 with all tests passing.

**Step 3: Update docs if needed**

Update any doc files referencing schema behavior.

**Step 4: Update ledger and commit**

Run: `jj describe -m "refactor: clean tarjanize-schemas"`
Expected: description updated for current working copy.

### Task 8: tarjanize-schedule Inventory

**Files:**
- Modify: `docs/cleanup-ledger.md`

**Step 1: Inventory symbols and TODOs with Serena**

Run: use Serena `get_symbols_overview` for `crates/tarjanize-schedule/src/*.rs`.
Expected: map of hot modules and duplication candidates.

**Step 2: Update ledger notes**

Add a brief note in `docs/cleanup-ledger.md` for tarjanize-schedule.

**Step 3: Commit**

Run: `jj describe -m "docs: note tarjanize-schedule inventory"`
Expected: description updated for current working copy.

### Task 9: tarjanize-schedule Comment Pass (core)

**Files:**
- Modify: `crates/tarjanize-schedule/src/lib.rs`
- Modify: `crates/tarjanize-schedule/src/data.rs`

**Step 1: Add what+why comments on public types and modules**

Ensure module docs and public type docs match rust-development guidelines.

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment tarjanize-schedule core"`
Expected: description updated for current working copy.

### Task 10: tarjanize-schedule Comment Pass (graph/schedule)

**Files:**
- Modify: `crates/tarjanize-schedule/src/schedule.rs`
- Modify: `crates/tarjanize-schedule/src/target_graph.rs`

**Step 1: Add what+why comments for algorithms and invariants**

Document algorithmic steps and why each stage is needed.

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment tarjanize-schedule graph"`
Expected: description updated for current working copy.

### Task 11: tarjanize-schedule Comment Pass (split/recommend/heatmap/export)

**Files:**
- Modify: `crates/tarjanize-schedule/src/split.rs`
- Modify: `crates/tarjanize-schedule/src/recommend.rs`
- Modify: `crates/tarjanize-schedule/src/heatmap.rs`
- Modify: `crates/tarjanize-schedule/src/export.rs`

**Step 1: Add what+why comments and document cost distribution logic**

Ensure every non-trivial block explains rationale and constraints.

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment tarjanize-schedule helpers"`
Expected: description updated for current working copy.

### Task 12: tarjanize-schedule Structural and Error Cleanup

**Files:**
- Modify: `crates/tarjanize-schedule/src/*.rs`
- Modify: `docs/cleanup-ledger.md`

**Step 1: Consolidate shared logic and remove duplication**

Refactor cost distribution, shatter grouping, and graph wiring helpers where
appropriate to reduce drift.

**Step 2: Run clippy and tests**

Run: `cargo clippy -p tarjanize-schedule --all-targets`
Expected: exit 0 with no warnings.

Run: `cargo test -p tarjanize-schedule`
Expected: exit 0 with all tests passing.

**Step 3: Update docs if needed**

Update any schedule-related docs that are out of date.

**Step 4: Update ledger and commit**

Run: `jj describe -m "refactor: clean tarjanize-schedule"`
Expected: description updated for current working copy.

### Task 13: cargo-tarjanize Inventory

**Files:**
- Modify: `docs/cleanup-ledger.md`

**Step 1: Inventory symbols and TODOs with Serena**

Run: use Serena `get_symbols_overview` for `crates/cargo-tarjanize/src/*.rs`.
Expected: map of hot modules and duplication candidates.

**Step 2: Update ledger notes**

Add a brief note in `docs/cleanup-ledger.md` for cargo-tarjanize.

**Step 3: Commit**

Run: `jj describe -m "docs: note cargo-tarjanize inventory"`
Expected: description updated for current working copy.

### Task 14: cargo-tarjanize Comment Pass (main/orchestrator)

**Files:**
- Modify: `crates/cargo-tarjanize/src/main.rs`
- Modify: `crates/cargo-tarjanize/src/orchestrator.rs`

**Step 1: Add what+why comments for orchestration flow**

Ensure the cargo wrapper flow and invariants are explained.

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment cargo-tarjanize orchestrator"`
Expected: description updated for current working copy.

### Task 15: cargo-tarjanize Comment Pass (driver/extract)

**Files:**
- Modify: `crates/cargo-tarjanize/src/driver.rs`
- Modify: `crates/cargo-tarjanize/src/extract.rs`

**Step 1: Add what+why comments for rustc hooks and extraction rules**

Document why each rustc callback is needed and what data is captured.

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment cargo-tarjanize driver"`
Expected: description updated for current working copy.

### Task 16: cargo-tarjanize Comment Pass (profile)

**Files:**
- Modify: `crates/cargo-tarjanize/src/profile.rs`

**Step 1: Add what+why comments for profile aggregation**

Document how events are attributed and why overhead is handled.

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment cargo-tarjanize profile"`
Expected: description updated for current working copy.

### Task 17: cargo-tarjanize Structural and Error Cleanup

**Files:**
- Modify: `crates/cargo-tarjanize/src/*.rs`
- Modify: `docs/cleanup-ledger.md`

**Step 1: Remove duplication and clarify error paths**

Replace inappropriate panics/expect with explicit error handling where
recoverable. Document invariant panics.

**Step 2: Run clippy and tests**

Run: `cargo clippy -p cargo-tarjanize --all-targets`
Expected: exit 0 with no warnings.

Run: `cargo test -p cargo-tarjanize`
Expected: exit 0 with all tests passing.

**Step 3: Update docs if needed**

Update any cargo-tarjanize related docs.

**Step 4: Update ledger and commit**

Run: `jj describe -m "refactor: clean cargo-tarjanize"`
Expected: description updated for current working copy.

### Task 18: tarjanize-condense + tarjanize-viz Inventory and Overlap Scan

**Files:**
- Modify: `docs/cleanup-ledger.md`

**Step 1: Inventory condense and viz with Serena**

Run: use Serena `get_symbols_overview` for:
`crates/tarjanize-condense/src/*.rs`
`crates/tarjanize-viz/src/*.rs`

**Step 2: Identify overlapping algorithms**

Note shared logic candidates (SCC handling, graph transforms, etc).

**Step 3: Update ledger notes**

Add overlap notes in `docs/cleanup-ledger.md`.

**Step 4: Commit**

Run: `jj describe -m "docs: note condense/viz overlap"`
Expected: description updated for current working copy.

### Task 19: tarjanize-condense Comment Pass

**Files:**
- Modify: `crates/tarjanize-condense/src/lib.rs`
- Modify: `crates/tarjanize-condense/src/error.rs`
- Modify: `crates/tarjanize-condense/src/scc.rs`

**Step 1: Add what+why comments for SCC and anchor constraints**

Document why anchor constraints require merging and how SCCs are condensed.

**Step 2: Run fmt**

Run: `cargo fmt`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment tarjanize-condense"`
Expected: description updated for current working copy.

### Task 20: tarjanize-viz Comment Pass (Rust + TS)

**Files:**
- Modify: `crates/tarjanize-viz/src/lib.rs`
- Modify: `crates/tarjanize-viz/src/data.rs`
- Modify: `crates/tarjanize-viz/src/error.rs`
- Modify: `crates/tarjanize-viz/src/server.rs`
- Modify: `crates/tarjanize-viz/templates/renderer.ts`

**Step 1: Add what+why comments for API handlers and UI flow**

Document why each endpoint exists and how the UI uses it.

**Step 2: Run fmt and lint**

Run: `cargo fmt`
Expected: exit 0.

Run: `npm run lint`
Expected: exit 0.

**Step 3: Commit**

Run: `jj describe -m "docs: comment tarjanize-viz"`
Expected: description updated for current working copy.

### Task 21: Shared Algorithm Extraction (condense + viz)

**Files:**
- Modify: `crates/tarjanize-condense/src/*.rs`
- Modify: `crates/tarjanize-viz/src/*.rs`
- Modify: `crates/tarjanize-schedule/src/*.rs`

**Step 1: Extract or align shared logic**

Move common SCC or graph logic into a shared module when justified.

**Step 2: Add or update tests**

Ensure tests cover the shared logic without changing semantics.

**Step 3: Run clippy and tests**

Run: `cargo clippy -p tarjanize-condense -p tarjanize-viz --all-targets`
Expected: exit 0 with no warnings.

Run: `cargo test -p tarjanize-condense -p tarjanize-viz`
Expected: exit 0 with all tests passing.

**Step 4: Update ledger and commit**

Run: `jj describe -m "refactor: unify condense/viz logic"`
Expected: description updated for current working copy.

### Task 22: tarjanize-cost + tarjanize-magsac Sweep

**Files:**
- Modify: `crates/tarjanize-cost/src/lib.rs`
- Modify: `crates/tarjanize-magsac/src/lib.rs`
- Modify: `docs/cleanup-ledger.md`

**Step 1: Comment pass and structural cleanup**

Add what+why comments and remove duplication or unused helpers.

**Step 2: Run clippy and tests**

Run: `cargo clippy -p tarjanize-cost -p tarjanize-magsac --all-targets`
Expected: exit 0 with no warnings.

Run: `cargo test -p tarjanize-cost -p tarjanize-magsac`
Expected: exit 0 with all tests passing.

**Step 3: Update ledger and commit**

Run: `jj describe -m "refactor: clean tarjanize-cost and magsac"`
Expected: description updated for current working copy.

### Task 23: tarjanize CLI Sweep

**Files:**
- Modify: `crates/tarjanize/src/main.rs`
- Modify: `docs/cleanup-ledger.md`

**Step 1: Comment pass and cleanup**

Add what+why comments for CLI flow and subcommands.

**Step 2: Run clippy and tests**

Run: `cargo clippy -p tarjanize --all-targets`
Expected: exit 0 with no warnings.

Run: `cargo test -p tarjanize`
Expected: exit 0 with all tests passing.

**Step 3: Update ledger and commit**

Run: `jj describe -m "refactor: clean tarjanize CLI"`
Expected: description updated for current working copy.

### Task 24: Final Docs and Ledger Pass

**Files:**
- Modify: `docs/cleanup-ledger.md`
- Modify: `docs/*.md`

**Step 1: Scan for stale docs**

Update any doc that references changed behavior or APIs.

**Step 2: Final lint and tests**

Run: `cargo clippy --all-targets`
Expected: exit 0 with no warnings.

Run: `cargo test`
Expected: exit 0 with all tests passing.

Run: `npm run lint`
Expected: exit 0 with no warnings.

**Step 3: Mark ledger complete and commit**

Run: `jj describe -m "docs: finalize cleanup ledger"`
Expected: description updated for current working copy.
