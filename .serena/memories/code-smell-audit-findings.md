# Code Smell Audit Findings (2026-02-13)

## Duplicate Code

### `duration_to_ms_f64()` — 7 copies across 6 files
All identical: `duration.as_secs_f64() * 1000.0`
- `tarjanize-condense/src/scc.rs` (lines 22-27)
- `tarjanize-schedule/src/lib.rs` (lines 221-226)
- `tarjanize-schedule/src/recommend.rs` (lines 428-433)
- `cargo-tarjanize/src/profile.rs` (lines 492-497, inside impl)
- `cargo-tarjanize/src/driver.rs` (lines 1177-1183)
- `tarjanize-cost/src/lib.rs` (lines 451-456)
- `cargo-tarjanize/tests/cost_extraction_tests.rs` (lines 36-42, test code)

### `count_symbols()` — 3 copies (2 production + 1 variant)
Recursive module-tree symbol counter:
- `cargo-tarjanize/src/driver.rs` (lines 921-928) — `count_symbols`
- `tarjanize-schedule/src/lib.rs` (lines 228-237) — `count_symbols`
- `tarjanize-cost/src/lib.rs` (lines 415-422) — `count_symbols_in_module`

### `collect_frontend_cost()` — 2 copies
Sums event times recursively through module tree:
- `tarjanize-schedule/src/lib.rs` (lines 204-210)
- `tarjanize-cost/src/lib.rs` (lines 402-410)

## Missing Debug on Public Types
- `CrateResult` in `cargo-tarjanize/src/driver.rs` (only has Serialize/Deserialize)
- `ExtractionResult` in `cargo-tarjanize/src/extract.rs` (no derives)
- `AppState` in `tarjanize-viz/src/server.rs` (no derives)

## Production expect/unwrap (27 call sites)
Most have invariant-explaining messages. Notable clusters:
- **tarjanize-viz/server.rs**: ~10 RwLock `.expect("...lock poisoned")` calls
- **tarjanize-schedule/recommend.rs**: ~8 `.expect()` on format assertions and u32 conversions
- **tarjanize-condense/scc.rs**: 3 `.expect()` on index lookups and non-empty assertions
- **tarjanize-schedule/target_graph.rs**: 3 `.expect()` on graph edge/node lookups
- **tarjanize-schedule/schedule.rs**: 1 `.expect()` on topological sort
- **tarjanize-magsac/lib.rs**: 1 `.unwrap()` on float comparison
- **tarjanize-cost/lib.rs**: 1 `.unwrap()` on duration rounding
- **cargo-tarjanize/profile.rs**: 2 `.expect()` on non-empty vec and stack pop
