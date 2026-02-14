# Frontend UX + Playwright Coverage Design

**Date:** 2026-02-13

## Goal
Define the intended UX for the tarjanize viz frontend and map it to Playwright coverage, identifying concrete gaps and a test strategy that reflects the intended experience (not just current behavior).

## Context
- The frontend lives in `crates/tarjanize-viz/templates/*` (renderer, sidebar, styles).
- Playwright is already configured (`playwright.config.ts`) with a global setup that runs `tarjanize viz` and uses a workspace-generated fixture (`e2e/fixture.json.gz`) produced via `cargo tarjanize`.
- Current e2e tests in `e2e/app.spec.ts` cover page load, sidebar stats, search input, target selection, and a mocked export download.
- Export is expected to **always** work and download valid `SymbolGraph` JSON.
- We have a one-time exception to strict TDD for adding Playwright tests post-hoc.

## Intended UX (Inferred)
1. **Page load**
   - The Gantt canvas renders in `#viewport`.
   - Sidebar stats populate (critical path, total cost, parallelism, targets, symbols, lanes).

2. **Hover tooltip**
   - Hovering a bar shows a tooltip with cost/start/finish/slack and dependency counts.
   - Hover state clears when leaving the chart viewport or moving into the sidebar.

3. **Search**
   - Typing filters highlight matches without altering underlying data.
   - Search does not break hover or selection behavior.

4. **Target selection + recommendations**
   - Clicking a bar selects a target and opens recommendations.
   - Clicking the same target again deselects and hides recommendations.

5. **Shatter by horizon**
   - Shatter updates the schedule and symbol graph.
   - Group targets (e.g., `::group_0`) appear and are selectable.
   - Tree view resolves group targets after shatter.

6. **Tree view**
   - Selecting a target populates the module tree and symbol list in the sidebar.
   - Dependency counts and cost breakdowns are consistent.

7. **Sidebar resize**
   - Dragging the handle updates sidebar width and chart viewport.
   - Resizing does not break hover, tooltip, or selection.

8. **Export**
   - Clicking Save always downloads valid `SymbolGraph` JSON.
   - Export should not rely on mocked responses in the main success path.

9. **Overlay layering**
   - Sidebar tooltips render above the resize handle and do not block chart interactions.

## Coverage Gaps (Playwright)
| UX Area | Existing Tests | Gap |
| --- | --- | --- |
| Hover tooltip + viewport bounds | None (new hover tests added ad-hoc) | Ensure hover only triggers within viewport, clears on sidebar hover |
| Sidebar resize behavior | None | Validate drag changes layout and chart still interactive |
| Shatter workflow | None | Verify shatter updates schedule + group targets |
| Tree view rendering | None | Verify sidebar tree contents after target selection |
| Export (real) | Mocked only | Real export download validates `SymbolGraph` JSON |
| Overlay layering | None | Tooltip above resize handle |
| Pan/zoom interactions | None | Hover and selection remain correct after pan/zoom |

## Proposed Test Strategy
1. **Deterministic UI cases**
   - Use `window.updateSchedule(...)` in tests to load a small, predictable schedule for hover/resize/pan scenarios.
   - This keeps interactions stable without needing to parse huge fixtures.

2. **Real export validation**
   - Regenerate the fixture from the workspace once using `cargo tarjanize`.
   - Replace `e2e/fixture.json.gz` in place (checked in) so export tests exercise real API data without extra routing.

3. **Coverage additions**
   - Hover tooltip only inside viewport (already prototyped).
   - Hover clears when pointer moves to sidebar.
   - Sidebar resize updates layout and does not allow chart hover bleed.
   - Shatter updates schedule and adds group targets.
   - Tree view populates after target selection (basic smoke).
   - Export downloads real `SymbolGraph` JSON (workspace fixture).

## Non-Goals
- Pixel-perfect visual regression.
- Cross-browser coverage (Chromium-only is fine).
- Performance benchmarking in Playwright.

## Risks / Constraints
- Regenerating the fixture must keep the payload small enough for reliable Playwright runs.
- The Playwright global setup currently runs `tarjanize viz` and can require escalated permissions in this environment.
- Export correctness relies on the server-side `SymbolGraph`; client-side `updateSchedule` does not affect export output.

## Recommendation
Proceed with the deterministic UI tests using `updateSchedule` and replace `e2e/fixture.json.gz` with a workspace-generated fixture to enable real export tests without massive payloads.
