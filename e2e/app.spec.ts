// End-to-end tests for the tarjanize interactive split explorer.
//
// These tests run against a live tarjanize viz web server (started by
// global-setup.ts) using a tokio symbol graph fixture (~10 packages,
// ~227 targets) checked into `e2e/fixture.json.gz`. The real data
// exercises the full rendering pipeline without relying on mocked
// responses.

import { test, expect, type Page } from "@playwright/test";
import type { ScheduleData } from "../types/generated/ScheduleData";
import type { TargetData } from "../types/generated/TargetData";
import { getBaseUrl } from "./helpers.ts";

// ---------------------------------------------------------------------------
// Named constants for magic numbers
// ---------------------------------------------------------------------------

/** Timeout (ms) for initial page load and rendering. */
const LOAD_TIMEOUT_MS = 15_000;

/** Timeout (ms) for UI interactions (tooltips, panels). */
const INTERACTION_TIMEOUT_MS = 10_000;

/** Timeout (ms) for short tooltip animations. */
const TOOLTIP_TIMEOUT_MS = 2000;

/** Delay (ms) to let the renderer process search input. */
const SEARCH_DEBOUNCE_MS = 200;

/** Minimum expected target count in the tokio fixture. */
const MIN_TARGET_COUNT = 10;

/** Small pixel offset used when positioning the mouse near element edges. */
const SMALL_OFFSET_PX = 10;

/** Pixel offset into the chart area for hover tests. */
const CHART_HOVER_X_PX = 30;

/** Pixel drag distance for resize and pan tests. */
const DRAG_DISTANCE_PX = 120;

/** Small offset (px) for positioning on a drag handle. */
const HANDLE_OFFSET_PX = 2;

/** Zoom delta for mouse wheel tests. */
const ZOOM_DELTA = -200;

/** Radix for parseInt calls. */
const DECIMAL_RADIX = 10;

/** Divisor for computing the center of a bounding box dimension. */
const HALF = 2;

/** Index of the first element in an array. */
const FIRST_INDEX = 0;

/** Count/ratio values for a single-target synthetic schedule. */
const SINGLE_COUNT = 1;

/**
 * Type guard: check that an unknown value has the ScheduleData shape.
 *
 * Full runtime validation isn't warranted in e2e tests — the server
 * already validates its own output. This performs a minimal structural
 * check to satisfy the strict type-assertion lint rule.
 */
function isScheduleData(value: unknown): value is ScheduleData {
  return (
    typeof value === "object" &&
    value !== null &&
    "targets" in value &&
    Array.isArray(value.targets) &&
    "summary" in value
  );
}

/** Assert that a value is ScheduleData, throwing if the shape is wrong. */
function assertScheduleData(value: unknown): ScheduleData {
  if (isScheduleData(value)) {
    return value;
  }
  throw new Error("value is not a ScheduleData");
}

/** Cost (ms) for the single-bar schedule in hover / resize tests. */
const SINGLE_BAR_COST_MS = 1000;

/** Cost (ms) for the single-bar schedule in pan/zoom tests. */
const PAN_ZOOM_COST_MS = 2000;

// ---------------------------------------------------------------------------
// Synthetic schedule data used in multiple tests
// ---------------------------------------------------------------------------

/** Single-bar schedule for tests that need a predictable chart layout. */
function makeSingleBarSchedule(costMs: number): ScheduleData {
  return {
    summary: {
      critical_path_ms: costMs,
      total_cost_ms: costMs,
      parallelism_ratio: SINGLE_COUNT,
      target_count: SINGLE_COUNT,
      symbol_count: SINGLE_COUNT,
      lane_count: SINGLE_COUNT,
    },
    targets: [
      {
        name: "test-pkg/lib",
        start: FIRST_INDEX,
        finish: costMs,
        cost: costMs,
        slack: FIRST_INDEX,
        lane: FIRST_INDEX,
        symbol_count: SINGLE_COUNT,
        deps: [],
        dependents: [],
        on_critical_path: true,
      },
    ],
    critical_path: [FIRST_INDEX],
  };
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

let baseUrl = "";

test.beforeAll(() => {
  baseUrl = getBaseUrl();
});

// -----------------------------------------------------------------------
// Page load and initial rendering
// -----------------------------------------------------------------------

test.describe("page load", () => {
  test("renders the Gantt chart canvas", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await page.goto(baseUrl);

    // The PixiJS canvas is rendered inside #viewport.
    const canvas = page.locator("#viewport canvas");
    await expect(canvas).toBeVisible({ timeout: LOAD_TIMEOUT_MS });
  });

  test("populates sidebar statistics", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await page.goto(baseUrl);

    // Wait for the stats to be filled in by renderer.js.
    const cpStat = page.locator("#stat-cp");
    await expect(cpStat).not.toHaveText("", { timeout: LOAD_TIMEOUT_MS });

    // All stat fields should be populated with non-empty values.
    await expect(page.locator("#stat-total")).not.toHaveText("");
    await expect(page.locator("#stat-parallel")).not.toHaveText("");
    await expect(page.locator("#stat-targets")).not.toHaveText("");
    await expect(page.locator("#stat-symbols")).not.toHaveText("");
    await expect(page.locator("#stat-lanes")).not.toHaveText("");
  });

  test("target count is at least 10", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await page.goto(baseUrl);

    // The tokio fixture has ~227 targets. Use a conservative floor
    // (exact count may vary with fixture updates).
    const targetsStat = page.locator("#stat-targets");
    await expect(targetsStat).not.toHaveText("", { timeout: LOAD_TIMEOUT_MS });
    const count = Number.parseInt(
      (await targetsStat.textContent()) ?? "0",
      DECIMAL_RADIX,
    );
    expect(count).toBeGreaterThanOrEqual(MIN_TARGET_COUNT);
  });

  test("recommendations hidden before target selection", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await page.goto(baseUrl);
    await expect(page.locator("#stat-cp")).not.toHaveText("", {
      timeout: LOAD_TIMEOUT_MS,
    });

    await expect(page.locator("#recommendations")).toBeHidden();
  });
});

// -----------------------------------------------------------------------
// Search
// -----------------------------------------------------------------------

test.describe("search", () => {
  test("search input accepts text", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await page.goto(baseUrl);
    await expect(page.locator("#stat-cp")).not.toHaveText("", {
      timeout: LOAD_TIMEOUT_MS,
    });

    const searchInput = page.locator("#search-input");
    await searchInput.fill("tokio");

    // Give the renderer time to process the search filter.
    await page.waitForTimeout(SEARCH_DEBOUNCE_MS);
    await expect(searchInput).toHaveValue("tokio");
  });
});

// -----------------------------------------------------------------------
// Hover behavior
// -----------------------------------------------------------------------

test.describe("hover", () => {
  test("sidebar hover clears chart tooltip", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);

    // Replace schedule with a single, predictable bar.
    await page.evaluate((schedule: ScheduleData) => {
      globalThis.updateSchedule(schedule);
    }, makeSingleBarSchedule(SINGLE_BAR_COST_MS));

    // Hover the bar to show the tooltip.
    const viewport = page.locator("#viewport");
    const box = await viewport.boundingBox();
    if (box === null) throw new Error("viewport not found");
    await page.mouse.move(
      box.x + CHART_HOVER_X_PX,
      box.y + SMALL_OFFSET_PX,
    );
    await expect(page.locator("#tooltip")).toBeVisible({
      timeout: TOOLTIP_TIMEOUT_MS,
    });

    // Move the pointer over the sidebar; tooltip should hide.
    const sidebar = page.locator("#sidebar");
    const sidebarBox = await sidebar.boundingBox();
    if (sidebarBox === null) throw new Error("sidebar not found");
    await page.mouse.move(
      sidebarBox.x + SMALL_OFFSET_PX,
      sidebarBox.y + SMALL_OFFSET_PX,
    );
    await expect(page.locator("#tooltip")).toBeHidden({
      timeout: TOOLTIP_TIMEOUT_MS,
    });
  });

  test("sidebar help tooltip layers above resize handle", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);

    const helpIcon = page.locator(".help-icon").first();
    await helpIcon.hover();

    const zValues = await page.evaluate(
      (): { tooltipZ: string; tooltipContent: string; handleZ: string } | undefined => {
        const icon = document.querySelector(".help-icon");
        const handle = document.querySelector("#sidebar-resize");
        if (icon === null || handle === null) return undefined;
        const tooltipStyle = getComputedStyle(icon, "::after");
        const handleStyle = getComputedStyle(handle);
        return {
          tooltipZ: tooltipStyle.zIndex,
          tooltipContent: tooltipStyle.content,
          handleZ: handleStyle.zIndex,
        };
      },
    );
    if (zValues === undefined) throw new Error("missing tooltip or handle");

    expect(zValues.tooltipContent).not.toBe("none");
    expect(Number(zValues.tooltipZ)).toBeGreaterThan(Number(zValues.handleZ));
  });
});

// -----------------------------------------------------------------------
// Target click -> Recommendations panel
// -----------------------------------------------------------------------

/** Assert that baseUrl has been initialized by beforeAll. */
function assertBaseUrl(
  value: string,
): asserts value is string {
  if (value === "") {
    throw new Error("baseUrl not initialized -- did beforeAll run?");
  }
}

// Helper: wait for initial page load to complete.
async function waitForLoad(page: Page, url: string): Promise<void> {
  await page.goto(url);
  await expect(page.locator("#stat-cp")).not.toHaveText("", {
    timeout: LOAD_TIMEOUT_MS,
  });
}

// Helper: select a target by dispatching the custom event.
async function selectTarget(
  page: Page,
  targetName: string,
): Promise<void> {
  await page.evaluate(({ name, index }) => {
    globalThis.dispatchEvent(
      new CustomEvent("target-click", {
        detail: { name, index },
      }),
    );
  }, { name: targetName, index: FIRST_INDEX });
}

// Helper: pick the first target from the loaded schedule data.
async function firstTargetName(page: Page): Promise<string> {
  return await page.evaluate(
    (index) => globalThis.DATA.targets[index].name,
    FIRST_INDEX,
  );
}

test.describe("target drill-down", () => {
  test("clicking a target shows recommendations", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);
    const target = await firstTargetName(page);
    await selectTarget(page, target);

    // Wait for the recommendations panel to appear.
    const recsDiv = page.locator("#recommendations");
    await expect(recsDiv).toBeVisible({ timeout: INTERACTION_TIMEOUT_MS });

    // The header should show the target name.
    const recHeader = page.locator("#rec-header");
    await expect(recHeader).toContainText(target);
  });

  test("clicking same target again hides recommendations", async ({
    page,
  }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);
    const target = await firstTargetName(page);
    await selectTarget(page, target);

    // Wait for recommendations to appear.
    await expect(page.locator("#recommendations")).toBeVisible({
      timeout: INTERACTION_TIMEOUT_MS,
    });

    // Click same target again to deselect.
    await selectTarget(page, target);

    await expect(page.locator("#recommendations")).toBeHidden();
  });

  test("target selection renders module tree", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);
    const target = await firstTargetName(page);
    await selectTarget(page, target);

    // Module headers should appear in the recommendations list.
    const header = page.locator("#rec-list .tree-module-header").first();
    await expect(header).toBeVisible({ timeout: INTERACTION_TIMEOUT_MS });

    // Expand the first module to reveal symbols.
    await header.click();
    const symbol = page.locator("#rec-list .tree-symbol").first();
    await expect(symbol).toBeVisible();
  });
});

// -----------------------------------------------------------------------
// Sidebar resize
// -----------------------------------------------------------------------

test.describe("sidebar resize", () => {
  test("sidebar resize updates layout and keeps hover working", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);
    await page.evaluate((schedule: ScheduleData) => {
      globalThis.updateSchedule(schedule);
    }, makeSingleBarSchedule(SINGLE_BAR_COST_MS));

    const handle = page.locator("#sidebar-resize");

    const beforeWidth = await page.evaluate(() => {
      const panel = document.querySelector("#right-panel");
      if (panel === null) throw new Error("missing #right-panel");
      return getComputedStyle(panel).getPropertyValue("--sidebar-w");
    });

    const handleBox = await handle.boundingBox();
    if (handleBox === null) throw new Error("missing handle");

    await page.mouse.move(
      handleBox.x + HANDLE_OFFSET_PX,
      handleBox.y + HANDLE_OFFSET_PX,
    );
    await page.mouse.down();
    await page.mouse.move(
      handleBox.x - DRAG_DISTANCE_PX,
      handleBox.y + HANDLE_OFFSET_PX,
    );
    await page.mouse.up();

    const afterWidth = await page.evaluate(() => {
      const panel = document.querySelector("#right-panel");
      if (panel === null) throw new Error("missing #right-panel");
      return getComputedStyle(panel).getPropertyValue("--sidebar-w");
    });
    expect(afterWidth).not.toBe(beforeWidth);

    // Verify chart hover still works after resize.
    const viewport = page.locator("#viewport");
    const box = await viewport.boundingBox();
    if (box === null) throw new Error("missing viewport");
    await page.mouse.move(
      box.x + CHART_HOVER_X_PX,
      box.y + SMALL_OFFSET_PX,
    );
    await expect(page.locator("#tooltip")).toBeVisible({
      timeout: TOOLTIP_TIMEOUT_MS,
    });
  });
});

// -----------------------------------------------------------------------
// Shatter workflow
// -----------------------------------------------------------------------

/** Find a target suitable for shattering (start > 0, multiple symbols). */
function findShatterableTarget(targets: TargetData[]): string | undefined {
  const found = targets.find(
    (element) => element.start > FIRST_INDEX && element.symbol_count > SINGLE_COUNT,
  );
  return found?.name;
}

test.describe("shatter", () => {
  test("shatter replaces target with group targets", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);

    // Pick a target with start > 0 so shatter has a horizon to split at.
    // Targets at t=0 have no predecessors and won't split meaningfully.
    const { targets } = await page.evaluate(() => globalThis.DATA);
    const target = findShatterableTarget(targets);
    if (target === undefined) {
      // No shatterable target in fixture -- skip gracefully.
      test.skip();
      return;
    }

    await selectTarget(page, target);

    // Wait for the shatter API call to complete, then read the updated
    // schedule from globalThis.DATA (set by updateSchedule). Reading the
    // intercepted response body directly races with the page's own fetch
    // handler which consumes it first.
    await Promise.all([
      page.waitForResponse("**/api/shatter"),
      page.click("#shatter-btn"),
    ]);

    // Wait for updateSchedule to process the response and update
    // globalThis.DATA. The response arrives before the page's fetch
    // handler calls updateSchedule, so poll until group targets appear.
    // The predicate string avoids max-nested-callbacks lint limits.
    await page.waitForFunction(
      "globalThis.DATA.targets.some(t => t.name.includes('::group_'))",
      { timeout: INTERACTION_TIMEOUT_MS },
    );
  });
});

// -----------------------------------------------------------------------
// Pan/zoom
// -----------------------------------------------------------------------

test.describe("pan/zoom", () => {
  test("pan/zoom keeps hover working", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);
    await page.evaluate((schedule: ScheduleData) => {
      globalThis.updateSchedule(schedule);
    }, makeSingleBarSchedule(PAN_ZOOM_COST_MS));

    const viewport = page.locator("#viewport");
    const box = await viewport.boundingBox();
    if (box === null) throw new Error("missing viewport");

    // Pan left.
    const centerX = box.x + box.width / HALF;
    const centerY = box.y + box.height / HALF;
    await page.mouse.move(centerX, centerY);
    await page.mouse.down();
    await page.mouse.move(centerX - DRAG_DISTANCE_PX, centerY);
    await page.mouse.up();

    // Zoom in.
    await page.mouse.wheel(FIRST_INDEX, ZOOM_DELTA);

    // Hover should still trigger a tooltip.
    await page.mouse.move(
      box.x + CHART_HOVER_X_PX,
      box.y + SMALL_OFFSET_PX,
    );
    await expect(page.locator("#tooltip")).toBeVisible({
      timeout: TOOLTIP_TIMEOUT_MS,
    });
  });
});

// -----------------------------------------------------------------------
// Save functionality
// -----------------------------------------------------------------------

/** Shape of the exported symbol graph JSON for type-safe assertions. */
interface ExportedSymbolGraph {
  packages: Record<string, unknown>;
}

/** Type guard for the exported symbol graph JSON. */
function isExportedSymbolGraph(value: unknown): value is ExportedSymbolGraph {
  if (typeof value !== "object" || value === null) return false;
  return "packages" in value && typeof value.packages === "object";
}

test.describe("save", () => {
  test("Save button triggers JSON download", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await waitForLoad(page, baseUrl);

    const downloadPromise = page.waitForEvent("download");
    await page.click("#save-btn");
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toBe("symbol_graph.json");

    const filePath = await download.path();
    const { readFileSync } = await import("node:fs");
    const parsed: unknown = JSON.parse(readFileSync(filePath, "utf8"));
    if (!isExportedSymbolGraph(parsed)) {
      throw new Error("downloaded JSON is not a valid symbol graph");
    }
    expect(Object.keys(parsed.packages).length).toBeGreaterThan(
      FIRST_INDEX,
    );
  });
});

// -----------------------------------------------------------------------
// API contract tests (browser-side fetch)
// -----------------------------------------------------------------------

test.describe("API from browser", () => {
  test("GET /api/schedule returns valid schedule data", async ({ page }) => {
    assertBaseUrl(baseUrl);
    await page.goto(baseUrl);

    // Fetch via Playwright's request API (Node-side) to avoid browser-
    // context type assertion issues. The evaluate-in-browser approach
    // would require an unsafe `as` cast from `any`.
    const apiResponse = await page.request.get(`${baseUrl}/api/schedule`);
    const schedule = assertScheduleData(await apiResponse.json());

    expect(schedule).toHaveProperty("targets");
    expect(schedule).toHaveProperty("summary");
    expect(schedule.summary.target_count).toBeGreaterThanOrEqual(
      MIN_TARGET_COUNT,
    );
  });

  // Note: GET /api/export is validated by the Save button test, which
  // exercises the real export payload via browser download.
});
