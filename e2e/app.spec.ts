// End-to-end tests for the tarjanize interactive split explorer.
//
// These tests run against a live tarjanize viz web server (started by
// global-setup.ts) using the Omicron symbol graph fixture (~160
// packages, ~316 targets). The real data exercises CDN loading,
// large-graph rendering, and realistic SCC structures.

import { test, expect, type Page } from "@playwright/test";
import { getBaseUrl } from "./helpers.ts";

let baseUrl: string;

test.beforeAll(() => {
  baseUrl = getBaseUrl();
});

// -----------------------------------------------------------------------
// Page load and initial rendering
// -----------------------------------------------------------------------

test.describe("page load", () => {
  test("renders the Gantt chart canvas", async ({ page }) => {
    await page.goto(baseUrl);

    // The PixiJS canvas is rendered inside #viewport.
    const canvas = page.locator("#viewport canvas");
    await expect(canvas).toBeVisible({ timeout: 15_000 });
  });

  test("populates sidebar statistics", async ({ page }) => {
    await page.goto(baseUrl);

    // Wait for the stats to be filled in by renderer.js.
    const cpStat = page.locator("#stat-cp");
    await expect(cpStat).not.toHaveText("", { timeout: 15_000 });

    // All stat fields should be populated with non-empty values.
    await expect(page.locator("#stat-total")).not.toHaveText("");
    await expect(page.locator("#stat-parallel")).not.toHaveText("");
    await expect(page.locator("#stat-targets")).not.toHaveText("");
    await expect(page.locator("#stat-symbols")).not.toHaveText("");
    await expect(page.locator("#stat-lanes")).not.toHaveText("");
  });

  test("target count is at least 100", async ({ page }) => {
    await page.goto(baseUrl);

    // The Omicron fixture has ~316 targets. Verify we loaded a
    // nontrivial number (exact count may vary with fixture updates).
    const targetsStat = page.locator("#stat-targets");
    await expect(targetsStat).not.toHaveText("", { timeout: 15_000 });
    const count = parseInt(await targetsStat.textContent() ?? "0");
    expect(count).toBeGreaterThan(100);
  });

  test("recommendations hidden before target selection", async ({ page }) => {
    await page.goto(baseUrl);
    await expect(page.locator("#stat-cp")).not.toHaveText("", {
      timeout: 15_000,
    });

    await expect(page.locator("#recommendations")).toBeHidden();
  });
});

// -----------------------------------------------------------------------
// Search
// -----------------------------------------------------------------------

test.describe("search", () => {
  test("search input accepts text", async ({ page }) => {
    await page.goto(baseUrl);
    await expect(page.locator("#stat-cp")).not.toHaveText("", {
      timeout: 15_000,
    });

    const searchInput = page.locator("#search-input");
    await searchInput.fill("nexus");

    // Give the renderer time to process the search filter.
    await page.waitForTimeout(200);
    await expect(searchInput).toHaveValue("nexus");
  });
});

// -----------------------------------------------------------------------
// Target click → Recommendations panel
// -----------------------------------------------------------------------

// Helper: wait for initial page load to complete.
async function waitForLoad(page: Page, url: string) {
  await page.goto(url);
  await expect(page.locator("#stat-cp")).not.toHaveText("", {
    timeout: 15_000,
  });
}

// Helper: select a target by dispatching the custom event.
// Returns the target name used.
async function selectTarget(page: Page, targetName: string) {
  await page.evaluate((name) => {
    window.dispatchEvent(
      new CustomEvent("target-click", {
        detail: { name, index: 0 },
      }),
    );
  }, targetName);
}

// Pick a target that will exist in the Omicron fixture and has enough
// symbols to have a non-trivial SCC DAG.
const TEST_TARGET = "omicron-rpaths/lib";

test.describe("target drill-down", () => {
  test("clicking a target shows recommendations", async ({ page }) => {
    await waitForLoad(page, baseUrl);
    await selectTarget(page, TEST_TARGET);

    // Wait for the recommendations panel to appear.
    const recsDiv = page.locator("#recommendations");
    await expect(recsDiv).toBeVisible({ timeout: 10_000 });

    // The header should show the target name.
    const recHeader = page.locator("#rec-header");
    await expect(recHeader).toContainText(TEST_TARGET);
  });

  test("clicking same target again hides recommendations", async ({
    page,
  }) => {
    await waitForLoad(page, baseUrl);
    await selectTarget(page, TEST_TARGET);

    // Wait for recommendations to appear.
    await expect(page.locator("#recommendations")).toBeVisible({
      timeout: 10_000,
    });

    // Click same target again to deselect.
    await selectTarget(page, TEST_TARGET);

    await expect(page.locator("#recommendations")).toBeHidden();
  });
});

// -----------------------------------------------------------------------
// Save functionality
// -----------------------------------------------------------------------

test.describe("save", () => {
  test("Save button triggers JSON download", async ({ page }) => {
    await waitForLoad(page, baseUrl);

    // Intercept the export API to return a small mock — the real
    // Omicron fixture is ~590MB and would timeout.
    await page.route("**/api/export", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ packages: { mock: { targets: {} } } }),
      }),
    );

    const downloadPromise = page.waitForEvent("download");
    await page.click("#save-btn");
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toBe("symbol_graph.json");

    // Verify the downloaded file is valid JSON.
    const path = await download.path();
    const { readFileSync } = await import("node:fs");
    const json = JSON.parse(readFileSync(path!, "utf-8"));
    expect(json).toHaveProperty("packages");
  });
});

// -----------------------------------------------------------------------
// API contract tests (browser-side fetch)
// -----------------------------------------------------------------------

test.describe("API from browser", () => {
  test("GET /api/schedule returns valid schedule data", async ({ page }) => {
    await page.goto(baseUrl);

    const schedule = await page.evaluate(async () => {
      const resp = await fetch("/api/schedule");
      return resp.json();
    });

    expect(schedule).toHaveProperty("targets");
    expect(schedule).toHaveProperty("summary");
    expect(schedule.summary.target_count).toBeGreaterThan(100);
  });

  // Note: GET /api/export is not tested here because the Omicron
  // fixture produces a ~590MB JSON response that exceeds browser
  // fetch limits. The export endpoint is covered by Rust-side API
  // contract tests in server.rs (api_export_returns_valid_symbol_graph).
});
