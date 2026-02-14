# Frontend E2E UX Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Playwright fixture with a workspace-generated symbol graph and expand Playwright coverage for the viz UX (resize, shatter, tree, export, pan/zoom) without changing JSON schema or UX semantics.

**Architecture:** Keep the existing Playwright global setup that pipes `e2e/fixture.json.gz` into `tarjanize viz`. Use deterministic `window.updateSchedule(...)` for interaction tests while validating real export with the workspace fixture. No new fixture routing or schema changes.

**Tech Stack:** Playwright (TypeScript), Node.js, tarjanize CLI (`cargo tarjanize`).

---

**Notes**
- Workflow: start each task in a fresh JJ change with `jj new` (skip if the working copy is empty), and commit with `jj describe` when the task is complete.
- TDD: Playwright tests have an approved post-hoc exception, but still run targeted tests after each change.
- Skills: @using-superpowers, @executing-plans, @test-driven-development (except the Playwright post-hoc exception).

### Task 1: Refresh Fixture + Real Export Test

**Files:**
- Modify: `e2e/app.spec.ts`
- Modify: `e2e/fixture.json.gz`

**Step 1: Write the failing test**
Update the Save test to use the real `/api/export` response (remove the route mock) and update the file header comment to describe the workspace-generated fixture.

```ts
test("Save button triggers JSON download", async ({ page }) => {
  await waitForLoad(page, baseUrl);

  const downloadPromise = page.waitForEvent("download");
  await page.click("#save-btn");
  const download = await downloadPromise;

  expect(download.suggestedFilename()).toBe("symbol_graph.json");

  const path = await download.path();
  const { readFileSync } = await import("node:fs");
  const json = JSON.parse(readFileSync(path!, "utf-8"));
  expect(json).toHaveProperty("packages");
  expect(Object.keys(json.packages).length).toBeGreaterThan(0);
});
```

**Step 2: Run test to verify it fails**
Run: `npx playwright test e2e/app.spec.ts -g "Save button triggers JSON download"`
Expected: FAIL (export too large or times out with the old fixture).

**Step 3: Write minimal implementation**
Regenerate the fixture from the workspace once and replace it in place (checked in).

Note: this is a one-time update. The Playwright global setup reads the
checked-in `e2e/fixture.json.gz`; it should not rerun `cargo tarjanize`
as part of the tests.

Run: `cargo tarjanize -o /tmp/symbol_graph.json`  
If nightly is required: `cargo +nightly tarjanize -o /tmp/symbol_graph.json`  
Run: `gzip -c /tmp/symbol_graph.json > e2e/fixture.json.gz`

Update the “target count is at least …” assertion to a conservative floor based on the new fixture.

```bash
python3 - <<'PY'
import json
with open("/tmp/symbol_graph.json") as f:
    data = json.load(f)
count = sum(len(p["targets"]) for p in data["packages"].values())
print(count)
PY
```

Set the expectation in `e2e/app.spec.ts` to a stable floor (for example, `>= max(5, count // 2)`).
Remove or rewrite the comment that claims `/api/export` is untested due to size.

**Step 4: Run test to verify it passes**
Run: `npx playwright test e2e/app.spec.ts -g "Save button triggers JSON download"`
Expected: PASS

**Step 5: Commit**
```bash
jj describe -m "test: refresh fixture and real export"
```

### Task 2: Dynamic Target Selection + Tree View Smoke Test

**Files:**
- Modify: `e2e/app.spec.ts`

**Step 1: Write the failing test**
Add a helper to pick a target dynamically, update the existing target drill-down tests to use it, and add a tree view smoke test.

```ts
async function firstTargetName(page: Page): Promise<string> {
  return page.evaluate(() => (window as any).DATA.targets[0].name);
}
```

```ts
test("target selection renders module tree", async ({ page }) => {
  await waitForLoad(page, baseUrl);
  const target = await firstTargetName(page);
  await selectTarget(page, target);

  const header = page.locator("#rec-list .tree-module-header").first();
  await expect(header).toBeVisible();

  const symbol = page.locator("#rec-list .tree-symbol").first();
  await expect(symbol).toBeVisible();
});
```

**Step 2: Run test to verify it fails**
Run: `npx playwright test e2e/app.spec.ts -g "target drill-down|module tree"`
Expected: FAIL (new test not yet present or missing waits).

**Step 3: Write minimal implementation**
If needed, add a wait for the recommendations panel or `#rec-list` before asserting.
Keep selectors anchored to `#rec-list` to avoid unrelated elements.

**Step 4: Run test to verify it passes**
Run: `npx playwright test e2e/app.spec.ts -g "target drill-down|module tree"`
Expected: PASS

**Step 5: Commit**
```bash
jj describe -m "test: dynamic target selection and tree view"
```

### Task 3: Sidebar Resize Interaction Test

**Files:**
- Modify: `e2e/app.spec.ts`

**Step 1: Write the failing test**
```ts
test("sidebar resize updates layout and keeps hover working", async ({ page }) => {
  await waitForLoad(page, baseUrl);
  await page.evaluate(() => {
    window.updateSchedule({
      summary: {
        critical_path_ms: 1000,
        total_cost_ms: 1000,
        parallelism_ratio: 1.0,
        target_count: 1,
        symbol_count: 1,
        lane_count: 1,
      },
      targets: [{
        name: "test-pkg/lib",
        start: 0,
        finish: 1000,
        cost: 1000,
        slack: 0,
        lane: 0,
        symbol_count: 1,
        deps: [],
        dependents: [],
        on_critical_path: true,
        forward_pred: null,
        backward_succ: null,
      }],
      critical_path: [0],
    });
  });

  const panel = page.locator("#right-panel");
  const handle = page.locator("#sidebar-resize");

  const before = await panel.boundingBox();
  const handleBox = await handle.boundingBox();
  if (!before || !handleBox) throw new Error("missing panel/handle");

  await page.mouse.move(handleBox.x + 2, handleBox.y + 2);
  await page.mouse.down();
  await page.mouse.move(handleBox.x - 120, handleBox.y + 2);
  await page.mouse.up();

  const after = await panel.boundingBox();
  if (!after) throw new Error("missing panel");
  expect(after.width).not.toBeCloseTo(before.width, 1);

  const viewport = page.locator("#viewport");
  const box = await viewport.boundingBox();
  if (!box) throw new Error("missing viewport");
  await page.mouse.move(box.x + 30, box.y + 10);
  await expect(page.locator("#tooltip")).toBeVisible({ timeout: 2000 });
});
```

**Step 2: Run test to verify it fails**
Run: `npx playwright test e2e/app.spec.ts -g "sidebar resize updates layout"`
Expected: FAIL (new test not yet present).

**Step 3: Write minimal implementation**
If the resize width check is flaky, switch to reading the CSS variable:

```ts
const sidebarW = await page.evaluate(() =>
  getComputedStyle(document.getElementById("right-panel")!)
    .getPropertyValue("--sidebar-w")
);
```

Keep the hover assertion to ensure chart interaction still works.

**Step 4: Run test to verify it passes**
Run: `npx playwright test e2e/app.spec.ts -g "sidebar resize updates layout"`
Expected: PASS

**Step 5: Commit**
```bash
jj describe -m "test: sidebar resize interaction"
```

### Task 4: Shatter Workflow Test

**Files:**
- Modify: `e2e/app.spec.ts`

**Step 1: Write the failing test**
```ts
test("shatter replaces target with group targets", async ({ page }) => {
  await waitForLoad(page, baseUrl);
  const target = await firstTargetName(page);
  await selectTarget(page, target);

  const responsePromise = page.waitForResponse("**/api/shatter");
  await page.click("#shatter-btn");
  await responsePromise;

  await page.waitForFunction(() =>
    (window as any).DATA.targets.some((t: any) => t.name.includes("::group_"))
  );
  const hasGroup = await page.evaluate(() =>
    (window as any).DATA.targets.some((t: any) => t.name.includes("::group_"))
  );
  expect(hasGroup).toBe(true);
});
```

**Step 2: Run test to verify it fails**
Run: `npx playwright test e2e/app.spec.ts -g "shatter replaces target"`
Expected: FAIL (new test not yet present).

**Step 3: Write minimal implementation**
If `::group_` names are inconsistent, assert that `summary.target_count` increases after shatter:

```ts
const before = await page.evaluate(() => (window as any).DATA.summary.target_count);
// ... shatter ...
const after = await page.evaluate(() => (window as any).DATA.summary.target_count);
expect(after).toBeGreaterThan(before);
```

**Step 4: Run test to verify it passes**
Run: `npx playwright test e2e/app.spec.ts -g "shatter replaces target"`
Expected: PASS

**Step 5: Commit**
```bash
jj describe -m "test: shatter workflow"
```

### Task 5: Pan/Zoom Interaction Test

**Files:**
- Modify: `e2e/app.spec.ts`

**Step 1: Write the failing test**
```ts
test("pan/zoom keeps hover working", async ({ page }) => {
  await waitForLoad(page, baseUrl);
  await page.evaluate(() => {
    window.updateSchedule({
      summary: {
        critical_path_ms: 2000,
        total_cost_ms: 2000,
        parallelism_ratio: 1.0,
        target_count: 1,
        symbol_count: 1,
        lane_count: 1,
      },
      targets: [{
        name: "test-pkg/lib",
        start: 0,
        finish: 2000,
        cost: 2000,
        slack: 0,
        lane: 0,
        symbol_count: 1,
        deps: [],
        dependents: [],
        on_critical_path: true,
        forward_pred: null,
        backward_succ: null,
      }],
      critical_path: [0],
    });
  });

  const viewport = page.locator("#viewport");
  const box = await viewport.boundingBox();
  if (!box) throw new Error("missing viewport");

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 120, box.y + box.height / 2);
  await page.mouse.up();

  await page.mouse.wheel(0, -200);

  await page.mouse.move(box.x + 30, box.y + 10);
  await expect(page.locator("#tooltip")).toBeVisible({ timeout: 2000 });
});
```

**Step 2: Run test to verify it fails**
Run: `npx playwright test e2e/app.spec.ts -g "pan/zoom keeps hover"`
Expected: FAIL (new test not yet present).

**Step 3: Write minimal implementation**
If hover is flaky after pan/zoom, add a short wait before asserting:

```ts
await page.waitForTimeout(100);
```

**Step 4: Run test to verify it passes**
Run: `npx playwright test e2e/app.spec.ts -g "pan/zoom keeps hover"`
Expected: PASS

**Step 5: Commit**
```bash
jj describe -m "test: pan/zoom hover stability"
```

### Task 6: Final Verification

**Files:**
- None (verification only)

**Step 1: Run the full lint and e2e suite**
Run: `npm run lint`  
Run: `npx playwright test e2e/app.spec.ts`  
Expected: PASS

**Step 2: Commit (only if fixes were required)**
```bash
jj describe -m "test: finalize frontend e2e coverage"
```
