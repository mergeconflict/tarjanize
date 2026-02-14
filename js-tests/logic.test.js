// Tests for pure functions in logic.js.
//
// logic.js is a proper ES module, so we import directly.

import { describe, it, expect } from "vitest";
import {
  niceTimeStep,
  formatMs,
  criticalPathThrough,
  searchFilter,
  slackColor,
  pointInRect,
} from "../crates/tarjanize-viz/templates/logic.js";

// ---------------------------------------------------------------------------
// niceTimeStep
// ---------------------------------------------------------------------------

describe("niceTimeStep", () => {
  it("returns 100ms at 1 px/ms with default 100px target", () => {
    expect(niceTimeStep(1, 100)).toBe(100);
  });

  it("returns 10000ms (10s) when zoomed out to 0.01 px/ms", () => {
    expect(niceTimeStep(0.01, 100)).toBe(10000);
  });

  it("returns 10ms when zoomed in to 10 px/ms", () => {
    expect(niceTimeStep(10, 100)).toBe(10);
  });

  it("returns 600000ms (10min) at extreme zoom-out 0.0001 px/ms", () => {
    // msPerTarget = 100 / 0.0001 = 1_000_000, exceeds all steps -> last step
    expect(niceTimeStep(0.0001, 100)).toBe(600000);
  });

  it("returns 1ms at extreme zoom-in 1000 px/ms", () => {
    // msPerTarget = 100 / 1000 = 0.1, first step (1) >= 0.1
    expect(niceTimeStep(1000, 100)).toBe(1);
  });

  it("uses default targetPx of 100 when omitted", () => {
    expect(niceTimeStep(1)).toBe(100);
  });

  it("picks the first step that satisfies the spacing", () => {
    // 100 / 0.5 = 200ms needed -> should pick 200
    expect(niceTimeStep(0.5, 100)).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// formatMs
// ---------------------------------------------------------------------------

describe("formatMs", () => {
  it("formats sub-second as milliseconds", () => {
    expect(formatMs(500)).toBe("500ms");
  });

  it("formats sub-minute as seconds with one decimal", () => {
    expect(formatMs(12300)).toBe("12.3s");
  });

  it("formats minutes and seconds", () => {
    expect(formatMs(70000)).toBe("1m 10s");
  });

  it("omits seconds when exactly on the minute", () => {
    expect(formatMs(120000)).toBe("2m");
  });

  it("rounds milliseconds", () => {
    expect(formatMs(0)).toBe("0ms");
    expect(formatMs(999)).toBe("999ms");
  });
});

// ---------------------------------------------------------------------------
// criticalPathThrough
// ---------------------------------------------------------------------------

describe("criticalPathThrough", () => {
  it("walks backward then forward through pred/succ links", () => {
    const targets = [
      { forward_pred: null, backward_succ: 1 },
      { forward_pred: 0, backward_succ: 2 },
      { forward_pred: 1, backward_succ: null },
    ];
    expect(criticalPathThrough(1, targets)).toEqual([0, 1, 2]);
  });

  it("returns single element when no pred or succ", () => {
    const targets = [{ forward_pred: null, backward_succ: null }];
    expect(criticalPathThrough(0, targets)).toEqual([0]);
  });
});

// ---------------------------------------------------------------------------
// searchFilter
// ---------------------------------------------------------------------------

describe("searchFilter", () => {
  const targets = [
    { name: "foo/lib" },
    { name: "bar/lib" },
    { name: "foo-bar/test" },
  ];

  it("returns all indices for empty query", () => {
    expect(searchFilter("", targets)).toEqual([0, 1, 2]);
  });

  it("filters by case-insensitive substring", () => {
    expect(searchFilter("foo", targets)).toEqual([0, 2]);
  });

  it("returns empty array when nothing matches", () => {
    expect(searchFilter("zzz", targets)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// slackColor
// ---------------------------------------------------------------------------

describe("slackColor", () => {
  it("returns red (hue 0) for zero slack", () => {
    expect(slackColor(0, 100)).toBe("hsl(0, 80%, 50%)");
  });

  it("returns blue (hue 220) for max slack", () => {
    expect(slackColor(100, 100)).toBe("hsl(220, 80%, 50%)");
  });

  it("handles zero maxSlack without division by zero", () => {
    expect(slackColor(0, 0)).toBe("hsl(0, 80%, 50%)");
  });
});

// ---------------------------------------------------------------------------
// pointInRect
// ---------------------------------------------------------------------------

describe("pointInRect", () => {
  const rect = { left: 10, right: 110, top: 20, bottom: 220 };

  it("returns true for points inside the rectangle", () => {
    expect(pointInRect(10, 20, rect)).toBe(true);
    expect(pointInRect(50, 100, rect)).toBe(true);
    expect(pointInRect(110, 220, rect)).toBe(true);
  });

  it("returns false for points outside the rectangle", () => {
    expect(pointInRect(9, 20, rect)).toBe(false);
    expect(pointInRect(111, 20, rect)).toBe(false);
    expect(pointInRect(10, 19, rect)).toBe(false);
    expect(pointInRect(10, 221, rect)).toBe(false);
  });
});
