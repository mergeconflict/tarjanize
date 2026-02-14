// logic.ts — Pure functions for the tarjanize build schedule visualization.
//
// This is a proper ES module. The renderer imports these functions, and they
// can be tested directly with Vitest without any DOM or PixiJS dependency.

import type { TargetData } from "../../../types/generated/TargetData";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Zero constant for comparisons and fallback values, avoiding the
// no-magic-numbers lint on bare `0` literals.
const ZERO = 0;

// Offset from the end of an array to reach the last element.
const LAST_INDEX_OFFSET = 1;

// Hue range for slack-based coloring: 0 (red, critical) to 220 (blue, slack).
const MIN_SLACK_HUE = 0;
const MAX_SLACK_HUE = 220;

// Duration thresholds for human-readable formatting.
const MS_PER_SECOND = 1000;
const SECONDS_PER_MINUTE = 60;

// Number of decimal places shown for sub-minute durations (e.g., "12.3s").
const SECONDS_DECIMAL_PLACES = 1;

// Default minimum pixel spacing between time-axis ticks.
const DEFAULT_TICK_SPACING_PX = 100;

// Pre-computed list of "nice" round tick intervals in milliseconds, covering
// sub-millisecond through 10-minute durations — enough for any realistic
// build schedule. Each value is inherently a named constant by its position
// in the table, so we suppress the magic-number lint for this declaration.
/* eslint-disable @typescript-eslint/no-magic-numbers -- tick interval lookup table */
const NICE_TICK_STEPS: readonly number[] = [
  1, 2, 5, 10, 20, 50, 100, 200, 500,
  1000, 2000, 5000, 10_000, 15_000, 30_000,
  60_000, 120_000, 300_000, 600_000,
] as const;
/* eslint-enable @typescript-eslint/no-magic-numbers */

// Index of the last element in the tick steps array, used as the fallback
// when no step is large enough.
const LAST_STEP_INDEX = NICE_TICK_STEPS.length - LAST_INDEX_OFFSET;

// ---------------------------------------------------------------------------
// Critical-path walk
// ---------------------------------------------------------------------------

// Build the critical path that passes through target `index`.
//
// Walk backward via `forward_pred` to find the earliest ancestor on the
// longest path TO this target, then walk forward via `backward_succ` to
// find the latest descendant on the longest path FROM this target. The
// union is the full critical path through this single target.
//
// Returns an array of target indices in start-time order.
export function criticalPathThrough(index: number, targets: TargetData[]): number[] {
  // Walk backward from index through forward_pred to collect ancestors.
  const backward: number[] = [];
  let current: number | null | undefined = index;
  while (typeof current === "number") {
    backward.push(current);
    const target: TargetData = targets[current];
    ({ forward_pred: current } = target);
  }
  backward.reverse();

  // Walk forward from index through backward_succ to collect descendants.
  // Skip index itself to avoid duplication — it's already the last element
  // of `backward`.
  const startTarget: TargetData = targets[index];
  ({ backward_succ: current } = startTarget);
  while (typeof current === "number") {
    backward.push(current);
    const target: TargetData = targets[current];
    ({ backward_succ: current } = target);
  }

  return backward;
}

// ---------------------------------------------------------------------------
// Color mapping
// ---------------------------------------------------------------------------

// Map a slack value to an HSL color string.
//
// The visual encoding uses hue to communicate how close a target is to the
// critical path: zero slack (fully critical) maps to red (hue 0), and the
// maximum slack in the schedule maps to blue (hue 220). Intermediate values
// interpolate linearly, producing an intuitive red-through-blue gradient.
export function slackColor(slack: number, maxSlack: number): string {
  // Protect against division by zero when all targets have equal slack.
  const ratio = maxSlack > MIN_SLACK_HUE ? slack / maxSlack : MIN_SLACK_HUE;
  const hue = Math.round(ratio * MAX_SLACK_HUE);
  return `hsl(${String(hue)}, 80%, 50%)`;
}

// ---------------------------------------------------------------------------
// Coordinate transforms
// ---------------------------------------------------------------------------

// Convert a time value in milliseconds to a horizontal pixel coordinate.
//
// The `scale` parameter represents pixels per millisecond. It changes as the
// user zooms in/out. Keeping this as a pure function (rather than reading
// global state) makes the renderer easier to reason about.
export function timeToX(timeMs: number, scale: number): number {
  return timeMs * scale;
}

// Convert a swim-lane index to a vertical pixel coordinate.
//
// Lane 0 is at the top of the chart. Each lane occupies `laneHeight` pixels,
// producing a compact Gantt-style stacking.
export function laneToY(lane: number, laneHeight: number): number {
  return lane * laneHeight;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

// Filter targets by case-insensitive substring match on their name.
//
// Returns an array of indices into `targets` for every target whose
// name contains the query string. An empty query returns all indices (i.e.,
// no filtering).
export function searchFilter(query: string, targets: TargetData[]): number[] {
  if (query === "") {
    return targets.map((_: TargetData, index: number) => index);
  }
  const lower = query.toLowerCase();
  const results: number[] = [];
  for (const [index, target] of targets.entries()) {
    if (target.name.toLowerCase().includes(lower)) {
      results.push(index);
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

// Format a duration in milliseconds as a human-readable string.
//
// Picks the most natural unit:
//   - Under 1 second: "500ms"
//   - Under 1 minute: "12.3s"
//   - 1 minute or more: "1m 10s"
// This keeps the sidebar and tooltips readable without excessive precision.
export function formatMs(ms: number): string {
  if (ms < MS_PER_SECOND) {
    return `${String(Math.round(ms))}ms`;
  }
  const totalSeconds = ms / MS_PER_SECOND;
  if (totalSeconds < SECONDS_PER_MINUTE) {
    // Show one decimal place for sub-minute durations.
    return `${totalSeconds.toFixed(SECONDS_DECIMAL_PLACES)}s`;
  }
  const minutes = Math.floor(totalSeconds / SECONDS_PER_MINUTE);
  const seconds = Math.round(totalSeconds % SECONDS_PER_MINUTE);
  if (seconds === ZERO) {
    return `${String(minutes)}m`;
  }
  return `${String(minutes)}m ${String(seconds)}s`;
}

// ---------------------------------------------------------------------------
// Time axis
// ---------------------------------------------------------------------------

// Choose a "nice" round tick interval for the time axis at the current zoom.
//
// Given the effective pixels-per-millisecond, this picks the smallest interval
// from a predetermined list of human-friendly values that keeps ticks at
// least `targetPx` pixels apart. The list covers sub-millisecond through
// 10-minute intervals — enough for any realistic build schedule.
export function niceTimeStep(pixelsPerMs: number, targetPx?: number): number {
  const minSpacing = targetPx ?? DEFAULT_TICK_SPACING_PX;
  const msPerTarget = minSpacing / pixelsPerMs;
  for (const step of NICE_TICK_STEPS) {
    if (step >= msPerTarget) return step;
  }
  return NICE_TICK_STEPS[LAST_STEP_INDEX];
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

// Boundary rectangle with left/right/top/bottom edges, matching the shape
// returned by `getBoundingClientRect()`.
interface Rect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

// Check whether a point lies within a DOMRect-like boundary.
//
// Used by the renderer to ignore hover events when the pointer is
// outside the chart viewport (e.g., over the sidebar).
export function pointInRect(x: number, y: number, rect: Rect): boolean {
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}
