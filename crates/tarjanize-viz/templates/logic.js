// logic.js — Pure functions for the tarjanize build schedule visualization.
//
// This is a proper ES module. The renderer imports these functions, and they
// can be tested directly with Vitest without any DOM or PixiJS dependency.

// ---------------------------------------------------------------------------
// Critical-path walk
// ---------------------------------------------------------------------------

// Build the critical path that passes through target `idx`.
//
// Walk backward via `forward_pred` to find the earliest ancestor on the
// longest path TO this target, then walk forward via `backward_succ` to
// find the latest descendant on the longest path FROM this target. The
// union is the full critical path through this single target.
//
// Returns an array of target indices in start-time order.
export function criticalPathThrough(idx, targets) {
  // Walk backward from idx through forward_pred to collect ancestors.
  const backward = [];
  let cur = idx;
  while (cur != null) {
    backward.push(cur);
    cur = targets[cur].forward_pred;
  }
  backward.reverse();

  // Walk forward from idx through backward_succ to collect descendants.
  // Skip idx itself to avoid duplication — it's already the last element
  // of `backward`.
  cur = targets[idx].backward_succ;
  while (cur != null) {
    backward.push(cur);
    cur = targets[cur].backward_succ;
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
export function slackColor(slack, maxSlack) {
  // Protect against division by zero when all targets have equal slack.
  const ratio = maxSlack > 0 ? slack / maxSlack : 0;
  const hue = Math.round(ratio * 220);
  return `hsl(${hue}, 80%, 50%)`;
}

// ---------------------------------------------------------------------------
// Coordinate transforms
// ---------------------------------------------------------------------------

// Convert a time value in milliseconds to a horizontal pixel coordinate.
//
// The `scale` parameter represents pixels per millisecond. It changes as the
// user zooms in/out. Keeping this as a pure function (rather than reading
// global state) makes the renderer easier to reason about.
export function timeToX(timeMs, scale) {
  return timeMs * scale;
}

// Convert a swim-lane index to a vertical pixel coordinate.
//
// Lane 0 is at the top of the chart. Each lane occupies `laneHeight` pixels,
// producing a compact Gantt-style stacking.
export function laneToY(lane, laneHeight) {
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
export function searchFilter(query, targets) {
  if (!query) {
    return targets.map((_, i) => i);
  }
  const lower = query.toLowerCase();
  const results = [];
  for (let i = 0; i < targets.length; i++) {
    if (targets[i].name.toLowerCase().includes(lower)) {
      results.push(i);
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
export function formatMs(ms) {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  const totalSeconds = ms / 1000;
  if (totalSeconds < 60) {
    // Show one decimal place for sub-minute durations.
    return `${totalSeconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  if (seconds === 0) {
    return `${minutes}m`;
  }
  return `${minutes}m ${seconds}s`;
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
export function niceTimeStep(pixelsPerMs, targetPx) {
  if (!targetPx) targetPx = 100;
  const msPerTarget = targetPx / pixelsPerMs;
  const steps = [
    1, 2, 5, 10, 20, 50, 100, 200, 500,
    1000, 2000, 5000, 10000, 15000, 30000,
    60000, 120000, 300000, 600000,
  ];
  for (const s of steps) {
    if (s >= msPerTarget) return s;
  }
  return steps[steps.length - 1];
}
