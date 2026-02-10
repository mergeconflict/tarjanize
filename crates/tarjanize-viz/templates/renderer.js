// renderer.js — PixiJS v8 renderer for the tarjanize build schedule Gantt chart.
//
// Imports pure functions from logic.js and reads schedule data from
// window.DATA (injected by the HTML template). Execution starts
// immediately at module load time — no wrapping function.

import { Application, Container, Graphics, Text } from "pixi.js";
import {
  criticalPathThrough,
  slackColor,
  timeToX,
  laneToY,
  searchFilter,
  formatMs,
  niceTimeStep,
} from "./logic.js";

const DATA = window.DATA;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Left padding in the chart area to avoid bars starting at the very edge.
const CHART_PADDING_LEFT = 20;

// Right padding so the last bar doesn't hug the viewport boundary.
const CHART_PADDING_RIGHT = 40;

// Height in pixels reserved for the fixed time axis at the bottom.
const AXIS_HEIGHT = 30;

// Bar height as a fraction of lane height, leaving a small gap between lanes.
const BAR_HEIGHT_RATIO = 0.8;

// Alpha for non-critical-path bars when nothing is hovered.
const DEFAULT_ALPHA = 0.4;

// Alpha for bars on the critical path through the hovered target.
const HIGHLIGHT_ALPHA = 1.0;

// Alpha for dimmed bars during hover (not on the local critical path).
const DIM_ALPHA = 0.15;

// Alpha for one-hop dependency edges (not on critical path).
const EDGE_OFFPATH_ALPHA = 0.25;

// Minimum / maximum lane height bounds for the adaptive calculation.
const MIN_LANE_HEIGHT = 1;
const MAX_LANE_HEIGHT = 20;

// Zoom speed factor. Each wheel tick multiplies or divides the scale by this.
const ZOOM_FACTOR = 1.1;

// ---------------------------------------------------------------------------
// Derived values from DATA
// ---------------------------------------------------------------------------

const { summary, targets, critical_path } = DATA;

// Precompute max slack for color mapping. Clamp to avoid zero-division.
const maxSlack = targets.reduce((m, t) => Math.max(m, t.slack), 0);

// ---------------------------------------------------------------------------
// Populate sidebar stats
// ---------------------------------------------------------------------------

document.getElementById("stat-cp").textContent = formatMs(summary.critical_path_ms);
document.getElementById("stat-total").textContent = formatMs(summary.total_cost_ms);
document.getElementById("stat-parallel").textContent = summary.parallelism_ratio.toFixed(2) + "x";
document.getElementById("stat-targets").textContent = summary.target_count;
document.getElementById("stat-symbols").textContent = summary.symbol_count;
document.getElementById("stat-lanes").textContent = summary.lane_count;

// ---------------------------------------------------------------------------
// Create PixiJS application
// ---------------------------------------------------------------------------

const viewport = document.getElementById("viewport");
const app = new Application();
await app.init({
  resizeTo: viewport,
  background: 0x1a1a2e,
  antialias: true,
  autoDensity: true,
  resolution: window.devicePixelRatio || 1,
});
viewport.appendChild(app.canvas);

// ---------------------------------------------------------------------------
// Scene graph
// ---------------------------------------------------------------------------
//
// app.stage
//   chartContainer          — receives pan/zoom transforms
//     barsContainer         — one Graphics per target bar
//     labelsContainer       — Text objects, visibility-culled by zoom level
//   edgesContainer          — dependency edges in screen space (not panned/zoomed)
//   axisContainer           — fixed time axis at the bottom (not panned/zoomed)

const chartContainer = new Container();
chartContainer.cullable = true;
app.stage.addChild(chartContainer);

const barsContainer = new Container();
barsContainer.cullable = true;
chartContainer.addChild(barsContainer);

// Edges are drawn in screen space (not inside chartContainer) so stroke
// widths stay constant regardless of non-uniform zoom.
const edgesContainer = new Container();
edgesContainer.cullable = true;
app.stage.addChild(edgesContainer);

const labelsContainer = new Container();
labelsContainer.cullable = true;
chartContainer.addChild(labelsContainer);

// Axis container is a sibling of chartContainer (not affected by pan/zoom).
// Added after chartContainer so it renders on top.
const axisContainer = new Container();
app.stage.addChild(axisContainer);

// ---------------------------------------------------------------------------
// Layout parameters
// ---------------------------------------------------------------------------

// Adaptive lane height: fill the canvas (minus axis) but clamp to readable bounds.
const canvasHeight = app.screen.height - AXIS_HEIGHT;
let laneHeight = Math.max(
  MIN_LANE_HEIGHT,
  Math.min(MAX_LANE_HEIGHT, canvasHeight / summary.lane_count)
);

// Initial scale: fit the entire critical path into the viewport width
// with some padding.
const viewportWidth = app.screen.width;
let scale = summary.critical_path_ms > 0
  ? (viewportWidth - CHART_PADDING_LEFT - CHART_PADDING_RIGHT) / summary.critical_path_ms
  : 1;

// ---------------------------------------------------------------------------
// Parse HSL string to a PixiJS-compatible hex integer
// ---------------------------------------------------------------------------

// Convert an "hsl(h, s%, l%)" string to a 0xRRGGBB integer. PixiJS v8 fill
// accepts numeric colors. We do the conversion once per bar at creation time
// so hover/redraw can reuse the stored value.
function hslToHex(hslStr) {
  const match = hslStr.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
  if (!match) return 0xffffff;

  let h = parseInt(match[1]) / 360;
  const s = parseInt(match[2]) / 100;
  const l = parseInt(match[3]) / 100;

  // HSL to RGB conversion using the standard algorithm.
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }

  const toInt = (v) => Math.round(v * 255);
  return (toInt(r) << 16) | (toInt(g) << 8) | toInt(b);
}

// ---------------------------------------------------------------------------
// Build bars
// ---------------------------------------------------------------------------

// Store references so we can update alpha/tint on hover and search.
const barGraphics = [];
const barColors = [];
const barLabels = [];
const barClipMasks = [];

for (let i = 0; i < targets.length; i++) {
  const t = targets[i];
  const x = CHART_PADDING_LEFT + timeToX(t.start, scale);
  const y = laneToY(t.lane, laneHeight);
  const w = Math.max(1, timeToX(t.cost, scale));
  const h = laneHeight * BAR_HEIGHT_RATIO;

  const color = hslToHex(slackColor(t.slack, maxSlack));
  const alpha = t.on_critical_path ? HIGHLIGHT_ALPHA : DEFAULT_ALPHA;

  const bar = new Graphics();
  bar.rect(0, 0, w, h).fill({ color });
  bar.x = x;
  bar.y = y;
  bar.alpha = alpha;

  // Enable pointer events for hover interactions.
  bar.eventMode = "static";
  bar.cursor = "pointer";

  // Stash index for event handlers.
  bar.targetIndex = i;

  bar.on("pointerover", onBarPointerOver);
  bar.on("pointerout", onBarPointerOut);
  bar.on("pointermove", onBarPointerMove);

  barsContainer.addChild(bar);
  barGraphics.push(bar);
  barColors.push(color);

  // Create a text label clipped to the bar bounds. A separate Graphics
  // rectangle acts as the stencil mask (using the bar itself would hide it,
  // since PixiJS doesn't render mask sources).
  // Resolution 4 rasterizes the bitmap at 4x CSS pixels, keeping text
  // crisp up to 4x zoom.
  const clipMask = new Graphics();
  clipMask.rect(0, 0, w, h).fill({ color: 0xffffff });
  clipMask.x = x;
  clipMask.y = y;
  labelsContainer.addChild(clipMask);

  const label = new Text({
    text: t.name,
    style: { fontSize: 10, fill: 0xffffff, fontFamily: "monospace" },
    resolution: 4,
  });
  label.x = x + 3;
  label.y = y + (h - 10) / 2;
  label.visible = false;
  label.mask = clipMask;

  labelsContainer.addChild(label);
  barLabels.push(label);
  barClipMasks.push(clipMask);
}

// ---------------------------------------------------------------------------
// Label visibility (zoom-dependent)
// ---------------------------------------------------------------------------

// Minimum screen-space lane height (pixels) to show labels. Below this the
// font doesn't fit and adjacent-lane labels overlap vertically.
const LABEL_MIN_LANE_PX = 14;

// Show or hide labels based on lane height, and counter-scale them so text
// doesn't stretch with zoom. The labels live inside chartContainer (so their
// positions track the bars), but we set inverse scale on each label to keep
// the font at its natural pixel size. We also reposition labels so the
// padding from the bar edge stays constant in screen pixels. The clip masks
// are NOT counter-scaled — they match the bar bounds which stretch with zoom.
function updateLabels() {
  const scaleX = chartContainer.scale.x;
  const scaleY = chartContainer.scale.y;
  const screenLaneHeight = laneHeight * scaleY;
  const visible = screenLaneHeight > LABEL_MIN_LANE_PX;
  const h = laneHeight * BAR_HEIGHT_RATIO;

  for (let i = 0; i < targets.length; i++) {
    barLabels[i].visible = visible;
    barLabels[i].scale.x = 1 / scaleX;
    barLabels[i].scale.y = 1 / scaleY;
    // Reposition so padding from bar edge is constant in screen pixels.
    // 3 / scaleX world units = 3 screen pixels from the bar's left edge.
    barLabels[i].x = barGraphics[i].x + 3 / scaleX;
    barLabels[i].y = barGraphics[i].y + (h - 10 / scaleY) / 2;
  }
}

// Initial label check.
updateLabels();

// ---------------------------------------------------------------------------
// Time axis
// ---------------------------------------------------------------------------

// Redraw the fixed time axis at the bottom of the canvas. Called on every
// zoom, pan, and resize. The axis shows tick marks and time labels that
// adapt to the current horizontal zoom level.
function updateAxis() {
  axisContainer.removeChildren();

  const screenW = app.screen.width;
  const screenH = app.screen.height;

  // Background rectangle to occlude chart content scrolling underneath.
  const bg = new Graphics();
  bg.rect(0, screenH - AXIS_HEIGHT, screenW, AXIS_HEIGHT)
    .fill({ color: 0x16213e });
  axisContainer.addChild(bg);

  // Thin top border for visual separation.
  const border = new Graphics();
  border.setStrokeStyle({ width: 1, color: 0x333333 });
  border.moveTo(0, screenH - AXIS_HEIGHT);
  border.lineTo(screenW, screenH - AXIS_HEIGHT);
  border.stroke();
  axisContainer.addChild(border);

  // Effective pixels per millisecond at the current zoom level.
  const pixelsPerMs = scale * chartContainer.scale.x;
  const step = niceTimeStep(pixelsPerMs);

  // Convert screen edges to time coordinates to find the visible range.
  // worldX = (screenX - chartContainer.x) / chartContainer.scale.x
  // timeMs = (worldX - CHART_PADDING_LEFT) / scale
  const timeAtLeft = ((0 - chartContainer.x) / chartContainer.scale.x - CHART_PADDING_LEFT) / scale;
  const timeAtRight = ((screenW - chartContainer.x) / chartContainer.scale.x - CHART_PADDING_LEFT) / scale;

  // Clamp start to >= 0 (no negative times).
  const firstTick = Math.max(0, Math.ceil(timeAtLeft / step)) * step;

  const tickG = new Graphics();
  tickG.setStrokeStyle({ width: 1, color: 0x666666 });

  for (let t = firstTick; t <= timeAtRight; t += step) {
    // Convert time back to screen X.
    const worldX = CHART_PADDING_LEFT + timeToX(t, scale);
    const screenX = worldX * chartContainer.scale.x + chartContainer.x;

    // 6px tick mark.
    tickG.moveTo(screenX, screenH - AXIS_HEIGHT);
    tickG.lineTo(screenX, screenH - AXIS_HEIGHT + 6);

    // Time label.
    const label = new Text({
      text: formatMs(t),
      style: { fontSize: 10, fill: 0x999999, fontFamily: "monospace" },
    });
    label.x = screenX + 3;
    label.y = screenH - AXIS_HEIGHT + 6;
    axisContainer.addChild(label);
  }

  tickG.stroke();
  axisContainer.addChild(tickG);
}

// Initial axis draw.
updateAxis();

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

const tooltipEl = document.getElementById("tooltip");

// Show the tooltip with details for target at `idx`.
function showTooltip(idx, globalX, globalY) {
  const t = targets[idx];
  tooltipEl.innerHTML = `
    <div class="tt-name">${t.name}</div>
    <div class="tt-row"><span>Cost:</span><span>${formatMs(t.cost)}</span></div>
    <div class="tt-row"><span>Start:</span><span>${formatMs(t.start)}</span></div>
    <div class="tt-row"><span>Finish:</span><span>${formatMs(t.finish)}</span></div>
    <div class="tt-row"><span>Slack:</span><span>${formatMs(t.slack)}</span></div>
    <div class="tt-row"><span>Symbols:</span><span>${t.symbol_count}</span></div>
    <div class="tt-row"><span>Deps:</span><span>${t.deps.length}</span></div>
    <div class="tt-row"><span>Dependents:</span><span>${t.dependents.length}</span></div>
  `;
  tooltipEl.style.display = "block";
  positionTooltip(globalX, globalY);
}

// Position the tooltip near the cursor, keeping it within the viewport.
function positionTooltip(x, y) {
  const pad = 12;
  const rect = tooltipEl.getBoundingClientRect();
  let left = x + pad;
  let top = y + pad;

  // Prevent overflow on the right.
  if (left + rect.width > window.innerWidth) {
    left = x - rect.width - pad;
  }
  // Prevent overflow at the bottom.
  if (top + rect.height > window.innerHeight) {
    top = y - rect.height - pad;
  }

  tooltipEl.style.left = `${left}px`;
  tooltipEl.style.top = `${top}px`;
}

function hideTooltip() {
  tooltipEl.style.display = "none";
}

// ---------------------------------------------------------------------------
// Edge drawing
// ---------------------------------------------------------------------------

// Draw dependency edges when a bar is hovered. Edges on the critical path
// through the hovered target are solid and bright; one-hop deps/dependents
// are dashed and faint.
function drawEdges(idx) {
  // Clear previous edges.
  edgesContainer.removeChildren();

  const cpSet = new Set(criticalPathThrough(idx, targets));
  const t = targets[idx];

  // Collect edges to draw: critical path edges first, then one-hop.
  // Critical path edges connect consecutive nodes in the cpSet chain.
  const cpArray = criticalPathThrough(idx, targets);
  for (let i = 0; i < cpArray.length - 1; i++) {
    drawEdge(cpArray[i], cpArray[i + 1], true);
  }

  // One-hop deps that aren't on the critical path.
  for (const dep of t.deps) {
    if (!cpSet.has(dep)) {
      drawEdge(dep, idx, false);
    }
  }

  // One-hop dependents that aren't on the critical path.
  for (const dependent of t.dependents) {
    if (!cpSet.has(dependent)) {
      drawEdge(idx, dependent, false);
    }
  }
}

// Convert a world-space coordinate to screen space using the current
// chartContainer transform.
function worldToScreen(wx, wy) {
  return {
    x: wx * chartContainer.scale.x + chartContainer.x,
    y: wy * chartContainer.scale.y + chartContainer.y,
  };
}

// Draw a single directed edge from source to destination target.
//
// Edges are drawn in screen space (edgesContainer is outside chartContainer)
// so stroke widths stay constant regardless of zoom. Critical path edges use
// a solid bezier curve; off-path edges use a dashed straight line.
function drawEdge(fromIdx, toIdx, isCritical) {
  const from = targets[fromIdx];
  const to = targets[toIdx];

  // Source: right edge of the from-bar, vertically centered (world space).
  const w1 = worldToScreen(
    CHART_PADDING_LEFT + timeToX(from.finish, scale),
    laneToY(from.lane, laneHeight) + (laneHeight * BAR_HEIGHT_RATIO) / 2,
  );

  // Destination: left edge of the to-bar, vertically centered (world space).
  const w2 = worldToScreen(
    CHART_PADDING_LEFT + timeToX(to.start, scale),
    laneToY(to.lane, laneHeight) + (laneHeight * BAR_HEIGHT_RATIO) / 2,
  );

  const x1 = w1.x, y1 = w1.y, x2 = w2.x, y2 = w2.y;

  const g = new Graphics();
  const color = isCritical ? 0x00d4ff : 0x888888;
  const alpha = isCritical ? 0.8 : EDGE_OFFPATH_ALPHA;

  if (isCritical) {
    // Solid bezier curve for critical path edges. The control points
    // create a gentle S-curve that looks cleaner than a straight line
    // when lanes differ.
    const cpx = (x1 + x2) / 2;
    g.setStrokeStyle({ width: 2, color, alpha });
    g.moveTo(x1, y1);
    g.bezierCurveTo(cpx, y1, cpx, y2, x2, y2);
    g.stroke();
  } else {
    // Dashed line for one-hop off-path edges. PixiJS v8 doesn't have
    // native dash support, so we approximate with short segments.
    const dashLen = 6;
    const gapLen = 4;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 1) {
      edgesContainer.addChild(g);
      return;
    }
    const ux = dx / dist;
    const uy = dy / dist;

    g.setStrokeStyle({ width: 1, color, alpha });
    let d = 0;
    while (d < dist) {
      const segEnd = Math.min(d + dashLen, dist);
      g.moveTo(x1 + ux * d, y1 + uy * d);
      g.lineTo(x1 + ux * segEnd, y1 + uy * segEnd);
      d = segEnd + gapLen;
    }
    g.stroke();
  }

  edgesContainer.addChild(g);
}

function clearEdges() {
  edgesContainer.removeChildren();
}

// ---------------------------------------------------------------------------
// Hover handlers
// ---------------------------------------------------------------------------

// Track whether we're currently hovering (to avoid conflicts with search).
let hoveredIdx = null;

function onBarPointerOver(event) {
  const idx = event.currentTarget.targetIndex;
  hoveredIdx = idx;

  // Compute the critical path through this target.
  const cp = new Set(criticalPathThrough(idx, targets));

  // Highlight critical-path bars, dim everything else.
  for (let i = 0; i < barGraphics.length; i++) {
    if (cp.has(i)) {
      barGraphics[i].alpha = HIGHLIGHT_ALPHA;
    } else {
      barGraphics[i].alpha = DIM_ALPHA;
    }
  }

  // Draw dependency edges.
  drawEdges(idx);

  // Show tooltip.
  const global = event.global;
  showTooltip(idx, global.x + 250, global.y);
}

function onBarPointerMove(event) {
  if (hoveredIdx == null) return;
  const global = event.global;
  positionTooltip(global.x + 250, global.y);
}

function onBarPointerOut() {
  hoveredIdx = null;

  // Restore default alpha values.
  for (let i = 0; i < barGraphics.length; i++) {
    barGraphics[i].alpha = targets[i].on_critical_path
      ? HIGHLIGHT_ALPHA
      : DEFAULT_ALPHA;
  }

  // Apply any active search filter on top of defaults.
  applySearchHighlight();

  clearEdges();
  hideTooltip();
}

// ---------------------------------------------------------------------------
// Pan and zoom
// ---------------------------------------------------------------------------

// Track drag state for panning.
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let containerStartX = 0;
let containerStartY = 0;

// Use the viewport div for pointer events (PixiJS canvas is inside it).
viewport.addEventListener("pointerdown", (e) => {
  // Only pan on the canvas background (not on bars, which handle their own
  // events). We detect this by checking if the target is the canvas element
  // or the viewport itself.
  isDragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  containerStartX = chartContainer.x;
  containerStartY = chartContainer.y;
});

window.addEventListener("pointermove", (e) => {
  if (!isDragging) return;
  const dx = e.clientX - dragStartX;
  const dy = e.clientY - dragStartY;
  chartContainer.x = containerStartX + dx;
  chartContainer.y = containerStartY + dy;
  updateLabels();
  updateAxis();
  if (hoveredIdx != null) drawEdges(hoveredIdx);
});

window.addEventListener("pointerup", () => {
  isDragging = false;
});

// Wheel zoom: horizontal zoom on plain wheel, vertical zoom on Shift+wheel.
//
// The math: we want the world-space point under the cursor to stay fixed
// after the scale change. We compute the world point from the cursor's
// screen position, update the scale, then recompute the container's
// position so the world point maps back to the same screen position.
viewport.addEventListener("wheel", (e) => {
  e.preventDefault();

  const direction = e.deltaY < 0 ? 1 : -1;
  const factor = direction > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;

  // Cursor position relative to the viewport.
  const rect = viewport.getBoundingClientRect();

  if (e.shiftKey) {
    // Shift+wheel: vertical zoom centered on cursor Y.
    const cursorY = e.clientY - rect.top;
    const worldY = (cursorY - chartContainer.y) / chartContainer.scale.y;
    chartContainer.scale.y = Math.max(1, chartContainer.scale.y * factor);
    chartContainer.y = cursorY - worldY * chartContainer.scale.y;
  } else {
    // Plain wheel: horizontal zoom centered on cursor X.
    const cursorX = e.clientX - rect.left;
    const worldX = (cursorX - chartContainer.x) / chartContainer.scale.x;
    chartContainer.scale.x = Math.max(1, chartContainer.scale.x * factor);
    chartContainer.x = cursorX - worldX * chartContainer.scale.x;
  }

  updateLabels();
  updateAxis();
  if (hoveredIdx != null) drawEdges(hoveredIdx);
}, { passive: false });

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

// Track the current search match set. null means no active filter.
let searchMatches = null;

const searchInput = document.getElementById("search-input");
searchInput.addEventListener("input", () => {
  const query = searchInput.value.trim();
  if (query === "") {
    searchMatches = null;
  } else {
    searchMatches = new Set(searchFilter(query, targets));
  }
  applySearchHighlight();
});

// Apply search highlighting: matching bars stay at normal alpha, non-matching
// bars are dimmed. If no search is active, restore defaults.
function applySearchHighlight() {
  // Don't override hover highlighting.
  if (hoveredIdx != null) return;

  if (searchMatches == null) {
    // No active search — restore defaults.
    for (let i = 0; i < barGraphics.length; i++) {
      barGraphics[i].alpha = targets[i].on_critical_path
        ? HIGHLIGHT_ALPHA
        : DEFAULT_ALPHA;
    }
  } else {
    for (let i = 0; i < barGraphics.length; i++) {
      if (searchMatches.has(i)) {
        barGraphics[i].alpha = HIGHLIGHT_ALPHA;
      } else {
        barGraphics[i].alpha = DIM_ALPHA;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Resize handling
// ---------------------------------------------------------------------------

// Recompute lane height and bar positions when the window resizes.
window.addEventListener("resize", () => {
  laneHeight = Math.max(
    MIN_LANE_HEIGHT,
    Math.min(MAX_LANE_HEIGHT, (app.screen.height - AXIS_HEIGHT) / summary.lane_count)
  );
  rebuildBars();
  updateLabels();
  updateAxis();
});

// Rebuild all bar positions and sizes after layout parameters change.
// This is cheaper than recreating Graphics objects — we just clear and
// redraw the rect in each existing Graphics.
function rebuildBars() {
  for (let i = 0; i < targets.length; i++) {
    const t = targets[i];
    const x = CHART_PADDING_LEFT + timeToX(t.start, scale);
    const y = laneToY(t.lane, laneHeight);
    const w = Math.max(1, timeToX(t.cost, scale));
    const h = laneHeight * BAR_HEIGHT_RATIO;

    const bar = barGraphics[i];
    bar.clear();
    bar.rect(0, 0, w, h).fill({ color: barColors[i] });
    bar.x = x;
    bar.y = y;

    const label = barLabels[i];
    label.x = x + 3;
    label.y = y + (h - 10) / 2;

    const mask = barClipMasks[i];
    mask.clear();
    mask.rect(0, 0, w, h).fill({ color: 0xffffff });
    mask.x = x;
    mask.y = y;
  }
}
