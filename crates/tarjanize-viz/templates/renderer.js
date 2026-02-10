// renderer.js — PixiJS v8 renderer for the tarjanize build schedule Gantt chart.
//
// This file is included after logic.js inside the same <script type="module">
// block, so all logic.js functions (criticalPathThrough, slackColor, timeToX,
// laneToY, searchFilter, formatMs) are available as bare identifiers. The
// `DATA` constant and pixi.js imports are also already in scope.
//
// Execution starts immediately at module load time — no wrapping function.

import { Application, Container, Graphics, Text } from "pixi.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Left padding in the chart area to avoid bars starting at the very edge.
const CHART_PADDING_LEFT = 20;

// Right padding so the last bar doesn't hug the viewport boundary.
const CHART_PADDING_RIGHT = 40;

// Minimum bar width (pixels) before we create a text label for it. Below
// this threshold labels would overlap or be unreadable.
const LABEL_MIN_WIDTH_PX = 60;

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
//     edgesContainer        — dependency edges, redrawn on hover
//     labelsContainer       — Text objects, visibility-culled by zoom level

const chartContainer = new Container();
chartContainer.cullable = true;
app.stage.addChild(chartContainer);

const barsContainer = new Container();
barsContainer.cullable = true;
chartContainer.addChild(barsContainer);

const edgesContainer = new Container();
edgesContainer.cullable = true;
chartContainer.addChild(edgesContainer);

const labelsContainer = new Container();
labelsContainer.cullable = true;
chartContainer.addChild(labelsContainer);

// ---------------------------------------------------------------------------
// Layout parameters
// ---------------------------------------------------------------------------

// Adaptive lane height: fill the canvas but clamp to readable bounds.
const canvasHeight = app.screen.height;
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

  // Create a text label (initially hidden; visibility managed by updateLabels).
  const label = new Text({
    text: t.name,
    style: { fontSize: 10, fill: 0xffffff, fontFamily: "monospace" },
  });
  label.x = x + 3;
  label.y = y + (h - 10) / 2;
  label.visible = false;
  labelsContainer.addChild(label);
  barLabels.push(label);
}

// ---------------------------------------------------------------------------
// Label visibility (zoom-dependent)
// ---------------------------------------------------------------------------

// Only show labels when the bar is wide enough on screen. Called after every
// zoom or pan to keep the display clean.
function updateLabels() {
  for (let i = 0; i < targets.length; i++) {
    const screenWidth = timeToX(targets[i].cost, scale) * chartContainer.scale.x;
    barLabels[i].visible = screenWidth > LABEL_MIN_WIDTH_PX;
  }
}

// Initial label check.
updateLabels();

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
    <div class="tt-row"><span>Critical:</span><span>${t.on_critical_path ? "yes" : "no"}</span></div>
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

  const cpSet = new Set(criticalPathThrough(idx));
  const t = targets[idx];

  // Collect edges to draw: critical path edges first, then one-hop.
  // Critical path edges connect consecutive nodes in the cpSet chain.
  const cpArray = criticalPathThrough(idx);
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

// Draw a single directed edge from source to destination target.
//
// Critical path edges use a solid bezier curve at full alpha. Off-path
// edges use a dashed straight line at reduced alpha to avoid visual clutter.
function drawEdge(fromIdx, toIdx, isCritical) {
  const from = targets[fromIdx];
  const to = targets[toIdx];

  // Source: right edge of the from-bar, vertically centered.
  const x1 = CHART_PADDING_LEFT + timeToX(from.finish, scale);
  const y1 = laneToY(from.lane, laneHeight) + (laneHeight * BAR_HEIGHT_RATIO) / 2;

  // Destination: left edge of the to-bar, vertically centered.
  const x2 = CHART_PADDING_LEFT + timeToX(to.start, scale);
  const y2 = laneToY(to.lane, laneHeight) + (laneHeight * BAR_HEIGHT_RATIO) / 2;

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
  const cp = new Set(criticalPathThrough(idx));

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
});

window.addEventListener("pointerup", () => {
  isDragging = false;
});

// Wheel zoom: scale the chartContainer around the cursor position.
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
  const cursorX = e.clientX - rect.left;
  const cursorY = e.clientY - rect.top;

  // World-space point under cursor before zoom.
  const worldX = (cursorX - chartContainer.x) / chartContainer.scale.x;
  const worldY = (cursorY - chartContainer.y) / chartContainer.scale.y;

  // Apply zoom, clamping so we never zoom out past the initial 1:1 view.
  chartContainer.scale.x = Math.max(1, chartContainer.scale.x * factor);
  chartContainer.scale.y = Math.max(1, chartContainer.scale.y * factor);

  // Reposition so the world point stays under the cursor.
  chartContainer.x = cursorX - worldX * chartContainer.scale.x;
  chartContainer.y = cursorY - worldY * chartContainer.scale.y;

  updateLabels();
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
    searchMatches = new Set(searchFilter(query));
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
    Math.min(MAX_LANE_HEIGHT, app.screen.height / summary.lane_count)
  );
  rebuildBars();
  updateLabels();
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
  }
}
