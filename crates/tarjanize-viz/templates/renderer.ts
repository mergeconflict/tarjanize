// renderer.ts — PixiJS v8 Gantt chart renderer for the build schedule.
//
// Entry point for the renderer ESM bundle. Builds the chart from
// globalThis.DATA, handles interactions (hover, click, pan, zoom,
// search), and exposes globalThis.updateSchedule for live data swaps.

/* eslint-disable @typescript-eslint/prefer-destructuring -- false positives on already-destructured code */
/* eslint-disable @typescript-eslint/init-declarations -- undefined sentinels need uninitialized declarations */

import { Application, Container, Graphics, Text } from "pixi.js";
import type { FederatedPointerEvent } from "pixi.js";
import {
  criticalPathThrough, slackColor, timeToX, laneToY,
  searchFilter, formatMs, niceTimeStep, pointInRect,
} from "./logic.ts";
import {
  CHART_PADDING_LEFT, CHART_PADDING_RIGHT, AXIS_HEIGHT, BAR_HEIGHT_RATIO,
  MIN_LANE_HEIGHT, MAX_LANE_HEIGHT, LABEL_MIN_LANE_PX, LABEL_PAD_PX,
  FONT_SIZE, TEXT_RESOLUTION, PARALLELISM_DECIMALS, TICK_HEIGHT,
  DEFAULT_ALPHA, HIGHLIGHT_ALPHA, DIM_ALPHA, ZOOM_FACTOR, MIN_SCALE,
  COLOR_BACKGROUND, COLOR_WHITE, COLOR_AXIS_BG, COLOR_AXIS_BORDER,
  COLOR_AXIS_TICK, COLOR_AXIS_LABEL, THIN_STROKE_WIDTH,
  HALF_DIVISOR, ORIGIN, UNIT, NEGATIVE_UNIT, ZOOM_DIRECTION_POSITIVE,
} from "./constants.ts";
import { hslToHex } from "./color.ts";
import { showTooltip, positionTooltip, hideTooltip } from "./tooltip.ts";
import { drawEdges, clearEdges, type EdgeContext } from "./edges.ts";
import type { ScheduleData } from "../../../types/generated/ScheduleData";

interface BarGraphics extends Graphics { targetIndex: number }

function isBarGraphics(value: unknown): value is BarGraphics {
  return value instanceof Graphics && "targetIndex" in value;
}

let { targets, summary } = globalThis.DATA;
let barGraphics: Graphics[] = [];
let barColors: number[] = [];
let barLabels: Text[] = [];
let barClipMasks: Graphics[] = [];
let laneHeight = UNIT;
let scale = UNIT;
let hoveredIndex: number | undefined;
let searchMatches: Set<number> | undefined;

globalThis.formatMs = formatMs;

// ---------------------------------------------------------------------------
// Sidebar stats
// ---------------------------------------------------------------------------

function populateSidebarStats(): void {
  const statCp = document.querySelector("#stat-cp");
  const statTotal = document.querySelector("#stat-total");
  const statParallel = document.querySelector("#stat-parallel");
  const statTargets = document.querySelector("#stat-targets");
  const statSymbols = document.querySelector("#stat-symbols");
  const statLanes = document.querySelector("#stat-lanes");
  if (statCp !== null) { statCp.textContent = formatMs(summary.critical_path_ms); }
  if (statTotal !== null) { statTotal.textContent = formatMs(summary.total_cost_ms); }
  if (statParallel !== null) { statParallel.textContent = `${summary.parallelism_ratio.toFixed(PARALLELISM_DECIMALS)}x`; }
  if (statTargets !== null) { statTargets.textContent = String(summary.target_count); }
  if (statSymbols !== null) { statSymbols.textContent = String(summary.symbol_count); }
  if (statLanes !== null) { statLanes.textContent = String(summary.lane_count); }
}
populateSidebarStats();

// ---------------------------------------------------------------------------
// PixiJS application and containers
// ---------------------------------------------------------------------------

const viewport = document.querySelector<HTMLElement>("#viewport");
const app = new Application();
await app.init({
  resizeTo: viewport ?? undefined,
  background: COLOR_BACKGROUND,
  antialias: true,
  autoDensity: true,
  resolution: globalThis.devicePixelRatio === ORIGIN ? UNIT : globalThis.devicePixelRatio,
});
if (viewport !== null) { viewport.append(app.canvas); }

const chartContainer = new Container();
chartContainer.cullable = true;
app.stage.addChild(chartContainer);
const barsContainer = new Container();
barsContainer.cullable = true;
chartContainer.addChild(barsContainer);
const edgesContainer = new Container();
edgesContainer.cullable = true;
app.stage.addChild(edgesContainer);
const labelsContainer = new Container();
labelsContainer.cullable = true;
chartContainer.addChild(labelsContainer);
const axisContainer = new Container();
app.stage.addChild(axisContainer);

function edgeContext(): EdgeContext {
  return { edgesContainer, chartContainer, targets, scale, laneHeight };
}

// ---------------------------------------------------------------------------
// Chart building
// ---------------------------------------------------------------------------

function buildChart(): void {
  ({ targets, summary } = globalThis.DATA);
  barsContainer.removeChildren();
  labelsContainer.removeChildren();
  edgesContainer.removeChildren();
  axisContainer.removeChildren();
  barGraphics = [];
  barColors = [];
  barLabels = [];
  barClipMasks = [];
  hoveredIndex = undefined;
  searchMatches = undefined;
  let maxSlack = ORIGIN;
  for (const target of targets) { maxSlack = Math.max(maxSlack, target.slack); }
  const canvasHeight = app.screen.height - AXIS_HEIGHT;
  laneHeight = Math.max(MIN_LANE_HEIGHT, Math.min(MAX_LANE_HEIGHT, canvasHeight / summary.lane_count));
  const { width: viewportWidth } = app.screen;
  scale = summary.critical_path_ms > ORIGIN
    ? (viewportWidth - CHART_PADDING_LEFT - CHART_PADDING_RIGHT) / summary.critical_path_ms : UNIT;
  chartContainer.scale.set(UNIT, UNIT);
  chartContainer.position.set(ORIGIN, ORIGIN);
  for (const [index, t] of targets.entries()) {
    const x = CHART_PADDING_LEFT + timeToX(t.start, scale);
    const y = laneToY(t.lane, laneHeight);
    const w = timeToX(t.cost, scale);
    const h = laneHeight * BAR_HEIGHT_RATIO;
    const color = hslToHex(slackColor(t.slack, maxSlack));
    const alpha = t.on_critical_path ? HIGHLIGHT_ALPHA : DEFAULT_ALPHA;
    const bar: BarGraphics = Object.assign(new Graphics(), { targetIndex: index });
    bar.rect(ORIGIN, ORIGIN, w, h).fill({ color });
    bar.x = x;
    bar.y = y;
    bar.alpha = alpha;
    bar.eventMode = "static";
    bar.cursor = "pointer";
    bar.on("pointerover", onBarPointerOver);
    bar.on("pointerout", onBarPointerOut);
    bar.on("pointermove", onBarPointerMove);
    bar.on("pointertap", onBarClick);
    barsContainer.addChild(bar);
    barGraphics.push(bar);
    barColors.push(color);
    const clipMask = new Graphics();
    clipMask.rect(ORIGIN, ORIGIN, w, h).fill({ color: COLOR_WHITE });
    clipMask.x = x;
    clipMask.y = y;
    labelsContainer.addChild(clipMask);
    const label = new Text({
      text: t.name,
      style: { fontSize: FONT_SIZE, fill: COLOR_WHITE, fontFamily: "monospace" },
      resolution: TEXT_RESOLUTION,
    });
    label.x = x + LABEL_PAD_PX;
    label.y = y + (h - FONT_SIZE) / HALF_DIVISOR;
    label.visible = false;
    label.mask = clipMask;
    labelsContainer.addChild(label);
    barLabels.push(label);
    barClipMasks.push(clipMask);
  }
  updateLabels();
  updateAxis();
}

function updateLabels(): void {
  const { x: scaleX, y: scaleY } = chartContainer.scale;
  const screenLaneHeight = laneHeight * scaleY;
  const visible = screenLaneHeight > LABEL_MIN_LANE_PX;
  const h = laneHeight * BAR_HEIGHT_RATIO;
  for (let index = ORIGIN; index < targets.length; index += UNIT) {
    barLabels[index].visible = visible;
    barLabels[index].scale.x = UNIT / scaleX;
    barLabels[index].scale.y = UNIT / scaleY;
    barLabels[index].x = barGraphics[index].x + LABEL_PAD_PX / scaleX;
    barLabels[index].y = barGraphics[index].y + (h - FONT_SIZE / scaleY) / HALF_DIVISOR;
  }
}

function updateAxis(): void {
  axisContainer.removeChildren();
  const { width: screenW, height: screenH } = app.screen;
  const background = new Graphics();
  background.rect(ORIGIN, screenH - AXIS_HEIGHT, screenW, AXIS_HEIGHT).fill({ color: COLOR_AXIS_BG });
  axisContainer.addChild(background);
  const border = new Graphics();
  border.setStrokeStyle({ width: THIN_STROKE_WIDTH, color: COLOR_AXIS_BORDER });
  border.moveTo(ORIGIN, screenH - AXIS_HEIGHT);
  border.lineTo(screenW, screenH - AXIS_HEIGHT);
  border.stroke();
  axisContainer.addChild(border);
  const pixelsPerMs = scale * chartContainer.scale.x;
  const step = niceTimeStep(pixelsPerMs);
  const timeAtLeft = ((ORIGIN - chartContainer.x) / chartContainer.scale.x - CHART_PADDING_LEFT) / scale;
  const timeAtRight = ((screenW - chartContainer.x) / chartContainer.scale.x - CHART_PADDING_LEFT) / scale;
  const firstTick = Math.max(ORIGIN, Math.ceil(timeAtLeft / step)) * step;
  const tickGraphics = new Graphics();
  tickGraphics.setStrokeStyle({ width: THIN_STROKE_WIDTH, color: COLOR_AXIS_TICK });
  for (let t = firstTick; t <= timeAtRight; t += step) {
    const worldX = CHART_PADDING_LEFT + timeToX(t, scale);
    const screenX = worldX * chartContainer.scale.x + chartContainer.x;
    tickGraphics.moveTo(screenX, screenH - AXIS_HEIGHT);
    tickGraphics.lineTo(screenX, screenH - AXIS_HEIGHT + TICK_HEIGHT);
    const label = new Text({
      text: formatMs(t),
      style: { fontSize: FONT_SIZE, fill: COLOR_AXIS_LABEL, fontFamily: "monospace" },
    });
    label.x = screenX + LABEL_PAD_PX;
    label.y = screenH - AXIS_HEIGHT + TICK_HEIGHT;
    axisContainer.addChild(label);
  }
  tickGraphics.stroke();
  axisContainer.addChild(tickGraphics);
}

function rebuildBars(): void {
  for (const [index, t] of targets.entries()) {
    const x = CHART_PADDING_LEFT + timeToX(t.start, scale);
    const y = laneToY(t.lane, laneHeight);
    const w = timeToX(t.cost, scale);
    const h = laneHeight * BAR_HEIGHT_RATIO;
    const { [index]: bar } = barGraphics;
    bar.clear();
    bar.rect(ORIGIN, ORIGIN, w, h).fill({ color: barColors[index] });
    bar.x = x;
    bar.y = y;
    const { [index]: label } = barLabels;
    label.x = x + LABEL_PAD_PX;
    label.y = y + (h - FONT_SIZE) / HALF_DIVISOR;
    const { [index]: mask } = barClipMasks;
    mask.clear();
    mask.rect(ORIGIN, ORIGIN, w, h).fill({ color: COLOR_WHITE });
    mask.x = x;
    mask.y = y;
  }
}

// ---------------------------------------------------------------------------
// Interactions: hover, click, search
// ---------------------------------------------------------------------------

function isPointerInViewport(event: FederatedPointerEvent): boolean {
  if (viewport === null) return true;
  const rectangle = viewport.getBoundingClientRect();
  const { clientX, clientY } = event;
  return pointInRect(clientX, clientY, rectangle);
}

function clearHoverState(): void {
  hoveredIndex = undefined;
  for (const [index, barGraphic] of barGraphics.entries()) {
    barGraphic.alpha = targets[index].on_critical_path ? HIGHLIGHT_ALPHA : DEFAULT_ALPHA;
  }
  applySearchHighlight();
  clearEdges(edgesContainer);
  hideTooltip();
}

function applySearchHighlight(): void {
  if (hoveredIndex !== undefined) return;
  if (searchMatches === undefined) {
    for (const [index, barGraphic] of barGraphics.entries()) {
      barGraphic.alpha = targets[index].on_critical_path ? HIGHLIGHT_ALPHA : DEFAULT_ALPHA;
    }
  } else {
    for (const [index, barGraphic] of barGraphics.entries()) {
      barGraphic.alpha = searchMatches.has(index) ? HIGHLIGHT_ALPHA : DIM_ALPHA;
    }
  }
}

function onBarPointerOver(event: FederatedPointerEvent): void {
  if (!isPointerInViewport(event)) { clearHoverState(); return; }
  if (!isBarGraphics(event.currentTarget)) return;
  const { targetIndex: index } = event.currentTarget;
  hoveredIndex = index;
  const cp = new Set(criticalPathThrough(index, targets));
  for (const [barIndex, barGraphic] of barGraphics.entries()) {
    barGraphic.alpha = cp.has(barIndex) ? HIGHLIGHT_ALPHA : DIM_ALPHA;
  }
  drawEdges(edgeContext(), index);
  const { global } = event;
  if (viewport !== null) {
    const vpRect = viewport.getBoundingClientRect();
    showTooltip(targets, index, global.x + vpRect.left, global.y + vpRect.top);
  }
}

function onBarPointerMove(event: FederatedPointerEvent): void {
  if (hoveredIndex === undefined) return;
  if (!isPointerInViewport(event)) { clearHoverState(); return; }
  const { global } = event;
  if (viewport !== null) {
    const vpRect = viewport.getBoundingClientRect();
    positionTooltip(global.x + vpRect.left, global.y + vpRect.top);
  }
}

function onBarPointerOut(): void { clearHoverState(); }

function onBarClick(event: FederatedPointerEvent): void {
  if (!isBarGraphics(event.currentTarget)) return;
  const { targetIndex: index } = event.currentTarget;
  const { name } = targets[index];
  globalThis.dispatchEvent(new CustomEvent("target-click", { detail: { name, index } }));
}

if (viewport !== null) {
  viewport.addEventListener("pointerleave", () => {
    if (hoveredIndex !== undefined) clearHoverState();
  });
}

// ---------------------------------------------------------------------------
// Pan and zoom
// ---------------------------------------------------------------------------

let isDragging = false;
let dragStartX = ORIGIN;
let dragStartY = ORIGIN;
let containerStartX = ORIGIN;
let containerStartY = ORIGIN;

if (viewport !== null) {
  viewport.addEventListener("pointerdown", (event: PointerEvent) => {
    isDragging = true;
    ({ clientX: dragStartX, clientY: dragStartY } = event);
    ({ x: containerStartX, y: containerStartY } = chartContainer);
  });
}
globalThis.addEventListener("pointermove", (event: PointerEvent) => {
  if (!isDragging) return;
  const { clientX, clientY } = event;
  chartContainer.x = containerStartX + clientX - dragStartX;
  chartContainer.y = containerStartY + clientY - dragStartY;
  updateLabels();
  updateAxis();
  if (hoveredIndex !== undefined) drawEdges(edgeContext(), hoveredIndex);
});
globalThis.addEventListener("pointerup", () => { isDragging = false; });

if (viewport !== null) {
  viewport.addEventListener("wheel", (event: WheelEvent) => {
    event.preventDefault();
    const direction = event.deltaY < ORIGIN ? UNIT : NEGATIVE_UNIT;
    const factor = direction > ZOOM_DIRECTION_POSITIVE ? ZOOM_FACTOR : UNIT / ZOOM_FACTOR;
    const rectangle = viewport.getBoundingClientRect();
    if (event.shiftKey) {
      const cursorY = event.clientY - rectangle.top;
      const worldY = (cursorY - chartContainer.y) / chartContainer.scale.y;
      chartContainer.scale.y = Math.max(MIN_SCALE, chartContainer.scale.y * factor);
      chartContainer.y = cursorY - worldY * chartContainer.scale.y;
    } else {
      const cursorX = event.clientX - rectangle.left;
      const worldX = (cursorX - chartContainer.x) / chartContainer.scale.x;
      chartContainer.scale.x = Math.max(MIN_SCALE, chartContainer.scale.x * factor);
      chartContainer.x = cursorX - worldX * chartContainer.scale.x;
    }
    updateLabels();
    updateAxis();
    if (hoveredIndex !== undefined) drawEdges(edgeContext(), hoveredIndex);
  }, { passive: false });
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

const searchInput = document.querySelector<HTMLInputElement>("#search-input");
if (searchInput !== null) {
  searchInput.addEventListener("input", () => {
    const query = searchInput.value.trim();
    searchMatches = query === "" ? undefined : new Set(searchFilter(query, targets));
    applySearchHighlight();
  });
}

// ---------------------------------------------------------------------------
// Resize and init
// ---------------------------------------------------------------------------

globalThis.addEventListener("resize", () => {
  laneHeight = Math.max(MIN_LANE_HEIGHT, Math.min(MAX_LANE_HEIGHT, (app.screen.height - AXIS_HEIGHT) / summary.lane_count));
  rebuildBars();
  updateLabels();
  updateAxis();
});

buildChart();

globalThis.updateSchedule = function updateSchedule(newData: ScheduleData): void {
  globalThis.DATA = newData;
  buildChart();
  populateSidebarStats();
};
