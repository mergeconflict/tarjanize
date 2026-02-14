// edges.ts — Dependency edge drawing for the Gantt chart.
//
// Draws critical-path (solid bezier) and off-path (dashed line)
// edges between bars when hovering over a target.

/* eslint-disable @typescript-eslint/prefer-destructuring -- false positives on already-destructured code */

import type { Container } from "pixi.js";
import { Graphics } from "pixi.js";
import type { TargetData } from "../../../types/generated/TargetData";
import { criticalPathThrough, timeToX, laneToY } from "./logic.ts";
import {
  CHART_PADDING_LEFT, BAR_HEIGHT_RATIO, HALF_DIVISOR,
  COLOR_EDGE_CRITICAL, COLOR_EDGE_OFFPATH,
  CRITICAL_EDGE_ALPHA, EDGE_OFFPATH_ALPHA,
  CRITICAL_EDGE_WIDTH, THIN_STROKE_WIDTH,
  MIN_EDGE_LENGTH, DASH_LENGTH, GAP_LENGTH, ORIGIN, UNIT,
} from "./constants.ts";

// Parameters shared across edge-drawing calls. Passed from the renderer
// so edge drawing doesn't own any mutable state.
export interface EdgeContext {
  edgesContainer: Container;
  chartContainer: Container;
  targets: TargetData[];
  scale: number;
  laneHeight: number;
}

function worldToScreen(chartContainer: Container, wx: number, wy: number): { x: number; y: number } {
  return {
    x: wx * chartContainer.scale.x + chartContainer.x,
    y: wy * chartContainer.scale.y + chartContainer.y,
  };
}

function drawSingleEdge(context: EdgeContext, fromIndex: number, toIndex: number, isCritical: boolean): void {
  const { targets, scale, laneHeight, chartContainer, edgesContainer } = context;
  const { finish: fromFinish, lane: fromLane } = targets[fromIndex];
  const { start: toStart, lane: toLane } = targets[toIndex];
  const w1 = worldToScreen(
    chartContainer,
    CHART_PADDING_LEFT + timeToX(fromFinish, scale),
    laneToY(fromLane, laneHeight) + (laneHeight * BAR_HEIGHT_RATIO) / HALF_DIVISOR,
  );
  const w2 = worldToScreen(
    chartContainer,
    CHART_PADDING_LEFT + timeToX(toStart, scale),
    laneToY(toLane, laneHeight) + (laneHeight * BAR_HEIGHT_RATIO) / HALF_DIVISOR,
  );
  const { x: x1, y: y1 } = w1;
  const { x: x2, y: y2 } = w2;
  const graphics = new Graphics();
  const color = isCritical ? COLOR_EDGE_CRITICAL : COLOR_EDGE_OFFPATH;
  const alpha = isCritical ? CRITICAL_EDGE_ALPHA : EDGE_OFFPATH_ALPHA;
  if (isCritical) {
    const cpx = (x1 + x2) / HALF_DIVISOR;
    graphics.setStrokeStyle({ width: CRITICAL_EDGE_WIDTH, color, alpha });
    graphics.moveTo(x1, y1);
    graphics.bezierCurveTo(cpx, y1, cpx, y2, x2, y2);
    graphics.stroke();
  } else {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const distribution = Math.hypot(dx, dy);
    if (distribution < MIN_EDGE_LENGTH) { edgesContainer.addChild(graphics); return; }
    const ux = dx / distribution;
    const uy = dy / distribution;
    graphics.setStrokeStyle({ width: THIN_STROKE_WIDTH, color, alpha });
    let d = ORIGIN;
    while (d < distribution) {
      const segEnd = Math.min(d + DASH_LENGTH, distribution);
      graphics.moveTo(x1 + ux * d, y1 + uy * d);
      graphics.lineTo(x1 + ux * segEnd, y1 + uy * segEnd);
      d = segEnd + GAP_LENGTH;
    }
    graphics.stroke();
  }
  edgesContainer.addChild(graphics);
}

export function drawEdges(context: EdgeContext, index: number): void {
  context.edgesContainer.removeChildren();
  const cpSet = new Set(criticalPathThrough(index, context.targets));
  const { deps, dependents } = context.targets[index];
  const cpArray = criticalPathThrough(index, context.targets);
  for (let cpIndex = ORIGIN; cpIndex < cpArray.length - UNIT; cpIndex += UNIT) {
    drawSingleEdge(context, cpArray[cpIndex], cpArray[cpIndex + UNIT], true);
  }
  for (const dep of deps) { if (!cpSet.has(dep)) drawSingleEdge(context, dep, index, false); }
  for (const dependent of dependents) { if (!cpSet.has(dependent)) drawSingleEdge(context, index, dependent, false); }
}

export function clearEdges(edgesContainer: Container): void {
  edgesContainer.removeChildren();
}
