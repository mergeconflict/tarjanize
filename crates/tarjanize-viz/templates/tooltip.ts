/* eslint-disable @typescript-eslint/prefer-destructuring -- false positives on already-destructured code */

// tooltip.ts — Tooltip DOM management for the Gantt chart.
//
// Positions and populates the floating tooltip element that appears
// when hovering over target bars in the schedule visualization.

import type { TargetData } from "../../../types/generated/TargetData";
import { formatMs } from "./logic.ts";
import { TOOLTIP_PAD } from "./constants.ts";

// Resolve the tooltip element once at module load time.
const tooltipElement = document.querySelector<HTMLElement>("#tooltip");

export function showTooltip(targets: TargetData[], index: number, globalX: number, globalY: number): void {
  const { name, cost, start, finish, slack, symbol_count: symbolCount, deps, dependents } = targets[index];
  if (tooltipElement !== null) {
    tooltipElement.innerHTML = `
      <div class="tt-name">${name}</div>
      <div class="tt-row"><span>Cost:</span><span>${formatMs(cost)}</span></div>
      <div class="tt-row"><span>Start:</span><span>${formatMs(start)}</span></div>
      <div class="tt-row"><span>Finish:</span><span>${formatMs(finish)}</span></div>
      <div class="tt-row"><span>Slack:</span><span>${formatMs(slack)}</span></div>
      <div class="tt-row"><span>Symbols:</span><span>${String(symbolCount)}</span></div>
      <div class="tt-row"><span>Deps:</span><span>${String(deps.length)}</span></div>
      <div class="tt-row"><span>Dependents:</span><span>${String(dependents.length)}</span></div>`;
    tooltipElement.style.display = "block";
  }
  positionTooltip(globalX, globalY);
}

export function positionTooltip(x: number, y: number): void {
  if (tooltipElement === null) return;
  const { width: rectWidth, height: rectHeight } = tooltipElement.getBoundingClientRect();
  let left = x + TOOLTIP_PAD;
  let top = y + TOOLTIP_PAD;
  if (left + rectWidth > globalThis.innerWidth) { left = x - rectWidth - TOOLTIP_PAD; }
  if (top + rectHeight > globalThis.innerHeight) { top = y - rectHeight - TOOLTIP_PAD; }
  tooltipElement.style.left = `${String(left)}px`;
  tooltipElement.style.top = `${String(top)}px`;
}

export function hideTooltip(): void {
  if (tooltipElement !== null) { tooltipElement.style.display = "none"; }
}
