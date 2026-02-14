// sidebar.ts -- Sidebar event wiring for the interactive split explorer.
//
// Manages state (selected target, original data), wires up DOM event
// handlers for target-click, shatter, save, and sidebar resize. Bundled
// as an IIFE (--format=iife) to avoid variable collisions with the ESM
// renderer bundle and avoid top-level await issues.
//
// Expects window.DATA and window.updateSchedule to be set by the time
// this script runs (the module loader in app.html guarantees this).

import type { ScheduleData } from '../../../types/generated/ScheduleData';
import type { Summary } from '../../../types/generated/Summary';
import { fmt, renderModuleTree, wireTreeToggles } from './tree';

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

// Number of fractional digits for the parallelism ratio display.
const PARALLELISM_FRACTION_DIGITS = 2;

// Number of fractional digits for millisecond display fallback.
const MS_FRACTION_DIGITS = 1;

// JSON indentation level for exported symbol graph.
const JSON_INDENT = 2;

// First slash offset for splitting package/target names.
const FIRST_ELEMENT_OFFSET = 1;

// Zero sentinel for Math.max floor and initial accumulator.
const ZERO_FLOOR = 0;

// Initial tree depth for the root module.
const ROOT_DEPTH = 0;

// Minimum and maximum sidebar widths in pixels.
const MIN_WIDTH = 180;
const MAX_WIDTH = 600;

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------

// Currently selected target name (undefined when nothing selected).
// eslint-disable-next-line unicorn/no-useless-undefined -- init-declarations requires explicit initialization
let selectedTarget: string | undefined = undefined;

// -----------------------------------------------------------------------
// Helper: refresh sidebar stats from the current schedule data
// -----------------------------------------------------------------------

// Update the sidebar summary stats to reflect the latest schedule.
// Called after shatter operations change the schedule.
//
// Why: the sidebar must stay in sync with the current schedule data.
function refreshSidebarStats(): void {
  // globalThis.formatMs may not yet be initialized at runtime despite
  // the type declaration. Cast to possibly-undefined for safe fallback.
  const formatter: ((v: number) => string) | undefined = globalThis.formatMs as ((v: number) => string) | undefined;
  const fmtMs = formatter ?? ((v: number) => `${v.toFixed(MS_FRACTION_DIGITS)} ms`);
  const { summary }: { summary: Summary } = globalThis.DATA;
  const statCp = document.querySelector<HTMLElement>('#stat-cp');
  const statTotal = document.querySelector<HTMLElement>('#stat-total');
  const statParallel = document.querySelector<HTMLElement>('#stat-parallel');
  const statTargets = document.querySelector<HTMLElement>('#stat-targets');
  const statSymbols = document.querySelector<HTMLElement>('#stat-symbols');
  const statLanes = document.querySelector<HTMLElement>('#stat-lanes');
  if (statCp !== null) { statCp.textContent = fmtMs(summary.critical_path_ms); }
  if (statTotal !== null) { statTotal.textContent = fmtMs(summary.total_cost_ms); }
  if (statParallel !== null) {
    statParallel.textContent = `${summary.parallelism_ratio.toFixed(PARALLELISM_FRACTION_DIGITS)}x`;
  }
  if (statTargets !== null) { statTargets.textContent = String(summary.target_count); }
  if (statSymbols !== null) { statSymbols.textContent = String(summary.symbol_count); }
  if (statLanes !== null) { statLanes.textContent = String(summary.lane_count); }
}

// -----------------------------------------------------------------------
// Deselect: hide the sidebar panel
// -----------------------------------------------------------------------

// Why: hides stale data when the user clears the current selection.
function deselect(): void {
  selectedTarget = undefined;
  const recommendations = document.querySelector<HTMLElement>('#recommendations');
  if (recommendations !== null) { recommendations.style.display = 'none'; }
}

// -----------------------------------------------------------------------
// Target-click event: fetch and display module/symbol tree
//
// Why: the sidebar is driven by the currently selected target.
// -----------------------------------------------------------------------

// Helper to calculate attr cost from the module tree.
//
// Why: attr cost must be recomputed from the current tree view.
function calculateAttributeCost(module_: any): number {
  let cost = 0;
  const symbols: Record<string, any> = module_.symbols ?? {};
  for (const sym of Object.values(symbols)) {
    const { event_times_ms: eventTimes } = sym;
    if (eventTimes !== undefined && eventTimes !== null) {
      const eventValues: number[] = Object.values(eventTimes);
      cost += eventValues.reduce((a: number, b: number) => a + b, ZERO_FLOOR);
    }
  }
  const submodules: Record<string, any> = module_.submodules ?? {};
  for (const sub of Object.values(submodules)) {
    cost += calculateAttributeCost(sub);
  }
  return cost;
}

// Build the cost components for a tree target (Attr, Meta, Other).
//
// Why: cost breakdown is needed for the sidebar header display.
function buildCostComponents(treeTarget: any): { totalAttribute: number; totalMeta: number; totalOther: number } {
  const totalAttribute = calculateAttributeCost(treeTarget.root);

  // Calculate Meta and Other from target timings
  let totalMeta = 0;
  let totalOther = 0;
  if (treeTarget.timings?.event_times_ms !== undefined && treeTarget.timings?.event_times_ms !== null) {
    for (const [key, value] of Object.entries(treeTarget.timings.event_times_ms)) {
      const numericValue = Number(value);
      if (key.startsWith('metadata_decode_')) {
        totalMeta += numericValue;
      } else {
        totalOther += numericValue;
      }
    }
  }

  return { totalAttribute, totalMeta, totalOther };
}

// Wrap the async handler so addEventListener receives a void-returning function.
// Why: addEventListener expects `() => void`, but the handler is async.
function handleTargetClick(event: Event): void {
  void (async () => {
    // This handler only fires for CustomEvent dispatches from
    // the 'target-click' event. Narrow via instanceof check.
    if (!(event instanceof CustomEvent)) return;
    const name: string = event.detail.name;

    // If clicking the same target again, deselect it.
    if (selectedTarget === name) {
      deselect();
      return;
    }

    selectedTarget = name;

    // Target names are "package/target" (target may contain slashes,
    // e.g. "bin/main"). Split on the first slash only; the rest is the
    // target name passed as-is to the catch-all route parameter.
    const slashIndex = name.indexOf('/');
    const package_ = name.slice(ZERO_FLOOR, Math.max(ZERO_FLOOR, slashIndex));
    const target = name.slice(Math.max(ZERO_FLOOR, slashIndex + FIRST_ELEMENT_OFFSET));
    const treeUrl = `/api/tree/${encodeURIComponent(package_)}/${target}`;

    try {
      const treeResponse = await fetch(treeUrl);
      if (!treeResponse.ok) return;

      const { target: treeTarget, symbol_costs: symbolCosts } = await treeResponse.json();

      // Show the panel.
      const recsDiv = document.querySelector<HTMLElement>('#recommendations');
      if (recsDiv === null) return;
      recsDiv.style.display = '';

      // Compute total cost components.
      // Attr: Sum of all symbol costs in the tree.
      // Meta: Sum of metadata_decode_* events in the target timings.
      // Other: Sum of remaining events in the target timings.
      const { totalAttribute, totalMeta, totalOther } = buildCostComponents(treeTarget);

      // Total cost is the sum of components.
      const totalCost = totalAttribute + totalMeta + totalOther;

      // Render the header with target name and detailed cost breakdown.
      const recHeader = document.querySelector<HTMLElement>('#rec-header');
      if (recHeader !== null) {
        recHeader.innerHTML =
          `<span class="rec-target-name">${name}</span>` +
          `<br><span class="rec-cost">Total: ${fmt(totalCost)}</span>` +
          `<br><span class="rec-cost-detail" title="Attribute Cost">Attr: ${fmt(totalAttribute)}</span>` +
          `<br><span class="rec-cost-detail" title="Metadata Cost">Meta: ${fmt(totalMeta)}</span>` +
          `<br><span class="rec-cost-detail" title="Other Cost">Other: ${fmt(totalOther)}</span>`;
      }

      // Render the module/symbol tree.
      const listDiv = document.querySelector<HTMLElement>('#rec-list');
      if (listDiv !== null) {
        listDiv.innerHTML = renderModuleTree({ name: '', module_: treeTarget.root, depth: ROOT_DEPTH, path: '', costs: symbolCosts });
        wireTreeToggles(listDiv);
      }
    } catch {
      // Network error -- silently ignore.
    }
  })();
}

globalThis.addEventListener('target-click', handleTargetClick);

// -----------------------------------------------------------------------
// "Shatter" button: replace target with horizon-grouped targets
// -----------------------------------------------------------------------
//
// Why: this drives the what-if split exploration flow.

// Wrap the async handler so addEventListener receives a void-returning function.
function handleShatterClick(): void {
  // Capture selectedTarget synchronously to avoid race condition
  // between the check and the async usage.
  const currentTarget = selectedTarget;
  void (async () => {
    if (currentTarget === undefined || currentTarget === '') return;

    const shatterResponse = await fetch('/api/shatter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: currentTarget }),
    });

    if (shatterResponse.ok) {
      const newSchedule: ScheduleData = await shatterResponse.json();
      globalThis.DATA = newSchedule;
      globalThis.updateSchedule(newSchedule);
      refreshSidebarStats();

      selectedTarget = undefined;
      const recommendations = document.querySelector<HTMLElement>('#recommendations');
      if (recommendations !== null) { recommendations.style.display = 'none'; }
    }
  })();
}

const shatterButton = document.querySelector<HTMLElement>('#shatter-btn');
if (shatterButton !== null) {
  shatterButton.addEventListener('click', handleShatterClick);
}

// -----------------------------------------------------------------------
// "Save Symbol Graph" button: download modified SymbolGraph as JSON
// -----------------------------------------------------------------------
//
// Why: exports the current state for use in downstream CLI stages.

// Wrap the async handler so addEventListener receives a void-returning function.
function handleSaveClick(): void {
  void (async () => {
    const response = await fetch('/api/export');
    const data = await response.json();
    const blob = new Blob([JSON.stringify(data, undefined, JSON_INDENT)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'symbol_graph.json';
    anchor.click();
    URL.revokeObjectURL(url);
  })();
}

const saveButton = document.querySelector<HTMLElement>('#save-btn');
if (saveButton !== null) {
  saveButton.addEventListener('click', handleSaveClick);
}

// -----------------------------------------------------------------------
// Sidebar resize: drag the handle to change sidebar width
// -----------------------------------------------------------------------
//
// Why: gives users control over the tree panel size.

{
  const handle = document.querySelector<HTMLElement>('#sidebar-resize');
  const panel = document.querySelector<HTMLElement>('#right-panel');

  let dragging = false;

  if (handle !== null) {
    handle.addEventListener('mousedown', (event: MouseEvent) => {
      event.preventDefault();
      dragging = true;
      document.body.style.cursor = 'col-resize';
      // Prevent text selection and pointer events on the canvas
      // while dragging, otherwise mousemove events get swallowed.
      document.body.style.userSelect = 'none';
    });
  }

  globalThis.addEventListener('mousemove', (event: MouseEvent) => {
    if (!dragging) return;
    const w = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, event.clientX));
    if (panel !== null) { panel.style.setProperty('--sidebar-w', `${w}px`); }
  });

  globalThis.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    // Tell the PixiJS renderer to resize to the new viewport
    // dimensions. The renderer listens for window resize events,
    // so we dispatch one.
    globalThis.dispatchEvent(new Event('resize'));
  });
}
