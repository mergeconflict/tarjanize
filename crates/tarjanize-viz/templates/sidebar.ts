// sidebar.ts -- Sidebar event wiring for the interactive split explorer.
//
// Manages state (selected target, original data), wires up DOM event
// handlers for target-click, shatter, save, and sidebar resize. Bundled
// as an IIFE (--format=iife) to avoid variable collisions with the ESM
// renderer bundle and avoid top-level await issues.
//
// Expects window.DATA and window.updateSchedule to be set by the time
// this script runs (the module loader in app.html guarantees this).

import {
  moduleCost,
  fmt,
  renderModuleTree,
  wireTreeToggles,
} from './tree';

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------

// Store the original schedule data so we can restore it when the user
// deselects a target or clicks away from a shatter operation.
let originalData = structuredClone((window as any).DATA);

// Currently selected target name (null when nothing selected).
let selectedTarget: string | null = null;

// -----------------------------------------------------------------------
// Helper: refresh sidebar stats from the current schedule data
// -----------------------------------------------------------------------

// Update the sidebar summary stats to reflect the latest schedule.
// Called after shatter operations change the schedule.
function refreshSidebarStats(): void {
  const fmtMs = (window as any).formatMs || ((v: number) => v.toFixed(1) + ' ms');
  const { summary } = (window as any).DATA;
  document.getElementById('stat-cp')!.textContent = fmtMs(summary.critical_path_ms);
  document.getElementById('stat-total')!.textContent = fmtMs(summary.total_cost_ms);
  document.getElementById('stat-parallel')!.textContent =
    summary.parallelism_ratio.toFixed(2) + 'x';
  document.getElementById('stat-targets')!.textContent = summary.target_count;
  document.getElementById('stat-symbols')!.textContent = summary.symbol_count;
  document.getElementById('stat-lanes')!.textContent = summary.lane_count;
}

// -----------------------------------------------------------------------
// Deselect: hide the sidebar panel
// -----------------------------------------------------------------------

function deselect(): void {
  selectedTarget = null;
  document.getElementById('recommendations')!.style.display = 'none';
}

// -----------------------------------------------------------------------
// Target-click event: fetch and display module/symbol tree
// -----------------------------------------------------------------------

window.addEventListener('target-click', async (event: Event) => {
  const { name } = (event as CustomEvent).detail;

  // If clicking the same target again, deselect it.
  if (selectedTarget === name) {
    deselect();
    return;
  }

  selectedTarget = name;

  // Target names are "package/target" (target may contain slashes,
  // e.g. "bin/main"). Split on the first slash only; the rest is the
  // target name passed as-is to the catch-all route parameter.
  const slashIdx = name.indexOf('/');
  const pkg = name.substring(0, slashIdx);
  const target = name.substring(slashIdx + 1);
  const treeUrl = `/api/tree/${encodeURIComponent(pkg)}/${target}`;

  try {
    const treeResp = await fetch(treeUrl);
    if (!treeResp.ok) return;

    const { target, symbol_costs } = await treeResp.json();

    // Show the panel.
    const recsDiv = document.getElementById('recommendations')!;
    recsDiv.style.display = '';

    // Compute total cost components.
    // Attr: Sum of all symbol costs in the tree.
    // Meta: Sum of metadata_decode_* events in the target timings.
    // Other: Sum of remaining events in the target timings.
    
    // Helper to calculate attr cost from the module tree
    function calculateAttrCost(mod: any): number {
      let cost = 0;
      for (const sym of Object.values(mod.symbols || {})) {
        if ((sym as any).event_times_ms) {
          cost += Object.values((sym as any).event_times_ms).reduce((a: any, b: any) => a + b, 0);
        }
      }
      for (const sub of Object.values(mod.submodules || {})) {
        cost += calculateAttrCost(sub);
      }
      return cost;
    }
    const totalAttr = calculateAttrCost(target.root);

    // Calculate Meta and Other from target timings
    let totalMeta = 0;
    let totalOther = 0;
    if (target.timings && target.timings.event_times_ms) {
      for (const [key, val] of Object.entries(target.timings.event_times_ms)) {
        const v = val as number;
        if (key.startsWith('metadata_decode_')) {
          totalMeta += v;
        } else {
          totalOther += v;
        }
      }
    }
    
    // Total cost is the sum of components.
    const totalCost = totalAttr + totalMeta + totalOther;

    // Render the header with target name and detailed cost breakdown.
    document.getElementById('rec-header')!.innerHTML =
      `<span class="rec-target-name">${name}</span>` +
      `<br><span class="rec-cost">Total: ${fmt(totalCost)}</span>` +
      `<br><span class="rec-cost-detail" title="Attribute Cost">Attr: ${fmt(totalAttr)}</span>` +
      `<br><span class="rec-cost-detail" title="Metadata Cost">Meta: ${fmt(totalMeta)}</span>` +
      `<br><span class="rec-cost-detail" title="Other Cost">Other: ${fmt(totalOther)}</span>`;

    // Render the module/symbol tree.
    const listDiv = document.getElementById('rec-list')!;
    listDiv.innerHTML = renderModuleTree('', target.root, 0, '', symbol_costs);
    wireTreeToggles(listDiv);
  } catch (_err) {
    // Network error -- silently ignore.
  }
});

// -----------------------------------------------------------------------
// "Shatter" button: replace target with horizon-grouped targets
// -----------------------------------------------------------------------

document.getElementById('shatter-btn')!.addEventListener('click', async () => {
  if (!selectedTarget) return;

  const shatterResp = await fetch('/api/shatter', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_id: selectedTarget }),
  });

  if (shatterResp.ok) {
    const newSchedule = await shatterResp.json();
    (window as any).DATA = newSchedule;
    (window as any).updateSchedule(newSchedule);
    refreshSidebarStats();

    // Shatter is persistent: update the baseline so subsequent
    // operations (including shattering other targets) build on it.
    originalData = structuredClone(newSchedule);
    selectedTarget = null;
    document.getElementById('recommendations')!.style.display = 'none';
  }
});

// -----------------------------------------------------------------------
// "Save Symbol Graph" button: download modified SymbolGraph as JSON
// -----------------------------------------------------------------------

document.getElementById('save-btn')!.addEventListener('click', async () => {
  const resp = await fetch('/api/export');
  const data = await resp.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'symbol_graph.json';
  a.click();
  URL.revokeObjectURL(url);
});

// -----------------------------------------------------------------------
// Sidebar resize: drag the handle to change sidebar width
// -----------------------------------------------------------------------

{
  const handle = document.getElementById('sidebar-resize')!;
  const panel = document.getElementById('right-panel')!;
  // Minimum and maximum sidebar widths in pixels.
  const MIN_WIDTH = 180;
  const MAX_WIDTH = 600;

  let dragging = false;

  handle.addEventListener('mousedown', (e: MouseEvent) => {
    e.preventDefault();
    dragging = true;
    document.body.style.cursor = 'col-resize';
    // Prevent text selection and pointer events on the canvas
    // while dragging, otherwise mousemove events get swallowed.
    document.body.style.userSelect = 'none';
  });

  window.addEventListener('mousemove', (e: MouseEvent) => {
    if (!dragging) return;
    const w = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX));
    panel.style.setProperty('--sidebar-w', w + 'px');
  });

  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    // Tell the PixiJS renderer to resize to the new viewport
    // dimensions. The renderer listens for window resize events,
    // so we dispatch one.
    window.dispatchEvent(new Event('resize'));
  });
}
