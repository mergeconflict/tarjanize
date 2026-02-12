// tree.ts -- Pure tree-rendering functions for the module/symbol sidebar.
//
// Renders a collapsible module/symbol tree from Target data fetched from
// the `/api/tree` API. All functions are pure (no DOM side-effects) except
// `wireTreeToggles` which attaches click handlers for collapse/expand.

// -----------------------------------------------------------------------
// Cost and count helpers
// -----------------------------------------------------------------------

/** Compute total cost for a symbol by summing its event_times_ms values. */
export function symbolCost(sym: any): number {
  if (!sym.event_times_ms) return 0;
  return Object.values(sym.event_times_ms).reduce(
    (a: number, b: number) => a + b,
    0
  );
}

/** Compute aggregate cost for a module (recursive sum of all symbols). */
export function moduleCost(mod_: any): number {
  let cost = 0;
  for (const sym of Object.values(mod_.symbols || {})) {
    cost += symbolCost(sym);
  }
  for (const sub of Object.values(mod_.submodules || {})) {
    cost += moduleCost(sub);
  }
  return cost;
}

/** Count total symbols in a module tree (recursive). */
export function moduleSymbolCount(mod_: any): number {
  let count = Object.keys(mod_.symbols || {}).length;
  for (const sub of Object.values(mod_.submodules || {})) {
    count += moduleSymbolCount(sub);
  }
  return count;
}

// -----------------------------------------------------------------------
// Symbol metadata helpers
// -----------------------------------------------------------------------

/** Return a short label for a symbol's kind. */
export function symbolKindLabel(sym: any): string {
  if (sym.module_def) return sym.module_def.kind.toLowerCase();
  if (sym['impl']) return 'impl';
  return '?';
}

/**
 * Return the display name for a symbol. For ModuleDef, use the HashMap
 * key (the symbol name). For Impl, use the human-readable name (e.g.
 * "impl Trait for Type") instead of the internal compiler DefPath key.
 */
export function symbolDisplayName(symName: string, sym: any): string {
  if (sym['impl']) return sym['impl'].name;
  return symName;
}

/** Return a visibility badge for public symbols. */
export function visBadge(sym: any): string {
  if (sym.module_def && sym.module_def.visibility === 'public') {
    return '<span class="tree-vis">pub</span>';
  }
  return '';
}

// -----------------------------------------------------------------------
// Format helper
// -----------------------------------------------------------------------

/** Format milliseconds for display, using the global formatMs if set. */
export function fmt(ms: number): string {
  return ((window as any).formatMs || ((v: number) => v.toFixed(1) + ' ms'))(
    ms
  );
}

// -----------------------------------------------------------------------
// Tree rendering
// -----------------------------------------------------------------------

/**
 * Render a module node as a collapsible tree. Returns an HTML string.
 * `name` is the module name (empty string for root). `depth` controls
 * indentation level.
 */
export function renderModuleTree(
  name: string,
  mod_: any,
  depth: number
): string {
  const cost = moduleCost(mod_);
  const symCount = moduleSymbolCount(mod_);
  // Sort submodules and symbols by cost descending. Submodules
  // always appear before symbols regardless of cost.
  const subs = mod_.submodules || {};
  const syms = mod_.symbols || {};
  const subKeys = Object.keys(subs).sort(
    (a, b) => moduleCost(subs[b]) - moduleCost(subs[a])
  );
  const symKeys = Object.keys(syms).sort(
    (a, b) => symbolCost(syms[b]) - symbolCost(syms[a])
  );
  const hasChildren = subKeys.length > 0 || symKeys.length > 0;

  // Module header row. Shows collapse toggle, name, aggregate cost,
  // and symbol count.
  const displayName = name || '(root)';
  const toggle = hasChildren
    ? '<span class="tree-toggle">\u25B6</span>'
    : '<span class="tree-toggle-spacer"></span>';
  const header =
    `<div class="tree-module-header" style="padding-left:${depth * 14}px">` +
    `${toggle}<span class="tree-mod-name">${displayName}</span>` +
    `<span class="tree-cost">${fmt(cost)}</span>` +
    `<span class="tree-count">${symCount} sym</span>` +
    `</div>`;

  // Children: submodules first, then symbols. Hidden by default
  // (collapsed). Each child is rendered at depth+1.
  let childrenHtml = '';
  for (const subName of subKeys) {
    childrenHtml += renderModuleTree(
      subName,
      mod_.submodules[subName],
      depth + 1
    );
  }
  for (const symName of symKeys) {
    const sym = mod_.symbols[symName];
    const cost_ = symbolCost(sym);
    const depCount = sym.dependencies ? sym.dependencies.length : 0;
    const kind = symbolKindLabel(sym);
    const vis = visBadge(sym);
    childrenHtml +=
      `<div class="tree-symbol" style="padding-left:${(depth + 1) * 14}px">` +
      `<span class="tree-sym-kind">${kind}</span>` +
      `${vis}` +
      `<span class="tree-sym-name">${symbolDisplayName(symName, sym)}</span>` +
      `<span class="tree-cost">${fmt(cost_)}</span>` +
      (depCount > 0
        ? `<span class="tree-deps">${depCount} dep${depCount !== 1 ? 's' : ''}</span>`
        : '') +
      `</div>`;
  }

  const children = hasChildren
    ? `<div class="tree-children" style="display:none">${childrenHtml}</div>`
    : '';

  return `<div class="tree-module">${header}${children}</div>`;
}

/**
 * Wire up collapse/expand toggle on all .tree-toggle elements inside
 * the given container. Clicking the toggle or the module header row
 * expands/collapses the immediate children.
 */
export function wireTreeToggles(container: HTMLElement): void {
  container.querySelectorAll('.tree-module-header').forEach((header) => {
    const toggle = header.querySelector('.tree-toggle');
    if (!toggle) return;
    const children = header.nextElementSibling as HTMLElement | null;
    if (!children) return;

    header.addEventListener('click', () => {
      const collapsed = children.style.display === 'none';
      children.style.display = collapsed ? '' : 'none';
      toggle.textContent = collapsed ? '\u25BC' : '\u25B6';
    });
  });
}
