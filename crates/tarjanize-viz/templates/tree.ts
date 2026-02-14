// tree.ts -- Pure tree-rendering functions for the module/symbol sidebar.
//
// Renders a collapsible module/symbol tree from Target data fetched from
// the `/api/tree` API. All functions are pure (no DOM side-effects) except
// `wireTreeToggles` which attaches click handlers for collapse/expand.

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

// Indentation depth multiplier in pixels for nested tree nodes.
const INDENT_PX = 14;

// Sentinel for "no cost" or "no items" comparisons.
const ZERO = 0;

// Single dependency threshold for singular/plural label.
const SINGLE_DEP = 1;

// Number of fractional digits for millisecond display fallback.
const MS_FRACTION_DIGITS = 1;

// -----------------------------------------------------------------------
// Cost and count helpers
// -----------------------------------------------------------------------

/** Compute total cost for a symbol using the breakdown map if available.
 *
 * Why: the UI needs a single scalar to sort and display symbols.
 */
export function symbolTotalCost(
  name: string,
  sym: any,
  path: string,
  costs: any
): number {
  if (costs !== undefined && costs !== null) {
    const fullPath = path === '' ? name : `${path}::${name}`;
    const { [fullPath]: costEntry } = costs;
    if (costEntry !== undefined && costEntry !== null) return costEntry.attr + costEntry.meta_share + costEntry.other_share;
  }
  // Fallback: sum raw events (attr only)
  if (sym.event_times_ms === undefined || sym.event_times_ms === null) return ZERO;
  const eventValues: number[] = Object.values(sym.event_times_ms);
  return eventValues.reduce(
    (a: number, b: number) => a + b,
    ZERO
  );
}

/** Compute aggregate cost for a module using the breakdown map.
 *
 * Why: module headers display a rolled-up cost for their subtree.
 */
export function moduleCost(module_: any, path: string, costs: any): number {
  let cost = 0;
  for (const [name, sym] of Object.entries(module_.symbols ?? {})) {
    cost += symbolTotalCost(name, sym, path, costs);
  }
  for (const [name, sub] of Object.entries(module_.submodules ?? {})) {
    const childPath = path === '' ? name : `${path}::${name}`;
    cost += moduleCost(sub, childPath, costs);
  }
  return cost;
}

/** Count total symbols in a module tree (recursive).
 *
 * Why: module headers show a quick symbol count for scale.
 */
export function moduleSymbolCount(module_: any): number {
  let { length: count } = Object.keys(module_.symbols ?? {});
  for (const sub of Object.values(module_.submodules ?? {})) {
    count += moduleSymbolCount(sub);
  }
  return count;
}

// -----------------------------------------------------------------------
// Symbol metadata helpers
// -----------------------------------------------------------------------

/** Return a short label for a symbol's kind.
 *
 * Why: the sidebar uses compact labels to save space.
 */
export function symbolKindLabel(sym: any): string {
  if (sym.module_def !== undefined && sym.module_def !== null) return sym.module_def.kind.toLowerCase();
  if (sym.impl !== undefined && sym.impl !== null) return 'impl';
  return '?';
}

/**
 * Return the display name for a symbol. For ModuleDef, use the HashMap
 * key (the symbol name). For Impl, use the human-readable name (e.g.
 * "impl Trait for Type") instead of the internal compiler DefPath key.
 *
 * Why: impl symbols should be readable and not expose compiler internals.
 */
export function symbolDisplayName(symName: string, sym: any): string {
  if (sym.impl !== undefined && sym.impl !== null) return sym.impl.name;
  return symName;
}

/** Return a visibility badge for public symbols.
 *
 * Why: the list view needs a lightweight public/private indicator.
 */
export function visBadge(sym: any): string {
  if (sym.module_def?.visibility === 'public') {
    return '<span class="tree-vis">pub</span>';
  }
  return '';
}

// -----------------------------------------------------------------------
// Format helper
// -----------------------------------------------------------------------

/**
 * Escape text for HTML injection-safe rendering.
 *
 * Why: symbol and impl names can contain angle brackets or other characters
 * that must be escaped to keep the sidebar list valid and readable.
 */
export function escapeHtml(text: string): string {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll('\'', '&#39;');
}

/** Format milliseconds for display, using the global formatMs if set.
 *
 * Why: the viz UI uses a shared formatter for consistent units.
 */
export function fmt(ms: number): string {
  // globalThis.formatMs may not yet be initialized at runtime despite
  // the type declaration. Use an intermediate typed as possibly undefined
  // to allow the nullish coalescing check.
  const formatter: ((v: number) => string) | undefined = globalThis.formatMs as ((v: number) => string) | undefined;
  return (formatter ?? ((v: number) => `${v.toFixed(MS_FRACTION_DIGITS)} ms`))(
    ms
  );
}

// -----------------------------------------------------------------------
// Tree rendering internals
// -----------------------------------------------------------------------

/** Parameters for renderModuleTree, bundled to satisfy max-params. */
export interface TreeRenderParameters {
  name: string;
  module_: any;
  depth: number;
  path: string;
  costs: any;
}

/** Parameters for renderSymbolRows, bundled to satisfy max-params. */
interface SymbolRowParameters {
  symKeys: string[];
  syms: Record<string, any>;
  prefix: string;
  depth: number;
  costs: any;
}

/** Parameters for rendering a single symbol row. */
interface SingleSymbolRowParameters {
  symName: string;
  sym: any;
  prefix: string;
  depth: number;
  costs: any;
}

/**
 * Render a single symbol row. Returns an HTML string for one symbol entry.
 *
 * Why: extracted from renderSymbolRows to reduce cyclomatic complexity.
 */
function renderSingleSymbolRow(parameters: SingleSymbolRowParameters): string {
  const { symName, sym, prefix, depth, costs } = parameters;
  const total = symbolTotalCost(symName, sym, prefix, costs);
  const depCount = sym.dependencies !== undefined && sym.dependencies !== null ? sym.dependencies.length : ZERO;
  const kind = escapeHtml(symbolKindLabel(sym));
  const vis = visBadge(sym);

  // Format breakdown tooltips or extra text
  let breakdown = '';
  if (costs !== undefined && costs !== null) {
    const fullPath = prefix === '' ? symName : `${prefix}::${symName}`;
    const { [fullPath]: costEntry } = costs;
    if (costEntry !== undefined && costEntry !== null) {
      breakdown = ` title="Attr: ${fmt(costEntry.attr)}, Meta: ${fmt(costEntry.meta_share)}, Other: ${fmt(costEntry.other_share)}"`;
    }
  }

  const displaySymName = escapeHtml(symbolDisplayName(symName, sym));
  const depsSuffix = depCount > ZERO
    ? `<span class="tree-deps">${depCount} dep${depCount === SINGLE_DEP ? '' : 's'}</span>`
    : '';

  return (
    `<div class="tree-symbol" style="padding-left:${(depth + SINGLE_DEP) * INDENT_PX}px"${breakdown}>` +
    `<span class="tree-sym-kind">${kind}</span>${
    vis
    }<span class="tree-sym-name">${displaySymName}</span>` +
    `<span class="tree-cost">${fmt(total)}</span>${
    depsSuffix
    }</div>`
  );
}

/**
 * Render symbol rows for a module node. Returns an HTML string fragment.
 *
 * Why: extracted from renderModuleTree to reduce cyclomatic complexity.
 */
function renderSymbolRows(parameters: SymbolRowParameters): string {
  const { symKeys, syms, prefix, depth, costs } = parameters;
  let html = '';
  for (const symName of symKeys) {
    const { [symName]: sym } = syms;
    html += renderSingleSymbolRow({ symName, sym, prefix, depth, costs });
  }
  return html;
}

/**
 * Build a module path prefix from the parent path and current name.
 *
 * Why: reused in multiple places to construct qualified symbol paths.
 */
function buildPrefix(path: string, name: string): string {
  if (path === '') return name;
  return name === '' ? path : `${path}::${name}`;
}

/**
 * Render a module node as a collapsible tree. Returns an HTML string.
 * `parameters.name` is the module name (empty string for root).
 * `parameters.depth` controls indentation level.
 *
 * Why: HTML string rendering keeps sidebar updates fast and isolated.
 */
export function renderModuleTree(parameters: TreeRenderParameters): string {
  const { name, module_, depth, path, costs } = parameters;

  const prefix = buildPrefix(path, name);

  const cost = moduleCost(module_, prefix, costs);
  const symCount = moduleSymbolCount(module_);

  // Sort submodules and symbols by cost descending.
  const subs = module_.submodules ?? {};
  const syms = module_.symbols ?? {};
  const subKeys = Object.keys(subs).toSorted(
    (a, b) => moduleCost(subs[b], prefix === '' ? b : `${prefix}::${b}`, costs) -
              moduleCost(subs[a], prefix === '' ? a : `${prefix}::${a}`, costs)
  );
  const symKeys = Object.keys(syms).toSorted(
    (a, b) => symbolTotalCost(b, syms[b], prefix, costs) -
              symbolTotalCost(a, syms[a], prefix, costs)
  );
  const hasChildren = subKeys.length > ZERO || symKeys.length > ZERO;

  // Module header row. Shows collapse toggle, name, aggregate cost,
  // and symbol count.
  const displayName = escapeHtml(name === '' ? '(root)' : name);
  const toggle = hasChildren
    ? '<span class="tree-toggle">\u25B6</span>'
    : '<span class="tree-toggle-spacer"></span>';
  const header =
    `<div class="tree-module-header" style="padding-left:${depth * INDENT_PX}px">` +
    `${toggle}<span class="tree-mod-name">${displayName}</span>` +
    `<span class="tree-cost">${fmt(cost)}</span>` +
    `<span class="tree-count">${symCount} sym</span>` +
    `</div>`;

  // Children: submodules first, then symbols. Hidden by default
  // (collapsed). Each child is rendered at depth+1.
  let childrenHtml = '';
  for (const subName of subKeys) {
    childrenHtml += renderModuleTree({
      name: subName,
      module_: module_.submodules[subName],
      depth: depth + SINGLE_DEP,
      path: prefix,
      costs,
    });
  }
  childrenHtml += renderSymbolRows({ symKeys, syms, prefix, depth, costs });

  const children = hasChildren
    ? `<div class="tree-children" style="display:none">${childrenHtml}</div>`
    : '';

  return `<div class="tree-module">${header}${children}</div>`;
}

/**
 * Wire up collapse/expand toggle on all .tree-toggle elements inside
 * the given container. Clicking the toggle or the module header row
 * expands/collapses the immediate children.
 *
 * Why: tree nodes must be interactive without a framework.
 */
export function wireTreeToggles(container: HTMLElement): void {
  for (const header of container.querySelectorAll('.tree-module-header')) {
    const toggle = header.querySelector('.tree-toggle');
    if (toggle === null) continue;
    const { nextElementSibling } = header;
    if (nextElementSibling === null) continue;

    header.addEventListener('click', () => {
      // nextElementSibling is an Element; narrow to HTMLElement for
      // style access via instanceof. This is safe because the tree
      // HTML only contains div elements as siblings.
      if (!(nextElementSibling instanceof HTMLElement)) return;
      const collapsed = nextElementSibling.style.display === 'none';
      nextElementSibling.style.display = collapsed ? '' : 'none';
      toggle.textContent = collapsed ? '\u25BC' : '\u25B6';
    });
  }
}
