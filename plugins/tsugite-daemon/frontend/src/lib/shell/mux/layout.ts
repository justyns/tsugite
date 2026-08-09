/**
 * Pure layout model for the spaces multiplexer: a tree of split nodes (row/col
 * with per-child size weights) whose leaves hold a stack of docked surface tabs.
 *
 * Everything here is a pure function - each op deep-clones its input and returns
 * a new `Layout`, so the store can treat layouts as immutable snapshots and the
 * reducers are trivially unit-testable in node (no DOM, no Svelte). The docking
 * UI, persistence, and keyboard paths all route through these ops.
 *
 * Invariants the ops maintain (and `deserializeLayout` enforces on load):
 *  - the root always exists; the only pane allowed to be empty is a lone root leaf.
 *  - a split has >= 2 children; a single-child split is collapsed (hoisted) away.
 *  - `sizes.length === children.length`, every size > 0, and a split's sizes sum to 1.
 *  - `activeTabId` is null iff the pane has no tabs, else it names an existing tab.
 *  - `focusedPaneId` names an existing leaf.
 */

export const LAYOUT_SCHEMA_VERSION = 1;

/** row = children side by side (a left/right split); col = stacked (top/bottom). */
export type SplitDir = 'row' | 'col';

/** Mirrors TabStrip's `TabState` union (kept local so this stays a DOM-free,
 *  node-testable module). The Mux maps these straight onto the tab-strip dots. */
export type PaneTabState = 'busy' | 'idle' | 'streaming' | 'blocked' | 'error' | 'done';

/** What a caller hands the model to open/dock: a surface reference. The model
 *  assigns the tab's tree id; identity for dedupe is the {kind, params} key. */
export interface SurfaceRef {
  kind: string;
  params?: Record<string, string>;
  title?: string;
  state?: PaneTabState;
  /** Open as a VSCode-style preview: the pane's single reusable "ephemeral" tab
   *  that the next preview replaces in place, until it is pinned. See openPreview. */
  ephemeral?: boolean;
}

export interface PaneTabModel {
  id: string;
  kind: string;
  params: Record<string, string>;
  title?: string;
  state?: PaneTabState;
  /** True while this is the pane's reusable preview tab (see openPreview/pinTab). */
  ephemeral?: boolean;
}

export interface LeafNode {
  type: 'leaf';
  id: string;
  tabs: PaneTabModel[];
  activeTabId: string | null;
}

export interface SplitNode {
  type: 'split';
  id: string;
  dir: SplitDir;
  children: LayoutNode[];
  sizes: number[];
}

export type LayoutNode = LeafNode | SplitNode;

export interface Layout {
  version: number;
  root: LayoutNode;
  focusedPaneId: string | null;
}

/** Smallest fraction a pane may shrink to under a resize (also the guard that
 *  keeps a divider drag from collapsing a neighbour to zero). */
const MIN_SIZE = 0.05;

let idSeq = 0;
function uid(prefix: string): string {
  idSeq += 1;
  return `${prefix}-${idSeq.toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

/** Stable identity for a surface, order-independent in its params, used to
 *  de-duplicate docking the same surface into the same pane twice. */
export function surfaceKey(ref: { kind: string; params?: Record<string, string> }): string {
  const params = ref.params ?? {};
  const flat = Object.keys(params)
    .sort()
    .map((k) => `${k}=${params[k]}`)
    .join('&');
  return `${ref.kind}\u0000${flat}`;
}

function toTab(ref: SurfaceRef): PaneTabModel {
  return {
    id: uid('tab'),
    kind: ref.kind,
    params: { ...(ref.params ?? {}) },
    ...(ref.title != null ? { title: ref.title } : {}),
    ...(ref.state != null ? { state: ref.state } : {}),
    ...(ref.ephemeral != null ? { ephemeral: ref.ephemeral } : {}),
  };
}

function makeLeaf(tabs: PaneTabModel[] = []): LeafNode {
  return {
    type: 'leaf',
    id: uid('pane'),
    tabs,
    activeTabId: tabs.length ? tabs[tabs.length - 1]!.id : null,
  };
}

export function defaultLayout(): Layout {
  const root = makeLeaf();
  return { version: LAYOUT_SCHEMA_VERSION, root, focusedPaneId: root.id };
}

/** Deep clone that tolerates a Svelte `$state` proxy being passed in (which
 *  `structuredClone` rejects with a DataCloneError). The layout is pure JSON, so
 *  the JSON fallback is lossless. Callers on the reactive side should still hand
 *  in a `$state.snapshot`, but this keeps the ops from crashing if they don't. */
function deepClone<T>(value: T): T {
  try {
    return structuredClone(value);
  } catch {
    return JSON.parse(JSON.stringify(value)) as T;
  }
}

function clone(layout: Layout): Layout {
  return deepClone(layout);
}

// ---------- traversal ----------

export function collectLeaves(node: LayoutNode): LeafNode[] {
  if (node.type === 'leaf') return [node];
  return node.children.flatMap(collectLeaves);
}

function findLeafIn(node: LayoutNode, paneId: string): LeafNode | null {
  if (node.type === 'leaf') return node.id === paneId ? node : null;
  for (const child of node.children) {
    const hit = findLeafIn(child, paneId);
    if (hit) return hit;
  }
  return null;
}

export function findLeaf(layout: Layout, paneId: string): LeafNode | null {
  return findLeafIn(layout.root, paneId);
}

/** The split that directly contains `paneId`, plus the child index, or null when
 *  the target is the root itself (no parent). */
function findParent(root: LayoutNode, paneId: string): { parent: SplitNode; index: number } | null {
  if (root.type === 'leaf') return null;
  for (let i = 0; i < root.children.length; i++) {
    const child = root.children[i]!;
    if (child.id === paneId) return { parent: root, index: i };
    const deeper = findParent(child, paneId);
    if (deeper) return deeper;
  }
  return null;
}

// ---------- open / dock ----------

function activateOrAdd(leaf: LeafNode, ref: SurfaceRef): void {
  const key = surfaceKey(ref);
  const existing = leaf.tabs.find((t) => surfaceKey(t) === key);
  if (existing) {
    leaf.activeTabId = existing.id;
    return;
  }
  const tab = toTab(ref);
  leaf.tabs.push(tab);
  leaf.activeTabId = tab.id;
}

export function dockAsTab(layout: Layout, paneId: string, ref: SurfaceRef): Layout {
  const next = clone(layout);
  const leaf = findLeafIn(next.root, paneId);
  if (!leaf) return next;
  activateOrAdd(leaf, ref);
  next.focusedPaneId = leaf.id;
  return next;
}

/** Open a surface, resolving the target pane: explicit id, else the focused
 *  pane, else the first leaf. Used by the palette / row-click path. */
/**
 * Rail-click semantics: point the existing surface of `ref.kind` at `ref`
 * (params/title update in place, tab id stable so the mounted surface follows)
 * instead of stacking a new tab per selection. Prefers the focused pane's tab
 * of the kind (active tab first), then any pane's; with none anywhere, opens a
 * new tab like openInPane. Drag-to-dock still creates fresh tabs.
 */
export function retargetOrOpen(layout: Layout, ref: SurfaceRef): Layout {
  const next = clone(layout);
  const leaves = collectLeaves(next.root);
  const focused = (next.focusedPaneId && findLeafIn(next.root, next.focusedPaneId)) || leaves[0];
  const ordered = focused ? [focused, ...leaves.filter((l) => l !== focused)] : leaves;
  for (const leaf of ordered) {
    const active = leaf.tabs.find((t) => t.id === leaf.activeTabId);
    const tab = active?.kind === ref.kind ? active : leaf.tabs.find((t) => t.kind === ref.kind);
    if (!tab) continue;
    tab.params = { ...(ref.params ?? {}) };
    if (ref.title != null) tab.title = ref.title;
    if (ref.state != null) tab.state = ref.state;
    leaf.activeTabId = tab.id;
    next.focusedPaneId = leaf.id;
    return next;
  }
  return openInPane(layout, ref);
}

/** Rename a docked tab, wherever it lives. Keyed by tab id alone because the
 *  caller is the surface mounted in it, which knows its own id but not its
 *  pane's - and a tab can be dragged to another pane while it's open. */
export function retitleTab(layout: Layout, tabId: string, title: string): Layout {
  const next = clone(layout);
  for (const leaf of collectLeaves(next.root)) {
    const tab = leaf.tabs.find((t) => t.id === tabId);
    if (tab) {
      tab.title = title;
      return next;
    }
  }
  return next;
}

export function openInPane(layout: Layout, ref: SurfaceRef, targetPaneId?: string): Layout {
  const target =
    (targetPaneId && findLeaf(layout, targetPaneId)?.id) ||
    (layout.focusedPaneId && findLeaf(layout, layout.focusedPaneId)?.id) ||
    collectLeaves(layout.root)[0]!.id;
  return dockAsTab(layout, target, ref);
}

/**
 * VSCode-style preview open: route `ref` into the target pane's single reusable
 * *ephemeral* tab. Resolves the pane like openInPane (explicit id, else focused,
 * else first leaf), then:
 *  1. if a NON-ephemeral (pinned) tab of this surface already exists, just
 *     activate it - clicking an already-open file focuses it, never previews a
 *     duplicate;
 *  2. else retarget the pane's ephemeral tab in place - same tab id, so the
 *     mounted surface follows the param change - keeping it ephemeral;
 *  3. else open a fresh ephemeral tab.
 */
export function openPreview(layout: Layout, ref: SurfaceRef, targetPaneId?: string): Layout {
  const next = clone(layout);
  const target =
    (targetPaneId && findLeafIn(next.root, targetPaneId)) ||
    (next.focusedPaneId && findLeafIn(next.root, next.focusedPaneId)) ||
    collectLeaves(next.root)[0]!;
  const key = surfaceKey(ref);

  const pinned = target.tabs.find((t) => !t.ephemeral && surfaceKey(t) === key);
  if (pinned) {
    target.activeTabId = pinned.id;
    next.focusedPaneId = target.id;
    return next;
  }

  const preview = target.tabs.find((t) => t.ephemeral);
  if (preview) {
    preview.kind = ref.kind;
    preview.params = { ...(ref.params ?? {}) };
    if (ref.title != null) preview.title = ref.title;
    else delete preview.title;
    if (ref.state != null) preview.state = ref.state;
    else delete preview.state;
    // preview.ephemeral stays true - this is the reused preview slot.
    target.activeTabId = preview.id;
    next.focusedPaneId = target.id;
    return next;
  }

  const tab = toTab({ ...ref, ephemeral: true });
  target.tabs.push(tab);
  target.activeTabId = tab.id;
  next.focusedPaneId = target.id;
  return next;
}

/** Pin a tab (VSCode "keep open"): drop its ephemeral flag so the next preview
 *  opens a fresh tab instead of replacing it. No-op if the pane/tab is gone. */
export function pinTab(layout: Layout, paneId: string, tabId: string): Layout {
  const next = clone(layout);
  const tab = findLeafIn(next.root, paneId)?.tabs.find((t) => t.id === tabId);
  if (tab) delete tab.ephemeral;
  return next;
}

// ---------- split ----------

export function splitPane(
  layout: Layout,
  paneId: string,
  dir: SplitDir,
  ref: SurfaceRef,
  position: 'before' | 'after' = 'after',
): Layout {
  const next = clone(layout);
  const target = findLeafIn(next.root, paneId);
  if (!target) return next;
  const newLeaf = makeLeaf([toTab(ref)]);

  const found = findParent(next.root, paneId);
  if (found && found.parent.dir === dir) {
    // Same-axis split: insert a flat sibling, halving the target's weight so the
    // split's total is preserved and the two resulting panes start even.
    const { parent, index } = found;
    const half = parent.sizes[index]! / 2;
    parent.sizes[index] = half;
    const insertAt = position === 'before' ? index : index + 1;
    parent.children.splice(insertAt, 0, newLeaf);
    parent.sizes.splice(insertAt, 0, half);
  } else {
    // Cross-axis (or root) split: wrap the target in a fresh 2-child split.
    const pair = position === 'before' ? [newLeaf, target] : [target, newLeaf];
    const wrapper: SplitNode = {
      type: 'split',
      id: uid('split'),
      dir,
      children: pair,
      sizes: [0.5, 0.5],
    };
    if (!found) {
      next.root = wrapper;
    } else {
      found.parent.children[found.index] = wrapper;
    }
  }
  next.focusedPaneId = newLeaf.id;
  return next;
}

// ---------- close / collapse ----------

/** Collapse any split that has been reduced to a single child by hoisting that
 *  child into the split's slot. Runs bottom-up so chains collapse in one pass. */
function collapse(node: LayoutNode): LayoutNode {
  if (node.type === 'leaf') return node;
  node.children = node.children.map(collapse);
  if (node.children.length === 1) return node.children[0]!;
  return node;
}

function normalizeSizes(node: LayoutNode): void {
  if (node.type === 'leaf') return;
  const sum = node.sizes.reduce((a, b) => a + b, 0);
  if (sum <= 0) {
    node.sizes = node.children.map(() => 1 / node.children.length);
  } else {
    node.sizes = node.sizes.map((s) => s / sum);
  }
  node.children.forEach(normalizeSizes);
}

/** Focusable leaf id nearest a removed pane: prefer the previous sibling subtree,
 *  else the next, else any surviving leaf. */
function pickFocusAfterRemoval(root: LayoutNode, parent: SplitNode, removedIndex: number): string {
  const sibling = parent.children[removedIndex] ?? parent.children[removedIndex - 1];
  if (sibling) return collectLeaves(sibling)[0]!.id;
  return collectLeaves(root)[0]!.id;
}

/** Close every tab in the pane except `tabId` (tab context menu). */
export function closeOtherTabs(layout: Layout, paneId: string, tabId: string): Layout {
  const next = clone(layout);
  const leaf = findLeafIn(next.root, paneId);
  if (!leaf) return next;
  const keep = leaf.tabs.find((t) => t.id === tabId);
  if (!keep) return next;
  leaf.tabs = [keep];
  leaf.activeTabId = keep.id;
  next.focusedPaneId = leaf.id;
  return next;
}

/** Close every tab in the pane (tab context menu); an empty root pane stays,
 *  a split member collapses away like the last closeTab would. */
export function closeAllTabs(layout: Layout, paneId: string): Layout {
  let next = layout;
  const leaf = findLeafIn(next.root, paneId);
  if (!leaf) return next;
  for (const tab of [...leaf.tabs]) next = closeTab(next, paneId, tab.id);
  return next;
}

export function closeTab(layout: Layout, paneId: string, tabId: string): Layout {
  const next = clone(layout);
  const leaf = findLeafIn(next.root, paneId);
  if (!leaf) return next;
  const idx = leaf.tabs.findIndex((t) => t.id === tabId);
  if (idx === -1) return next;

  leaf.tabs.splice(idx, 1);
  if (leaf.tabs.length > 0) {
    // Reactivate a neighbour if the closed tab was active.
    if (leaf.activeTabId === tabId) {
      const neighbour = leaf.tabs[idx] ?? leaf.tabs[idx - 1] ?? leaf.tabs[0]!;
      leaf.activeTabId = neighbour.id;
    }
    return next;
  }

  // The pane is now empty.
  leaf.activeTabId = null;
  const found = findParent(next.root, paneId);
  if (!found) {
    // Lone root leaf: keep it (empty pane shows the drag hint).
    next.focusedPaneId = leaf.id;
    return next;
  }
  const { parent, index } = found;
  parent.children.splice(index, 1);
  parent.sizes.splice(index, 1);
  const refocus = pickFocusAfterRemoval(next.root, parent, index);
  next.root = collapse(next.root);
  normalizeSizes(next.root);
  next.focusedPaneId = findLeafIn(next.root, refocus)?.id ?? collectLeaves(next.root)[0]!.id;
  return next;
}

// ---------- move ----------

export function moveTab(
  layout: Layout,
  fromPaneId: string,
  tabId: string,
  toPaneId: string,
  position: 'before' | 'after' | number = 'after',
): Layout {
  const source = findLeaf(layout, fromPaneId);
  if (!source) return clone(layout);
  const tab = source.tabs.find((t) => t.id === tabId);
  if (!tab) return clone(layout);

  if (fromPaneId === toPaneId) {
    // Reorder within the pane.
    const next = clone(layout);
    const leaf = findLeafIn(next.root, toPaneId)!;
    const from = leaf.tabs.findIndex((t) => t.id === tabId);
    const [moved] = leaf.tabs.splice(from, 1);
    const to =
      typeof position === 'number'
        ? Math.max(0, Math.min(leaf.tabs.length, position))
        : position === 'before'
          ? Math.max(0, from - 1)
          : Math.min(leaf.tabs.length, from + 1);
    leaf.tabs.splice(to, 0, moved!);
    leaf.activeTabId = moved!.id;
    return next;
  }

  // Cross-pane: drop into the target (preserving the tab's identity), then close
  // it out of the source - which collapses the source pane if it empties.
  let next = clone(layout);
  const target = findLeafIn(next.root, toPaneId);
  if (!target) return next;
  const key = surfaceKey(tab);
  const dup = target.tabs.find((t) => surfaceKey(t) === key);
  if (dup) {
    target.activeTabId = dup.id;
  } else {
    target.tabs.push(deepClone(tab));
    target.activeTabId = tab.id;
  }
  next = closeTab(next, fromPaneId, tabId);
  next.focusedPaneId = findLeafIn(next.root, toPaneId)?.id ?? next.focusedPaneId;
  return next;
}

// ---------- resize ----------

export function resizeSplit(
  layout: Layout,
  splitId: string,
  dividerIndex: number,
  deltaFraction: number,
): Layout {
  const next = clone(layout);
  const split = findSplit(next.root, splitId);
  if (!split) return next;
  const a = split.sizes[dividerIndex];
  const b = split.sizes[dividerIndex + 1];
  if (a == null || b == null) return next;
  const pair = a + b;
  // Keep both panes strictly positive. Each side normally floors at MIN_SIZE, but
  // an over-split neighbourhood can leave the pair itself below 2*MIN_SIZE, where
  // that floor can't be honoured for both; there we fall back to the largest floor
  // the pair can seat (pair/2). Either way both weights stay > 0 and their total is
  // preserved, so the result never fails its own deserializer (which would wipe the
  // whole layout back to a blank pane on the next load).
  const floor = Math.min(MIN_SIZE, pair / 2);
  const newA = Math.max(floor, Math.min(pair - floor, a + deltaFraction));
  split.sizes[dividerIndex] = newA;
  split.sizes[dividerIndex + 1] = pair - newA;
  return next;
}

function findSplit(node: LayoutNode, splitId: string): SplitNode | null {
  if (node.type === 'leaf') return null;
  if (node.id === splitId) return node;
  for (const child of node.children) {
    const hit = findSplit(child, splitId);
    if (hit) return hit;
  }
  return null;
}

// ---------- focus / select ----------

export function focusPane(layout: Layout, paneId: string): Layout {
  if (!findLeaf(layout, paneId)) return clone(layout);
  const next = clone(layout);
  next.focusedPaneId = paneId;
  return next;
}

export function selectTab(layout: Layout, paneId: string, tabId: string): Layout {
  const next = clone(layout);
  const leaf = findLeafIn(next.root, paneId);
  if (!leaf || !leaf.tabs.some((t) => t.id === tabId)) return next;
  leaf.activeTabId = tabId;
  return next;
}

/** Step the focused pane's active tab one place in `dir` (the ⌘/Ctrl+Shift+[ ]
 *  chord). Clamped: a no-op at either end of the tab strip, on a pane with fewer
 *  than two tabs, or with no focusable leaf. Falls back to the first leaf when
 *  no pane is focused. */
export function cycleTab(layout: Layout, dir: 1 | -1): Layout {
  const next = clone(layout);
  const leaf =
    (next.focusedPaneId && findLeafIn(next.root, next.focusedPaneId)) ||
    collectLeaves(next.root)[0];
  if (!leaf || leaf.tabs.length < 2) return next;
  const i = leaf.tabs.findIndex((t) => t.id === leaf.activeTabId);
  const j = i + dir;
  if (i === -1 || j < 0 || j >= leaf.tabs.length) return next;
  leaf.activeTabId = leaf.tabs[j]!.id;
  return next;
}

// ---------- persistence ----------

export function serializeLayout(layout: Layout): string {
  return JSON.stringify(layout);
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function validTab(v: unknown): v is PaneTabModel {
  return (
    isRecord(v) && typeof v.id === 'string' && typeof v.kind === 'string' && isRecord(v.params)
  );
}

function validNode(node: unknown, ids: Set<string>, leafIds: Set<string>): boolean {
  if (!isRecord(node)) return false;
  if (node.type === 'leaf') {
    if (typeof node.id !== 'string' || ids.has(node.id)) return false;
    ids.add(node.id);
    leafIds.add(node.id);
    if (!Array.isArray(node.tabs) || !node.tabs.every(validTab)) return false;
    const active = node.activeTabId;
    if (active !== null && !(node.tabs as PaneTabModel[]).some((t) => t.id === active))
      return false;
    if (active === null && node.tabs.length > 0) return false;
    return true;
  }
  if (node.type === 'split') {
    if (typeof node.id !== 'string' || ids.has(node.id)) return false;
    ids.add(node.id);
    if (node.dir !== 'row' && node.dir !== 'col') return false;
    if (!Array.isArray(node.children) || node.children.length < 2) return false;
    if (!Array.isArray(node.sizes) || node.sizes.length !== node.children.length) return false;
    if (!node.sizes.every((s) => typeof s === 'number' && Number.isFinite(s) && s > 0))
      return false;
    return node.children.every((c) => validNode(c, ids, leafIds));
  }
  return false;
}

/** Parse a persisted layout, or fall back to a fresh default on ANY corruption:
 *  bad JSON, a schema-version mismatch, orphan single-child splits, size/child
 *  length drift, dangling active/focus ids, unknown node types. */
export function deserializeLayout(raw: string | null): Layout {
  if (!raw) return defaultLayout();
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || parsed.version !== LAYOUT_SCHEMA_VERSION) return defaultLayout();
    const ids = new Set<string>();
    const leafIds = new Set<string>();
    if (!validNode(parsed.root, ids, leafIds)) return defaultLayout();
    const focus = parsed.focusedPaneId;
    if (focus !== null && !(typeof focus === 'string' && leafIds.has(focus)))
      return defaultLayout();
    const layout = parsed as unknown as Layout;
    // Enforce the documented "a split's sizes sum to 1" invariant on load. validNode
    // only checks each size > 0, so a hand-persisted or older layout can carry
    // un-normalized weights; normalizing keeps render + resize math proportional and
    // matches what the reducers always emit.
    normalizeSizes(layout.root);
    return layout;
  } catch {
    // The recursive validate/normalize walk is unbounded, so a JSON-parseable but
    // pathologically deep tree can overflow the stack. Self-heal to a fresh default
    // (the documented "fall back on ANY corruption" contract) instead of letting the
    // throw propagate and white-screen the app at store construction.
    return defaultLayout();
  }
}
