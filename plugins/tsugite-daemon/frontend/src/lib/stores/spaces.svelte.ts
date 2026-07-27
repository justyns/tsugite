/**
 * Spaces store: named spaces, each owning a multiplexer layout, plus the active
 * space and a per-space state rollup. Persisted to localStorage under a
 * schema-versioned envelope; a corrupt envelope falls back to a single default
 * space and a single corrupt layout self-heals to a default (without dropping
 * its sibling spaces).
 *
 * The parse / serialize / rollup helpers are pure (node-unit-tested); the class
 * wires them to `$state` + localStorage and delegates every mutation to the pure
 * layout reducers in `$lib/shell/mux/layout`. Exported as a class instance -
 * never a reassigned binding.
 */
import {
  type Layout,
  type PaneTabModel,
  type SplitDir,
  type SurfaceRef,
  closeAllTabs,
  closeOtherTabs,
  closeTab,
  collectLeaves,
  cycleTab,
  defaultLayout,
  deserializeLayout,
  dockAsTab,
  findLeaf,
  focusPane,
  moveTab,
  openInPane,
  openPreview,
  pinTab,
  resizeSplit,
  retargetOrOpen,
  selectTab,
  splitPane,
} from '$lib/shell/mux/layout';
import { readLocal, writeLocal } from '$lib/storage';

export const SPACES_SCHEMA_VERSION = 1;

export type SpaceRollupState = 'working' | 'idle' | 'blocked' | 'done';
export type SpaceRollup = Record<SpaceRollupState, number>;

export interface Space {
  id: string;
  name: string;
  layout: Layout;
}

interface SpacesState {
  spaces: Space[];
  activeSpaceId: string;
}

const KEY = 'tsugite_spaces';

let seq = 0;
function spaceId(): string {
  seq += 1;
  return `space-${seq.toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

/** The daily-default space: one pane holding a single chat surface. */
export function defaultSpace(name = 'Main'): Space {
  const base = defaultLayout();
  // The 'chat' surface kind is the docking alias of the default view
  // (views' DEFAULT_VIEW_ID = 'chats'), mapped by shellNav's SURFACE_FOR_VIEW.
  const layout = dockAsTab(base, base.root.id, { kind: 'chat', title: 'Chat' });
  return { id: spaceId(), name, layout };
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function freshState(): SpacesState {
  const s = defaultSpace();
  return { spaces: [s], activeSpaceId: s.id };
}

export function serializeSpaces(spaces: Space[], activeSpaceId: string): string {
  return JSON.stringify({
    version: SPACES_SCHEMA_VERSION,
    activeSpaceId,
    spaces: spaces.map((s) => ({
      id: s.id,
      name: s.name,
      // Callers pass a $state.snapshot, so the layout is already a plain object;
      // the outer JSON.stringify serializes it directly.
      layout: s.layout,
    })),
  });
}

export function parseSpaces(raw: string | null): SpacesState {
  if (!raw) return freshState();
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return freshState();
  }
  if (!isRecord(parsed) || parsed.version !== SPACES_SCHEMA_VERSION) return freshState();
  if (!Array.isArray(parsed.spaces) || parsed.spaces.length === 0) return freshState();

  const spaces: Space[] = [];
  for (const s of parsed.spaces) {
    // Skip a malformed entry (non-string id/name) rather than discarding the whole
    // list - its valid siblings and their layouts survive. A corrupt layout inside
    // an otherwise-valid entry self-heals to a default (deserializeLayout owns the
    // structural validation).
    if (!isRecord(s) || typeof s.id !== 'string' || typeof s.name !== 'string') continue;
    spaces.push({
      id: s.id,
      name: s.name,
      layout: deserializeLayout(JSON.stringify(s.layout ?? null)),
    });
  }
  if (spaces.length === 0) return freshState();
  const activeSpaceId =
    typeof parsed.activeSpaceId === 'string' && spaces.some((s) => s.id === parsed.activeSpaceId)
      ? parsed.activeSpaceId
      : spaces[0]!.id;
  return { spaces, activeSpaceId };
}

const DEFAULT_ROLLUP: (tab: PaneTabModel) => SpaceRollupState | null = (tab) => {
  switch (tab.state) {
    case 'busy':
    case 'streaming':
      return 'working';
    case 'blocked':
    case 'error':
      return 'blocked';
    case 'idle':
      return 'idle';
    case 'done':
      return 'done';
    default:
      return null;
  }
};

/**
 * Per-space state rollup (working / idle / blocked / done counts).
 *
 * Placeholder derivation: counts docked tabs by their declared `state`. The
 * live counts render here once session data is wired - the chrome will pass a
 * `resolve` that maps a tab to its authoritative session state instead of the
 * tab's own cached state.
 */
export function computeRollup(
  space: Space,
  resolve: (tab: PaneTabModel) => SpaceRollupState | null = DEFAULT_ROLLUP,
): SpaceRollup {
  const roll: SpaceRollup = { working: 0, idle: 0, blocked: 0, done: 0 };
  for (const leaf of collectLeaves(space.layout.root)) {
    for (const tab of leaf.tabs) {
      const bucket = resolve(tab);
      if (bucket) roll[bucket] += 1;
    }
  }
  return roll;
}

/** Trailing debounce for layout writes - coalesces a drag's mutation burst. */
const PERSIST_DEBOUNCE_MS = 150;

export class SpacesStore {
  spaces = $state<Space[]>([]);
  activeSpaceId = $state('');
  private persistTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    const initial = parseSpaces(readLocal(KEY));
    this.spaces = initial.spaces;
    this.activeSpaceId = initial.activeSpaceId;
    // A debounced layout write can still be pending when the tab is hidden or
    // closed; flush it first so the last change survives. Guarded for the node
    // test env, which has no window/document.
    if (typeof window !== 'undefined') {
      window.addEventListener('pagehide', () => this.flush());
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') this.flush();
      });
    }
  }

  get active(): Space {
    return this.spaces.find((s) => s.id === this.activeSpaceId) ?? this.spaces[0]!;
  }

  rollup(space: Space = this.active): SpaceRollup {
    return computeRollup(space);
  }

  // --- layout mutations (all route through the pure reducers on the active space) ---

  private apply(fn: (layout: Layout) => Layout): void {
    const space = this.active;
    // Snapshot to hand the reducers a plain (non-proxied) layout, then let the
    // reassignment re-proxy the result.
    space.layout = fn($state.snapshot(space.layout) as Layout);
    this.persist();
  }

  dock(paneId: string, ref: SurfaceRef): void {
    this.apply((l) => dockAsTab(l, paneId, ref));
  }
  open(ref: SurfaceRef, targetPaneId?: string): void {
    this.apply((l) => openInPane(l, ref, targetPaneId));
  }
  /** Rail-click semantics: reuse (retarget) the existing surface of ref.kind
   *  instead of stacking a tab per selection; drag-to-dock makes new tabs. */
  openReusing(ref: SurfaceRef): void {
    this.apply((l) => retargetOrOpen(l, ref));
  }
  /** VSCode-style preview open: route into the pane's single reusable ephemeral
   *  tab (the next preview replaces it) unless the surface is already pinned. */
  openPreview(ref: SurfaceRef, targetPaneId?: string): void {
    this.apply((l) => openPreview(l, ref, targetPaneId));
  }
  /** Pin a tab so the next preview opens fresh instead of replacing it. */
  pinTab(paneId: string, tabId: string): void {
    this.apply((l) => pinTab(l, paneId, tabId));
  }
  /** Pin the focused pane's preview tab (a file's double-click-to-keep). No-op
   *  when the focused pane holds no preview - e.g. the file was already pinned. */
  pinPreviewInFocusedPane(): void {
    this.apply((l) => {
      const paneId = l.focusedPaneId;
      if (!paneId) return l;
      const preview = findLeaf(l, paneId)?.tabs.find((t) => t.ephemeral);
      return preview ? pinTab(l, paneId, preview.id) : l;
    });
  }
  split(paneId: string, dir: SplitDir, ref: SurfaceRef, position?: 'before' | 'after'): void {
    this.apply((l) => splitPane(l, paneId, dir, ref, position));
  }
  closeTab(paneId: string, tabId: string): void {
    this.apply((l) => closeTab(l, paneId, tabId));
  }
  closeOtherTabs(paneId: string, tabId: string): void {
    this.apply((l) => closeOtherTabs(l, paneId, tabId));
  }
  closeAllTabs(paneId: string): void {
    this.apply((l) => closeAllTabs(l, paneId));
  }
  selectTab(paneId: string, tabId: string): void {
    this.apply((l) => selectTab(l, paneId, tabId));
  }
  /** Step the focused pane's active tab (⌘/Ctrl+Shift+[ ]). */
  cycleTab(dir: 1 | -1): void {
    this.apply((l) => cycleTab(l, dir));
  }
  focusPane(paneId: string): void {
    this.apply((l) => focusPane(l, paneId));
  }
  // This method and its reducer are pre-built for a tab drag/reorder gesture; the
  // pointer affordance isn't shipped yet, so callers are keyboard/programmatic only.
  moveTab(
    fromPaneId: string,
    tabId: string,
    toPaneId: string,
    position?: 'before' | 'after' | number,
  ): void {
    this.apply((l) => moveTab(l, fromPaneId, tabId, toPaneId, position));
  }
  resize(splitId: string, dividerIndex: number, deltaFraction: number): void {
    this.apply((l) => resizeSplit(l, splitId, dividerIndex, deltaFraction));
  }

  // --- space management ---

  setActive(id: string): void {
    if (this.spaces.some((s) => s.id === id)) {
      this.activeSpaceId = id;
      this.persist();
    }
  }
  addSpace(name = 'New space'): string {
    const s = defaultSpace(name);
    this.spaces.push(s);
    this.activeSpaceId = s.id;
    this.persist();
    return s.id;
  }
  renameSpace(id: string, name: string): void {
    const s = this.spaces.find((x) => x.id === id);
    if (!s) return;
    s.name = name;
    this.persist();
  }
  removeSpace(id: string): void {
    // Never drop below one space - the shell always has somewhere to dock.
    if (this.spaces.length <= 1) return;
    const idx = this.spaces.findIndex((s) => s.id === id);
    if (idx === -1) return;
    this.spaces.splice(idx, 1);
    if (this.activeSpaceId === id) this.activeSpaceId = this.spaces[Math.max(0, idx - 1)]!.id;
    this.persist();
  }

  // Coalesce a burst of mutations (e.g. a divider drag) into one trailing write.
  private persist(): void {
    if (this.persistTimer !== null) clearTimeout(this.persistTimer);
    this.persistTimer = setTimeout(() => {
      this.persistTimer = null;
      this.writeNow();
    }, PERSIST_DEBOUNCE_MS);
  }

  /** Write a pending debounced change immediately (on tab hide/close). */
  flush(): void {
    if (this.persistTimer === null) return;
    clearTimeout(this.persistTimer);
    this.persistTimer = null;
    this.writeNow();
  }

  private writeNow(): void {
    writeLocal(KEY, serializeSpaces($state.snapshot(this.spaces) as Space[], this.activeSpaceId));
  }
}

export const spaces = new SpacesStore();
