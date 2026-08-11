/**
 * Shell view + chrome state: which nav view is active, the last workspace view
 * (so the shared context rail restores when returning from a full view), and the
 * two collapse axes (nav rail, and the context rail per workspace type). Persisted
 * to localStorage so a reload lands exactly where the user left off.
 *
 * The active view is mirrored from the hash router (App wires that), but kept here
 * so `workspaceView` and the rail highlight read from one place. Which region a
 * view gets is the view registry's call (views/index.ts), not this store's.
 * Exported as a class instance - never a reassigned binding.
 */
import { readLocal, writeLocal } from '$lib/storage';

export type WorkspaceView = 'chats' | 'terminals' | 'files';

/** The three views that share the workspace (context rail + one mux tab area). */
const WORKSPACE_VIEWS: readonly WorkspaceView[] = ['chats', 'terminals', 'files'];

export function isWorkspaceView(id: string): id is WorkspaceView {
  return (WORKSPACE_VIEWS as readonly string[]).includes(id);
}

const VIEW_KEY = 'tsugite_shell_view';
const NAV_KEY = 'tsugite_nav_collapsed';
const RAIL_KEY = 'tsugite_rail_collapsed';

type RailCollapse = Record<WorkspaceView, boolean>;

function readRailCollapse(): RailCollapse {
  const base: RailCollapse = { chats: false, terminals: false, files: false };
  const raw = readLocal(RAIL_KEY);
  if (!raw) return base;
  try {
    const parsed = JSON.parse(raw) as Partial<RailCollapse>;
    for (const v of WORKSPACE_VIEWS) if (typeof parsed[v] === 'boolean') base[v] = parsed[v]!;
  } catch {
    // corrupt entry - keep the all-expanded default
  }
  return base;
}

export class ShellViewStore {
  /** The nav view currently selected (drives the region shown + rail highlight). */
  activeViewId = $state<string>('chats');
  /** The most recent workspace view - the context rail renders this even while a
   *  full view shows, so switching back restores the right rail. */
  workspaceView = $state<WorkspaceView>('chats');
  navCollapsed = $state(false);
  private railCollapsed = $state<RailCollapse>({ chats: false, terminals: false, files: false });

  constructor() {
    const stored = readLocal(VIEW_KEY);
    if (stored) {
      this.activeViewId = stored;
      if (isWorkspaceView(stored)) this.workspaceView = stored;
    }
    this.navCollapsed = readLocal(NAV_KEY) === '1';
    this.railCollapsed = readRailCollapse();
  }

  /** Select a nav view. A workspace view also becomes the active context rail. */
  activate(id: string): void {
    this.activeViewId = id;
    if (isWorkspaceView(id)) this.workspaceView = id;
    writeLocal(VIEW_KEY, id);
  }

  toggleNav(): void {
    this.navCollapsed = !this.navCollapsed;
    writeLocal(NAV_KEY, this.navCollapsed ? '1' : '0');
  }

  isRailCollapsed(view: WorkspaceView = this.workspaceView): boolean {
    return this.railCollapsed[view];
  }

  toggleRail(view: WorkspaceView = this.workspaceView): void {
    this.railCollapsed = { ...this.railCollapsed, [view]: !this.railCollapsed[view] };
    writeLocal(RAIL_KEY, JSON.stringify(this.railCollapsed));
  }
}

export const shellView = new ShellViewStore();
