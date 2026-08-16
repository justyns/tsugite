/**
 * Cross-component drag transport for docking surfaces. A drag source (a session
 * row in the spaces rail, a file in the tree, a job card - all owned by the
 * chrome/views) writes a `SurfaceRef` under a private MIME type; the Mux panes
 * read it on drop. Using the native DataTransfer keeps the source and the drop
 * target fully decoupled - they share only this module, not any state object.
 */
import type { SurfaceRef } from './layout';

export const MUX_SURFACE_MIME = 'application/x-tsugite-surface';

export function writeSurfaceDrag(dt: DataTransfer, ref: SurfaceRef): void {
  dt.setData(MUX_SURFACE_MIME, JSON.stringify(ref));
  dt.effectAllowed = 'copyMove';
}

/** True when a drag carries a mux surface. Safe to call during `dragover`, where
 *  the payload itself is not yet readable but `types` is. */
export function hasSurfaceDrag(dt: DataTransfer | null): boolean {
  return !!dt && Array.from(dt.types).includes(MUX_SURFACE_MIME);
}

// Carried alongside the surface ref so a receiving pane can move the tab
// rather than dock a second copy.
export const MUX_TAB_MIME = 'application/x-tsugite-tab';

export interface TabOrigin {
  paneId: string;
  tabId: string;
}

export function writeTabDrag(dt: DataTransfer, origin: TabOrigin): void {
  dt.setData(MUX_TAB_MIME, JSON.stringify(origin));
}

export function readTabDrag(dt: DataTransfer | null): TabOrigin | null {
  if (!dt) return null;
  const raw = dt.getData(MUX_TAB_MIME);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    const o = parsed as TabOrigin;
    if (o && typeof o.paneId === 'string' && typeof o.tabId === 'string') return o;
  } catch {
    // malformed payload -> treat as a plain surface drag
  }
  return null;
}

export function readSurfaceDrag(dt: DataTransfer | null): SurfaceRef | null {
  if (!dt) return null;
  const raw = dt.getData(MUX_SURFACE_MIME);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && typeof (parsed as SurfaceRef).kind === 'string') {
      return parsed as SurfaceRef;
    }
  } catch {
    // malformed payload -> not a droppable surface
  }
  return null;
}
