/**
 * Pure drop-zone geometry for docking a dragged surface onto a pane. Splitting
 * the hit-testing out of the component keeps it node-testable and keeps the
 * DnD handlers a thin shell over these two functions.
 *
 * The outer thirds of a pane split it (a new sibling pane), and the middle
 * block docks the surface as another tab in that pane. A point in two edge
 * thirds (a corner) takes the nearer edge; an exact tie takes left/right.
 */
import type { SplitDir } from './layout';

export type DropZone = 'left' | 'right' | 'top' | 'bottom' | 'center';

/** Fraction of the pane width and height claimed by each edge (split) zone. */
export const DROP_EDGE = 1 / 3;

export function dropZoneAt(
  width: number,
  height: number,
  offsetX: number,
  offsetY: number,
): DropZone {
  if (width <= 0 || height <= 0) return 'center';
  const x = offsetX / width;
  const y = offsetY / height;
  const edges: [DropZone, number][] = [
    ['left', x],
    ['right', 1 - x],
    ['top', y],
    ['bottom', 1 - y],
  ];
  const [zone, distance] = edges.reduce((best, e) => (e[1] < best[1] ? e : best));
  return distance < DROP_EDGE ? zone : 'center';
}

export type DropIntent =
  { action: 'dock' } | { action: 'split'; dir: SplitDir; position: 'before' | 'after' };

export function dropIntent(zone: DropZone): DropIntent {
  if (zone === 'left') return { action: 'split', dir: 'row', position: 'before' };
  if (zone === 'right') return { action: 'split', dir: 'row', position: 'after' };
  if (zone === 'top') return { action: 'split', dir: 'col', position: 'before' };
  if (zone === 'bottom') return { action: 'split', dir: 'col', position: 'after' };
  return { action: 'dock' };
}
