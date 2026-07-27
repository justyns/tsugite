/**
 * Pure drop-zone geometry for docking a dragged surface onto a pane. Splitting
 * the hit-testing out of the component keeps it node-testable and keeps the
 * DnD handlers a thin shell over these two functions.
 *
 * The left/right thirds of a pane split it (a new sibling pane), and the middle
 * band docks the surface as another tab in that pane.
 */
import type { SplitDir } from './layout';

export type DropZone = 'left' | 'right' | 'center';

/** Fraction of the pane width claimed by each edge (split) zone. */
export const DROP_EDGE = 1 / 3;

export function dropZoneAt(width: number, offsetX: number): DropZone {
  if (width <= 0) return 'center';
  const x = offsetX / width;
  if (x < DROP_EDGE) return 'left';
  if (x > 1 - DROP_EDGE) return 'right';
  return 'center';
}

export type DropIntent =
  { action: 'dock' } | { action: 'split'; dir: SplitDir; position: 'before' | 'after' };

export function dropIntent(zone: DropZone): DropIntent {
  if (zone === 'left') return { action: 'split', dir: 'row', position: 'before' };
  if (zone === 'right') return { action: 'split', dir: 'row', position: 'after' };
  return { action: 'dock' };
}
