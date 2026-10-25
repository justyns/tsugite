/**
 * The context rail's dragged width in px, persisted to localStorage
 * (`tsugite_rail_width`). Null until the person resizes it, so the CSS default
 * applies. The 60% ceiling lives in the work-shell grid.
 */
import { readLocal, removeLocal, writeLocal } from '$lib/storage';

const KEY = 'tsugite_rail_width';
export const RAIL_MIN_PX = 200;

class RailWidth {
  px = $state<number | null>(Number(readLocal(KEY)) || null);

  set(px: number): void {
    this.px = Math.max(RAIL_MIN_PX, Math.round(px));
    writeLocal(KEY, String(this.px));
  }

  reset(): void {
    this.px = null;
    removeLocal(KEY);
  }
}

export const railWidth = new RailWidth();
