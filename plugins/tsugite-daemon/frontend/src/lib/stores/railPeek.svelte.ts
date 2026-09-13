/**
 * Hover-to-peek state for the two collapsed rails (nav and context). Nothing here
 * is persisted, and it never touches the collapsed flags in shellView.
 */
export type PeekKind = 'nav' | 'rail';

/** Pointer-intent delays, long enough that crossing a rail does not open it. */
const OPEN_DELAY = 150;
const CLOSE_DELAY = 150;

export class RailPeek {
  open = $state<PeekKind | null>(null);
  /** Mirrors `(hover: hover) and (pointer: fine)`. */
  hoverable = $state(false);
  private timer: ReturnType<typeof setTimeout> | null = null;

  enter(kind: PeekKind): void {
    if (!this.hoverable) return;
    this.clear();
    if (this.open === kind) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.open = kind;
    }, OPEN_DELAY);
  }

  leave(): void {
    this.clear();
    if (this.open === null) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.open = null;
    }, CLOSE_DELAY);
  }

  close(): void {
    this.clear();
    this.open = null;
  }

  private clear(): void {
    if (this.timer === null) return;
    clearTimeout(this.timer);
    this.timer = null;
  }
}

export const railPeek = new RailPeek();
