/**
 * Drag-to-reorder bookkeeping for a strip of items: which item is in flight, and
 * the insertion index the pointer is currently over. `dragover` exposes the
 * payload's types but not its contents, so the dragged id is tracked here rather
 * than read back off the DataTransfer.
 *
 * Callers keep their own eligibility guards (a strip that cannot reorder, a row
 * that is not pinned) and call in only once they pass, so refusing a drag stays
 * the caller's decision and this stays the bookkeeping.
 */
export class ReorderDrag {
  dragging = $state<string | null>(null);
  dropAt = $state<number | null>(null);

  #indexOf: (id: string) => number;
  #axis: 'x' | 'y';
  #stopPropagation: boolean;

  /**
   * @param indexOf Position of an id in the current order.
   * @param axis Which coordinate splits an item at its midpoint.
   * @param stopPropagation Set when a drop target sits above this one and would
   *   otherwise act on the same event (a tab strip inside a mux pane).
   */
  constructor(
    indexOf: (id: string) => number,
    { axis = 'x', stopPropagation = false }: { axis?: 'x' | 'y'; stopPropagation?: boolean } = {},
  ) {
    this.#indexOf = indexOf;
    this.#axis = axis;
    this.#stopPropagation = stopPropagation;
  }

  start(id: string): void {
    this.dragging = id;
  }

  end(): void {
    this.dragging = null;
    this.dropAt = null;
  }

  /** Track the insertion index for the item being hovered, splitting it at its midpoint. */
  over(event: DragEvent, id: string): void {
    if (this.dragging === null) return;
    event.preventDefault();
    if (this.#stopPropagation) event.stopPropagation();
    const r = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const past =
      this.#axis === 'y'
        ? event.clientY >= r.top + r.height / 2
        : event.clientX >= r.left + r.width / 2;
    this.dropAt = this.#indexOf(id) + (past ? 1 : 0);
  }

  /**
   * The move this drop performs, or null when the drag never started or lands
   * where it already was. Clears the drag either way.
   */
  drop(event: DragEvent): { id: string; insertAt: number } | null {
    const id = this.dragging;
    const insertAt = this.dropAt;
    if (id === null || insertAt === null) return null;
    event.preventDefault();
    if (this.#stopPropagation) event.stopPropagation();
    const from = this.#indexOf(id);
    this.end();
    return insertAt === from || insertAt === from + 1 ? null : { id, insertAt };
  }
}
