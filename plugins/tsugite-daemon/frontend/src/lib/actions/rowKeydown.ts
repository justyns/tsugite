import { nextRovingIndex } from './rovingNav';

/** Enter/Space select the row; arrows, Home and End move focus among `[data-rowidx]` rows of the enclosing listbox. */
export function rowKeydown<T>(
  e: KeyboardEvent,
  index: number,
  items: T[],
  onSelect: (item: T) => void,
): void {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    const item = items[index];
    if (item) onSelect(item);
    return;
  }
  const next = nextRovingIndex(index, e.key, items.length);
  if (next === null) return;
  e.preventDefault();
  const host = e.currentTarget as HTMLElement;
  const rows = host.closest('[role="listbox"]')?.querySelectorAll<HTMLElement>('[data-rowidx]');
  rows?.[next]?.focus();
}
