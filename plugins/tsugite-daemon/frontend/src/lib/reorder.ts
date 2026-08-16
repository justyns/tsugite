/** Move one item to an insertion point expressed in the ORIGINAL list's
 *  coordinates, which is what a drop between two rows gives you. */
export function moveItem<T>(items: T[], from: number, insertAt: number): T[] {
  if (from < 0 || from >= items.length) return items;
  const next = items.slice();
  const [moved] = next.splice(from, 1);
  const to = from < insertAt ? insertAt - 1 : insertAt;
  next.splice(Math.max(0, Math.min(next.length, to)), 0, moved!);
  return next;
}
