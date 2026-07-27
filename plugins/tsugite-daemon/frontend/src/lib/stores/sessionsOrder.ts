/**
 * Pure list helpers for the sessions store: pinned ordering + id-keyed upsert /
 * patch. Kept DOM-free and node-testable; the store wires them to $state and the
 * SSE broadcast. The server already returns the unpinned tail in recency order,
 * so ordering only has to float pinned rows to the top by pin_position.
 */

export interface SessionRowLike {
  id: string;
  pinned?: boolean;
  pin_position?: number | null;
  [key: string]: unknown;
}

/** Pinned rows first (ascending pin_position, nulls last), then the rest in the
 *  order the server returned them (recency). Stable for equal keys. */
export function orderSessions<T extends SessionRowLike>(rows: T[]): T[] {
  const pinned = rows.filter((r) => r.pinned);
  const rest = rows.filter((r) => !r.pinned);
  pinned.sort((a, b) => {
    const ap = a.pin_position ?? Number.POSITIVE_INFINITY;
    const bp = b.pin_position ?? Number.POSITIVE_INFINITY;
    return ap - bp;
  });
  return [...pinned, ...rest];
}

/** Shallow-merge a patch onto the row with the given id (no-op if absent). */
export function patchRow<T extends SessionRowLike>(rows: T[], id: string, patch: Partial<T>): T[] {
  const idx = rows.findIndex((r) => r.id === id);
  if (idx === -1) return rows;
  const next = rows.slice();
  // idx is in range, so the row exists; the cast is the generic-merge escape
  // hatch (T & Partial<T> doesn't narrow to T structurally).
  next[idx] = { ...rows[idx], ...patch } as T;
  return next;
}

/** Reorder pinned rows to match an explicit id sequence (drag-reorder echo),
 *  leaving unknown ids and unpinned rows where they are. */
export function reorderPins<T extends SessionRowLike>(rows: T[], ids: string[]): T[] {
  const position = new Map(ids.map((id, i) => [id, i]));
  return orderSessions(
    rows.map((r) => (position.has(r.id) ? { ...r, pin_position: position.get(r.id)! } : r)),
  );
}
