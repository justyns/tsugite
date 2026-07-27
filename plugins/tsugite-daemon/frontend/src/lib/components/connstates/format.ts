/** Pure formatting helpers shared by the connstates components (kept out of
 * the .svelte files so they're directly unit-testable). */

/** "stale" with no timestamp, or "stale · Ns" once we know when it went stale. */
export function formatStale(sinceMs: number | null, nowMs: number): string {
  if (sinceMs == null) return 'stale';
  const seconds = Math.max(0, Math.round((nowMs - sinceMs) / 1000));
  return `stale · ${seconds}s`;
}

const SKELETON_WIDTHS = [72, 88, 55, 80, 65, 92];

/** Percentage widths for `lines` skeleton bars, cycling a fixed palette so the
 * loading pane keeps the same "realistic ragged text" look at any line count. */
export function paneSkeletonWidths(lines: number): number[] {
  return Array.from(
    { length: Math.max(0, lines) },
    // modulo of a non-empty fixed array is always in range
    (_, i) => SKELETON_WIDTHS[i % SKELETON_WIDTHS.length]!,
  );
}
