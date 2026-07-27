/** Pure formatting helpers for the feedback components (kept out of the
 * .svelte files so they're directly unit-testable). */

/** mm:ss elapsed-time label - unbounded minutes, zero-padded, floors
 * fractional seconds, never goes negative. */
export function formatElapsed(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}
