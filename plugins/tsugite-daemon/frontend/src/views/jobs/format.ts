/** Relative-time formatting for job rows/cards. Pure and node-testable. */

/** Coarse magnitude since `iso` ("now" under 45s, then Nm / Nh / Nd). Empty
 *  string for a missing/unparseable timestamp. */
export function relativeTime(iso: string | null | undefined, nowMs: number): string {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const sec = Math.max(0, Math.round((nowMs - t) / 1000));
  if (sec < 45) return 'now';
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h`;
  return `${Math.round(hr / 24)}d`;
}

/** As `relativeTime`, but reads as an elapsed phrase ("now", "12m ago"). */
export function relativeAgo(iso: string | null | undefined, nowMs: number): string {
  const rt = relativeTime(iso, nowMs);
  if (!rt || rt === 'now') return rt;
  return `${rt} ago`;
}
