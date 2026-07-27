/**
 * A compact "time since" label for a UI timestamp: "just now", then minutes,
 * hours, days. Returns "" for an unparseable input so a caller can drop it.
 *
 * Computed against the wall clock at call time (not reactive), so it refreshes
 * whenever its host re-renders rather than ticking on its own - fine for the
 * approximate "N ago" it produces.
 */
export function relativeTime(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 45) return 'just now';
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}
