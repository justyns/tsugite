const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/**
 * A compact "time since" label: "just now", then minutes, hours, days, then a short local
 * date once older than a month, with the year added once older than a year. Returns "" for a
 * missing or unparseable input so a caller can drop it or substitute its own wording.
 *
 * `style: 'bare'` drops the "ago" suffix ("12m", "now") for table cells and callers that
 * supply their own phrasing ("12m in queue").
 *
 * `now` defaults to the wall clock at call time (not reactive), so a caller that doesn't pass
 * one refreshes whenever its host re-renders rather than ticking on its own.
 *
 * The date fallback formats in local time on purpose: the activity feed buckets its day headers
 * locally (`activityFeed.ts` `dayKey`), so a UTC stamp would print a date that disagrees with
 * the local heading above it for anyone off UTC.
 */
export function formatAgo(
  iso: string | null | undefined,
  now: number = Date.now(),
  style: 'ago' | 'bare' = 'ago',
): string {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const delta = now - t;
  const suffix = style === 'ago' ? ' ago' : '';
  if (delta < MINUTE) return style === 'ago' ? 'just now' : 'now';
  if (delta < HOUR) return `${Math.floor(delta / MINUTE)}m${suffix}`;
  if (delta < DAY) return `${Math.floor(delta / HOUR)}h${suffix}`;
  if (delta < 30 * DAY) return `${Math.floor(delta / DAY)}d${suffix}`;
  const opts: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric' };
  if (delta >= 365 * DAY) opts.year = 'numeric';
  return new Intl.DateTimeFormat('en-US', opts).format(new Date(t)).toLowerCase();
}
