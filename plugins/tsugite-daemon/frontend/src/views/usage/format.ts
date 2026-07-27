/** Pure formatting helpers for the Usage view (kept out of View.svelte so
 * they're directly unit-testable). Numbers arrive from /api/usage/*; `SUM()`
 * over a column with no non-null rows yields SQL NULL (e.g. cost_usd is
 * optional per usage record), so every formatter tolerates null/undefined. */

const USD = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function formatUsd(n: number | null | undefined): string {
  return USD.format(n ?? 0);
}

const RUNS = new Intl.NumberFormat('en-US');

export function formatRuns(n: number | null | undefined): string {
  return RUNS.format(n ?? 0);
}

const TOKEN_TIERS: [number, string][] = [
  [1_000_000_000, 'B'],
  [1_000_000, 'M'],
  [1_000, 'k'],
];

/** Compact token count: 999 -> "999", 412000 -> "412k", 40700000 -> "40.7M".
 * Rounds to one decimal and drops a trailing ".0" rather than forcing it. */
export function formatTokensCompact(n: number | null | undefined): string {
  const value = n ?? 0;
  const abs = Math.abs(value);
  for (const [div, suffix] of TOKEN_TIERS) {
    if (abs >= div) {
      const scaled = (Math.round((value / div) * 10) / 10).toFixed(1);
      return `${scaled.endsWith('.0') ? scaled.slice(0, -2) : scaled}${suffix}`;
    }
  }
  return String(Math.round(value));
}

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

/** "2026-07-12" -> "jul 12" (period='day' rows only). Parsed as plain text,
 * not `Date`, so it's immune to local-timezone shift. Anything else (a
 * period='week'/'month' bucket key) passes through unchanged. */
export function formatDayLabel(period: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(period);
  if (!match) return period;
  const month = MONTHS[Number(match[2]) - 1];
  if (!month) return period;
  return `${month} ${match[3]}`;
}

/** ISO timestamp (a schedule's last_run) -> "jul 15 08:04". Like
 * formatDayLabel, the date and time are sliced as text (the value is UTC), so
 * the label never shifts with the viewer's timezone. Null/undefined (never
 * run) renders as "-"; an unparseable value passes through unchanged. */
export function formatLastRun(iso: string | null | undefined): string {
  if (!iso) return '-';
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso);
  if (!match) return iso;
  const month = MONTHS[Number(match[2]) - 1];
  if (!month) return iso;
  return `${month} ${match[3]} ${match[4]}:${match[5]}`;
}
