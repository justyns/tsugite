/**
 * Pure presentation logic for the schedules table (sorting, status derivation,
 * summary counts, sparkline mapping, and the relative/absolute time labels).
 * Kept out of the .svelte file so every rule is directly unit-testable and the
 * component stays a thin render layer.
 */
import type { Schedule } from '$lib/stores/schedules.svelte';

/** Row status the pill renders. Derived from `enabled` + `last_status`; there is
 *  no live "running" flag on a schedule entry, so that state is not synthesized. */
export type RunStatus = 'off' | 'errored' | 'done' | 'skipped' | 'queued';

export type SortDir = 'ascending' | 'descending';

/** One `run_history` entry as persisted by the scheduler. */
export interface RunHistoryItem {
  timestamp?: string;
  status?: string | null;
  error?: string | null;
  session_id?: string | null;
}

export type SparkStatus = 'ok' | 'fail' | 'skip';
export interface SparkPoint {
  height: number;
  status?: SparkStatus;
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
/** Below this horizon a future run reads as a relative countdown; beyond it, an
 *  absolute weekday/time - the "in 12h 31m" vs "mon 09:00" cut. */
const RELATIVE_HORIZON = 13 * HOUR;
const WEEK = 7 * DAY;

export function deriveRunStatus(s: Pick<Schedule, 'enabled' | 'last_status'>): RunStatus {
  if (!s.enabled) return 'off';
  switch (s.last_status) {
    case 'success':
      return 'done';
    case 'error':
      return 'errored';
    case 'skipped':
      return 'skipped';
    default:
      return 'queued';
  }
}

/** Sort key: the next-run instant in ms, or +Infinity when a schedule won't run
 *  next (disabled / no next_run / expired), so those always land at the far end. */
function nextKey(s: Schedule): number {
  if (!s.enabled || !s.next_run) return Infinity;
  const t = Date.parse(s.next_run);
  return Number.isNaN(t) ? Infinity : t;
}

/** Stable sort by next run. Ascending puts the soonest first and null-next last;
 *  descending flips both (mirrors the aria-sort toggle on the header). */
export function sortSchedules(list: Schedule[], dir: SortDir): Schedule[] {
  const sign = dir === 'ascending' ? 1 : -1;
  return list
    .map((s, i) => [s, i] as const)
    .sort((a, b) => {
      const ka = nextKey(a[0]);
      const kb = nextKey(b[0]);
      if (ka === kb) return a[1] - b[1];
      return (ka < kb ? -1 : 1) * sign;
    })
    .map(([s]) => s);
}

export interface ScheduleSummary {
  total: number;
  failing: number;
  disabled: number;
}

export function summarize(list: Schedule[]): ScheduleSummary {
  let failing = 0;
  let disabled = 0;
  for (const s of list) {
    if (!s.enabled) disabled++;
    else if (s.last_status === 'error') failing++;
  }
  return { total: list.length, failing, disabled };
}

/** The enabled schedule with the soonest future next run, or null. */
export function nextUp(list: Schedule[], now: number): { schedule: Schedule; at: number } | null {
  let best: { schedule: Schedule; at: number } | null = null;
  for (const s of list) {
    if (!s.enabled || !s.next_run) continue;
    const t = Date.parse(s.next_run);
    if (Number.isNaN(t) || t < now) continue;
    if (!best || t < best.at) best = { schedule: s, at: t };
  }
  return best;
}

// `run_history` isn't a declared field on Schedule (it arrives via the entry's
// index signature, typed `unknown`), so callers pass the raw value.
function historyItems(runHistory: unknown): RunHistoryItem[] {
  return Array.isArray(runHistory) ? (runHistory as RunHistoryItem[]) : [];
}

const SPARK_STATUS: Record<string, SparkStatus> = { success: 'ok', error: 'fail', skipped: 'skip' };
// Deterministic, decorative heights (px). Schedule runs carry no magnitude, so
// this only gives the bar row visual texture - never implies a measured value.
const SPARK_HEIGHTS = [6, 8, 7, 9, 7, 8, 6, 9, 8, 7];

export interface SparkModel {
  points: SparkPoint[];
  label: string;
}

/** Build the last-N-runs sparkline from `run_history` (oldest→newest, capped 10). */
export function buildSpark(runHistory: unknown, limit = 10): SparkModel {
  const items = historyItems(runHistory).slice(-limit);
  if (items.length === 0) return { points: [], label: 'no recent runs' };

  let fails = 0;
  let skips = 0;
  const points: SparkPoint[] = items.map((it, i) => {
    const status = SPARK_STATUS[it.status ?? ''] ?? 'skip';
    if (status === 'fail') fails++;
    else if (status === 'skip') skips++;
    return { height: SPARK_HEIGHTS[i % SPARK_HEIGHTS.length] ?? 8, status };
  });

  const n = items.length;
  let detail = 'all ok';
  if (fails && skips) detail = `${fails} failed, ${skips} skipped`;
  else if (fails) detail = `${fails} failed`;
  else if (skips) detail = `${skips} skipped`;
  return { points, label: `last ${n} run${n === 1 ? '' : 's'}: ${detail}` };
}

/** Order run sessions newest-first (by created_at) and cap the list. Unparsable
 *  stamps sort last; ties keep their input order. */
export function recentRuns<T extends { created_at: string | null }>(runs: T[], limit = 10): T[] {
  const key = (r: T): number => {
    const t = r.created_at ? Date.parse(r.created_at) : NaN;
    return Number.isNaN(t) ? -Infinity : t;
  };
  return runs
    .map((r, i) => [r, i] as const)
    .sort((a, b) => key(b[0]) - key(a[0]) || a[1] - b[1])
    .slice(0, limit)
    .map(([r]) => r);
}

function fmt(iso: string, tz: string, opts: Intl.DateTimeFormatOptions): string {
  return new Intl.DateTimeFormat('en-US', { ...opts, timeZone: tz })
    .format(new Date(iso))
    .toLowerCase();
}

/**
 * Label for the next run: a relative countdown for near-future runs, an absolute
 * weekday/time within a week, and a short date beyond that. Timezone-correct so a
 * "daily 03:00" schedule reads 03:00 in its own zone. Returns "—" when there is
 * no next run. `tz` falls back to UTC on an unknown zone (Intl would otherwise throw).
 */
export function formatNextRun(iso: string | null | undefined, tz: string, now: number): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '—';
  const delta = t - now;
  if (delta <= 0) return 'due';

  const mins = Math.floor(delta / MINUTE);
  if (mins < 1) return 'in <1m';
  if (mins < 60) return `in ${mins}m`;
  if (delta < RELATIVE_HORIZON) {
    const h = Math.floor(delta / HOUR);
    const m = Math.floor((delta % HOUR) / MINUTE);
    return m === 0 ? `in ${h}h` : `in ${h}h ${m}m`;
  }
  const zone = safeZone(tz);
  const time = fmt(iso, zone, { hour: '2-digit', minute: '2-digit', hour12: false });
  if (delta < WEEK) return `${fmt(iso, zone, { weekday: 'short' })} ${time}`;
  return fmt(iso, zone, { month: 'short', day: 'numeric' });
}

/** Absolute short stamp ("jul 11 03:00") in local time, for a run/one-off instant. */
export function formatStamp(iso: string | null | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
    .format(d)
    .replace(',', '')
    .toLowerCase();
}

/** Compact elapsed label ("4s", "9m 12s", "1h 03m") for a run duration in ms. */
export function formatDuration(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  if (m < 60) return `${m}m ${String(s).padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, '0')}m`;
}

function safeZone(tz: string): string {
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: tz });
    return tz;
  } catch {
    return 'UTC';
  }
}
