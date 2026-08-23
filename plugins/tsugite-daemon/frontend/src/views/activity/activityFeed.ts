/**
 * Pure presentation logic for the activity feed: the filter set, the day
 * grouping, the per-row glyph, and where a row links to. Kept out of the
 * .svelte file so every rule is directly unit-testable.
 */
import type { IconName } from '$lib/components/icon/icons';
import type { Route } from '$lib/router.svelte';
import type { ActivityEntry, ActivityType } from '$lib/stores/activity.svelte';

export const ACTIVITY_FILTERS = [
  { key: 'all', label: 'all' },
  { key: 'session', label: 'sessions' },
  { key: 'job', label: 'jobs' },
  { key: 'schedule', label: 'schedules' },
  { key: 'compaction', label: 'compactions' },
] as const;

export type ActivityFilter = (typeof ACTIVITY_FILTERS)[number]['key'];

export interface DayGroup {
  /** The day bucket (`YYYY-MM-DD` or `undated`). */
  day: string;
  label: string;
  entries: ActivityEntry[];
}

export const ENTRY_ICON: Record<ActivityType, IconName> = {
  session: 'chat',
  job: 'jobs',
  schedule: 'sched',
  compaction: 'compress',
};

/** Local calendar day of an instant, as a stable bucket key. */
function dayKey(date: Date): string {
  const month = String(date.getMonth() + 1).padStart(2, '0');
  return `${date.getFullYear()}-${month}-${String(date.getDate()).padStart(2, '0')}`;
}

function dayLabel(date: Date | null, key: string, today: string, yesterday: string): string {
  if (!date) return 'undated';
  if (key === today) return 'today';
  if (key === yesterday) return 'yesterday';
  return new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
    .format(date)
    .replace(',', '')
    .toLowerCase();
}

/** Group a newest-first feed into day buckets, preserving the incoming order. */
export function groupByDay(entries: ActivityEntry[], now: number): DayGroup[] {
  const today = dayKey(new Date(now));
  // Calendar arithmetic, not now - 24h: on a DST day the local day is not 24 hours
  // long, and the wall-clock version mislabels the headings around the transition.
  const y = new Date(now);
  y.setDate(y.getDate() - 1);
  const yesterday = dayKey(y);
  const groups: DayGroup[] = [];
  for (const entry of entries) {
    const t = Date.parse(entry.timestamp);
    const date = Number.isNaN(t) ? null : new Date(t);
    const key = date ? dayKey(date) : 'undated';
    let group = groups[groups.length - 1];
    if (!group || group.day !== key) {
      group = { day: key, label: dayLabel(date, key, today, yesterday), entries: [] };
      groups.push(group);
    }
    group.entries.push(entry);
  }
  return groups;
}

/** Where clicking a row goes, or null when the row links nowhere. A job opens
 *  the board (the caller also asks it for that job's drawer); everything else
 *  prefers the conversation it produced. */
export function entryRoute(entry: ActivityEntry): Route | null {
  if (entry.job_id) return { view: 'jobs', params: {} };
  if (entry.session_id) return { view: 'chats', params: { sessionId: entry.session_id } };
  if (entry.schedule_id) return { view: 'schedules', params: {} };
  return null;
}
