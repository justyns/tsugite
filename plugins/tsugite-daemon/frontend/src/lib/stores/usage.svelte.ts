/**
 * Usage store: the cost dashboard's four series (summary trend, top agents, top
 * models, grand total) fetched together on load / range change. All read-only
 * GETs under /api/usage. The range is "last N days": `since` is an ISO date N
 * days back (the store only supports a start bound - the backend accepts an
 * `until` end but no route wires it yet). Exported as a class instance.
 */
import { api } from '$lib/api/client';

export type UsagePeriod = 'day' | 'week' | 'month';

/** Cache-split columns shared by every usage aggregation. Honest SUMs of the
 *  stored per-turn values; a provider that reports no cache usage contributes
 *  0, which the backend can't tell apart from a genuine zero. */
export interface UsageCacheSplit {
  cache_creation_tokens: number;
  cache_read_tokens: number;
}

export interface UsageSummaryRow extends UsageCacheSplit {
  period: string;
  runs: number;
  total_tokens: number;
  total_cost: number;
  input_tokens: number;
  output_tokens: number;
  total_duration_ms: number;
}

export interface UsageAgentRow extends UsageCacheSplit {
  agent: string;
  runs: number;
  total_tokens: number;
  total_cost: number;
}

export interface UsageModelRow extends UsageCacheSplit {
  model: string;
  runs: number;
  total_tokens: number;
  total_cost: number;
}

export interface UsageScheduleRow extends UsageCacheSplit {
  /** null for the unattributed bucket: scheduled runs recorded before the
   *  schedule_id marker existed. */
  schedule_name: string | null;
  runs: number;
  total_tokens: number;
  total_cost: number;
  last_run: string | null;
}

export interface UsageTotal extends UsageCacheSplit {
  runs: number;
  total_tokens: number;
  total_cost: number;
  input_tokens: number;
  output_tokens: number;
}

export interface UsageRange {
  sinceDays: number;
  period: UsagePeriod;
  agent?: string;
}

/** ISO date (YYYY-MM-DD) `days` before `from`. The backend compares `since`
 *  lexicographically against ISO timestamps, so a date-only prefix is valid. */
export function daysAgoISO(days: number, from: Date = new Date()): string {
  const d = new Date(from);
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString().slice(0, 10);
}

export class UsageStore {
  summary = $state<UsageSummaryRow[]>([]);
  agents = $state<UsageAgentRow[]>([]);
  models = $state<UsageModelRow[]>([]);
  schedules = $state<UsageScheduleRow[]>([]);
  total = $state<UsageTotal | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);
  range = $state<UsageRange>({ sinceDays: 30, period: 'day' });

  /** A real "since UTC midnight today" total for the always-visible keystrip
   *  widget - deliberately separate from `total`/`range` above, which are
   *  scoped to the dashboard's own range picker (7/30/90 days). Coupling the
   *  two would make the keystrip's "today" figure jump around as a user
   *  browses the Usage view, and would leave it unpopulated until they visited
   *  that view at least once. Best-effort: a failed fetch just leaves the
   *  keystrip on its previous (or placeholder) value, since there's no error
   *  slot for this ambient widget. */
  today = $state<UsageTotal | null>(null);

  async loadToday(): Promise<void> {
    try {
      this.today = await api.get<UsageTotal>(`/api/usage/total?since=${daysAgoISO(0)}`);
    } catch {
      // best-effort - see field doc above
    }
  }

  async load(range: Partial<UsageRange> = {}): Promise<void> {
    this.range = { ...this.range, ...range };
    const since = daysAgoISO(this.range.sinceDays);
    const summaryParams = new URLSearchParams({ period: this.range.period, since });
    if (this.range.agent) summaryParams.set('agent', this.range.agent);
    const sinceQs = new URLSearchParams({ since }).toString();

    this.loading = true;
    this.error = null;
    try {
      const [summary, agents, models, schedules, total] = await Promise.all([
        api.get<UsageSummaryRow[]>(`/api/usage/summary?${summaryParams.toString()}`),
        api.get<UsageAgentRow[]>(`/api/usage/agents?${sinceQs}`),
        api.get<UsageModelRow[]>(`/api/usage/models?${sinceQs}`),
        api.get<UsageScheduleRow[]>(`/api/usage/schedules?${sinceQs}`),
        api.get<UsageTotal>(`/api/usage/total?${sinceQs}`),
      ]);
      this.summary = summary;
      this.agents = agents;
      this.models = models;
      this.schedules = schedules;
      this.total = total;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }
}

export const usage = new UsageStore();
