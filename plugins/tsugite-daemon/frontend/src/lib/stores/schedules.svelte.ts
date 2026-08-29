/**
 * Schedules store: cron / one-off task entries (GET /api/schedules), full CRUD
 * plus enable/disable/run, and the per-schedule session history. `schedule_update`
 * broadcasts ({action, id}) only name what changed, not the new row, so they
 * trigger a debounced reload rather than a field patch. Exported as a class
 * instance.
 */
import { api } from '$lib/api/client';

export interface Schedule {
  id: string;
  agent: string;
  prompt: string;
  schedule_type: 'cron' | 'once';
  cron_expr: string | null;
  run_at: string | null;
  enabled: boolean;
  created_at: string;
  last_run: string | null;
  next_run: string | null;
  last_status: string | null;
  last_error: string | null;
  timezone: string;
  execution_type: 'agent' | 'script' | 'session_message';
  command: string | null;
  run_count: number;
  disabled_reason: string | null;
  [key: string]: unknown;
}

export interface ScheduleSession {
  id: string;
  status: string;
  created_at: string;
  last_active: string | null;
  result: string;
  error: string | null;
}

const RELOAD_DEBOUNCE_MS = 250;

export class SchedulesStore {
  list = $state<Schedule[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);
  private reloadTimer: ReturnType<typeof setTimeout> | null = null;

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await api.get<{ schedules: Schedule[] }>('/api/schedules/');
      this.list = res.schedules;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }

  async create(
    body: Partial<Schedule> & { id: string; agent: string; prompt: string; schedule_type: string },
  ): Promise<Schedule> {
    return api.post<Schedule>('/api/schedules/', body);
  }

  async get(id: string): Promise<Schedule> {
    return api.get<Schedule>(`/api/schedules/${encodeURIComponent(id)}`);
  }

  async update(id: string, fields: Partial<Schedule>): Promise<Schedule> {
    return api.patch<Schedule>(`/api/schedules/${encodeURIComponent(id)}`, fields);
  }

  async remove(id: string): Promise<void> {
    await api.del(`/api/schedules/${encodeURIComponent(id)}`);
  }

  async enable(id: string): Promise<void> {
    await api.post(`/api/schedules/${encodeURIComponent(id)}/enable`);
  }

  async disable(id: string): Promise<void> {
    await api.post(`/api/schedules/${encodeURIComponent(id)}/disable`);
  }

  async run(id: string): Promise<void> {
    await api.post(`/api/schedules/${encodeURIComponent(id)}/run`);
  }

  /** Remove auto-disabled entries (expired / max-runs / missed one-offs). Returns
   *  the ids the daemon dropped. */
  async cleanup(): Promise<string[]> {
    const res = await api.post<{ removed: string[]; count: number }>('/api/schedules/cleanup');
    return res.removed;
  }

  async sessions(id: string): Promise<ScheduleSession[]> {
    const res = await api.get<{ sessions: ScheduleSession[] }>(
      `/api/schedules/${encodeURIComponent(id)}/sessions`,
    );
    return res.sessions;
  }

  /** A schedule_update broadcast changed the set; reload (debounced). */
  applyScheduleUpdate(_data: Record<string, unknown>): void {
    if (this.reloadTimer !== null) clearTimeout(this.reloadTimer);
    this.reloadTimer = setTimeout(() => {
      this.reloadTimer = null;
      void this.load();
    }, RELOAD_DEBOUNCE_MS);
  }
}

export const schedules = new SchedulesStore();
