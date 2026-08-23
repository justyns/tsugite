/**
 * Activity store: the cross-cutting recent-activity feed (GET /api/activity),
 * newest first. The global broadcasts don't carry a feed row, so a live signal
 * bumps `rev` (debounced) and a mounted feed refetches its window off that.
 * Exported as a class instance - never a reassigned $state binding.
 */
import { api } from '$lib/api/client';
import type { SSEEvent } from '$lib/api/sse';

export type ActivityType = 'session' | 'job' | 'schedule' | 'compaction';
export type ActivityStatus = 'ok' | 'error' | 'cancelled' | 'skipped';

export interface ActivityEntry {
  id: string;
  type: ActivityType;
  timestamp: string;
  title: string;
  summary: string;
  /** Tone the row is tinted with, or null for none. */
  status: ActivityStatus | null;
  /** The word the status pill prints. */
  label: string;
  session_id: string | null;
  job_id: string | null;
  schedule_id: string | null;
}

const SIGNAL_DEBOUNCE_MS = 400;
// A session_update only names the transition; these are the ones that end a run
// and therefore add a feed row.
const END_ACTIONS = new Set(['completed', 'failed', 'cancelled']);
// Other broadcasts that mean the feed gained a row. history_update covers
// interactive chats: a settled turn is the only signal one gives that it
// finished a run.
const FEED_EVENTS = new Set([
  'compaction_finished',
  'job_update',
  'schedule_update',
  'history_update',
]);

export class ActivityStore {
  entries = $state<ActivityEntry[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);
  /** Bumped when a broadcast means the feed has something new; a mounted view
   *  reads it to refetch. Nothing fetches while no view is watching. */
  rev = $state(0);

  private timer: ReturnType<typeof setTimeout> | null = null;
  private loadSeq = 0;

  async load(opts: { types?: string } = {}): Promise<void> {
    // The view refetches on every filter change and live signal, so two loads
    // can be in flight; the newest one wins.
    const seq = ++this.loadSeq;
    this.loading = true;
    this.error = null;
    try {
      const params = new URLSearchParams();
      if (opts.types) params.set('types', opts.types);
      const qs = params.toString();
      const res = await api.get<{ entries: ActivityEntry[] }>(`/api/activity${qs ? `?${qs}` : ''}`);
      if (seq === this.loadSeq) this.entries = res.entries;
    } catch (err) {
      if (seq === this.loadSeq) this.error = err instanceof Error ? err.message : String(err);
    } finally {
      if (seq === this.loadSeq) this.loading = false;
    }
  }

  /** Route one broadcast frame: bump `rev` when it means the feed changed. */
  applyEvent(event: SSEEvent): void {
    if (event.type === 'session_update') {
      if (END_ACTIONS.has(String(event.data?.action))) this.signal();
      return;
    }
    if (FEED_EVENTS.has(event.type)) this.signal();
  }

  /** Debounced: one settling turn fires several broadcasts. */
  private signal(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.rev++;
    }, SIGNAL_DEBOUNCE_MS);
  }
}

export const activity = new ActivityStore();
