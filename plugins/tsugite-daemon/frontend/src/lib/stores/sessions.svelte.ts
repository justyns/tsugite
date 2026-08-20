/**
 * Sessions store: the chat sidebar's backing data. Lists the rich per-agent
 * session rows (GET /api/agents/{agent}/sessions - the only list that carries
 * pinned/progress/label/busy), overlays a live per-session progress rollup fed
 * by cross-session `session_event` broadcasts, and owns every lifecycle mutation
 * (all under /api/sessions/{id}/*, keyed by session id across agents).
 *
 * The own-tab / broadcast split matters here: turn-end + streaming events are
 * withheld from the broadcast, so this store models progress from the mid-turn
 * ticks it does see (turn_start, tool events, hook_status, llm_wait_progress) and
 * clears the overlay when a session_update reports the session finished. The
 * active surface's own per-chat stream (api/chat.ts) covers the rest.
 *
 * Cache: stale-while-revalidate. A cold load paints the last cached list for the
 * agent immediately (key tsugite_sessions_${agent}) before the network returns.
 * Exported as a class instance - never a reassigned $state binding.
 */
import { untrack } from 'svelte';
import { api } from '$lib/api/client';
import { readSwr, writeSwr } from '$lib/api/swr';
import { applyEventToProgress, emptyProgress, type Progress } from './progress';
import {
  orderSessions,
  patchRow,
  reorderPins as reorderPinRows,
  type SessionRowLike,
} from './sessionsOrder';

export interface SessionRow extends SessionRowLike {
  id: string;
  user_id: string;
  label: string;
  source: string;
  status: string;
  state: string;
  created_at: string;
  last_active: string | null;
  parent_id: string | null;
  prompt: string;
  model: string | null;
  error: string | null;
  result: string | null;
  title: string | null;
  is_default: boolean;
  metadata: Record<string, unknown>;
  pinned: boolean;
  pin_position: number | null;
  last_viewed_at: string | null;
  superseded_by: string | null;
  unread: boolean;
  is_primary: boolean;
  busy: boolean;
  compacting?: boolean;
  needs_attention?: boolean;
  pending_deliveries?: string[];
  waiting_on?: string[];
  progress?: Progress;
}

/** GET/PATCH /api/sessions/{id}/settings payload. */
export interface SessionSettings {
  reasoning_effort: string | null;
  model: string | null;
  agent: string | null;
}

export interface SessionListOpts {
  source?: string;
  status?: string;
  parentId?: string;
  includeSuperseded?: boolean;
  limit?: number;
  q?: string;
}

const TERMINAL_ACTIONS = new Set(['completed', 'cancelled', 'failed']);
const REVALIDATE_DEBOUNCE_MS = 250;

function pendingIds(data: Record<string, unknown>): string[] {
  return Array.isArray(data.pending_deliveries) ? (data.pending_deliveries as string[]) : [];
}

function cacheKey(agent: string): string {
  return `tsugite_sessions_${agent}`;
}

function buildQuery(opts: SessionListOpts): string {
  const params = new URLSearchParams();
  if (opts.source) params.set('source', opts.source);
  if (opts.status) params.set('status', opts.status);
  if (opts.parentId) params.set('parent_id', opts.parentId);
  if (opts.includeSuperseded) params.set('include_superseded', '1');
  if (opts.limit != null) params.set('limit', String(opts.limit));
  if (opts.q) params.set('q', opts.q);
  const qs = params.toString();
  return qs ? `?${qs}` : '';
}

export class SessionsStore {
  rows = $state<SessionRow[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);
  agent = $state('');
  // Live progress overlay keyed by session id, folded from session_event ticks.
  // Read through progressFor() so a row's server-sent progress is the fallback.
  progress = $state<Record<string, Progress>>({});
  // Per-session revision counter bumped on a `settings` broadcast, so the header
  // model/effort pickers refetch and their chips update live in every tab.
  settingsRev = $state<Record<string, number>>({});

  private lastOpts: SessionListOpts = {};
  private revalidateTimer: ReturnType<typeof setTimeout> | null = null;
  private burst = new Map<string, Promise<void>>();
  private convSinks = new Map<string, Set<(data: Record<string, unknown>) => void>>();

  get ordered(): SessionRow[] {
    return orderSessions(this.rows);
  }
  get pinned(): SessionRow[] {
    return this.ordered.filter((r) => r.pinned);
  }
  get unpinned(): SessionRow[] {
    return this.ordered.filter((r) => !r.pinned);
  }

  progressFor(id: string): Progress | null {
    return this.progress[id] ?? this.rows.find((r) => r.id === id)?.progress ?? null;
  }

  /** Load (or reload) the session list for an agent. Paints the SWR cache first
   *  on a cold load, then revalidates from the network and re-caches. */
  async load(agent: string, opts: SessionListOpts = {}): Promise<void> {
    // Superseded (compacted-away) sessions stay in the store so a compaction
    // banner's "view source" can resolve them; the rail filters them from display.
    opts = { includeSuperseded: true, ...opts };
    const key = agent + buildQuery(opts);
    const started = this.burst.get(key);
    if (started) return started;
    const run = this.fetchList(agent, opts);
    this.burst.set(key, run);
    queueMicrotask(() => this.burst.delete(key));
    return run;
  }

  private async fetchList(agent: string, opts: SessionListOpts): Promise<void> {
    untrack(() => {
      const prevAgent = this.agent;
      this.agent = agent;
      this.lastOpts = opts;
      // Only hydrate the cache when switching to an agent we have nothing for -
      // an in-place reload keeps the current rows painted while the fetch runs.
      if (this.rows.length === 0 || prevAgent !== agent) {
        const cached = readSwr<SessionRow[]>(cacheKey(agent));
        if (cached) this.rows = orderSessions(cached);
      }
      this.loading = true;
      this.error = null;
    });
    try {
      const res = await api.get<{ sessions: SessionRow[] }>(
        `/api/agents/${encodeURIComponent(agent)}/sessions${buildQuery(opts)}`,
      );
      this.rows = orderSessions(res.sessions);
      writeSwr(cacheKey(agent), res.sessions);
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }

  /** Server-merge search: the sidebar holds ~100 recent rows, but a query scans
   *  the full store, so this reloads with ?q= rather than filtering locally. */
  async search(q: string): Promise<void> {
    if (!this.agent) return;
    await this.load(this.agent, { ...this.lastOpts, q });
  }

  // --- SSE broadcast application ---

  /** Fold a `session_event` broadcast ({session_id, event_type, ...}) into the
   *  live progress overlay for that session, then forward it to any open
   *  conversation surface bound to that session so its timeline grows live. */
  applySessionEvent(data: Record<string, unknown>): void {
    const id = data.session_id as string | undefined;
    if (!id) return;
    const event = { type: data.event_type as string | undefined, ...data };
    const prev =
      this.progress[id] ?? this.rows.find((r) => r.id === id)?.progress ?? emptyProgress();
    this.progress = { ...this.progress, [id]: applyEventToProgress(prev, event) };
    const sinks = this.convSinks.get(id);
    if (sinks) for (const sink of sinks) sink(data);
  }

  /** Subscribe an open conversation surface to its session's live broadcast
   *  frames (the cross-session `session_event` feed). Returns an unbind fn. */
  bindConversation(id: string, sink: (data: Record<string, unknown>) => void): () => void {
    let set = this.convSinks.get(id);
    if (!set) {
      set = new Set();
      this.convSinks.set(id, set);
    }
    set.add(sink);
    return () => {
      const current = this.convSinks.get(id);
      if (!current) return;
      current.delete(sink);
      if (current.size === 0) this.convSinks.delete(id);
    };
  }

  /** Apply a `compaction_started` / `compaction_finished` broadcast to the
   *  session's authoritative compacting flag, so the pill tracks the live
   *  transitions between server list loads. */
  applyCompaction(data: Record<string, unknown>, compacting: boolean): void {
    const id = data.session_id as string | undefined;
    if (!id) return;
    this.rows = patchRow(this.rows, id, { compacting } as Partial<SessionRow>);
  }

  /** Apply a `session_update` broadcast. Fast-path the field-level actions;
   *  anything that changes the set (create/complete/primary/...) triggers a
   *  debounced revalidate so adds/removes and primary shifts stay correct. */
  applySessionUpdate(data: Record<string, unknown>): void {
    const action = data.action as string | undefined;
    const id = data.id as string | undefined;
    switch (action) {
      case 'busy':
        if (id)
          this.rows = patchRow(this.rows, id, { busy: Boolean(data.busy) } as Partial<SessionRow>);
        return;
      case 'titled':
        if (id)
          this.rows = patchRow(this.rows, id, {
            title: (data.title as string) ?? null,
          } as Partial<SessionRow>);
        return;
      case 'viewed':
        if (id) this.rows = patchRow(this.rows, id, { unread: false } as Partial<SessionRow>);
        return;
      case 'delivered':
        if (id)
          this.rows = patchRow(this.rows, id, {
            needs_attention: data.needs_attention === true,
            pending_deliveries: pendingIds(data),
            unread: true,
          } as Partial<SessionRow>);
        return;
      case 'attention_cleared':
        if (id)
          this.rows = patchRow(this.rows, id, {
            needs_attention: data.needs_attention === true,
            pending_deliveries: pendingIds(data),
          } as Partial<SessionRow>);
        return;
      case 'reordered':
        if (Array.isArray(data.ids)) this.rows = reorderPinRows(this.rows, data.ids as string[]);
        return;
      case 'settings':
        // Model / reasoning-effort changed (settings PATCH, /model, /effort).
        // Patch the row's model and bump the rev so the header pickers refetch
        // and their chips update live in every tab.
        if (id) {
          if ('model' in data)
            this.rows = patchRow(this.rows, id, {
              model: (data.model as string) ?? null,
            } as Partial<SessionRow>);
          this.settingsRev = { ...this.settingsRev, [id]: (this.settingsRev[id] ?? 0) + 1 };
        }
        return;
      default:
        if (action && TERMINAL_ACTIONS.has(action) && id) {
          this.dropProgress(id);
          this.rows = patchRow(this.rows, id, {
            status: action,
            state: action,
          } as Partial<SessionRow>);
        }
        this.scheduleRevalidate();
    }
  }

  private dropProgress(id: string): void {
    if (!(id in this.progress)) return;
    const next = { ...this.progress };
    delete next[id];
    this.progress = next;
  }

  private scheduleRevalidate(): void {
    if (!this.agent) return;
    if (this.revalidateTimer !== null) clearTimeout(this.revalidateTimer);
    this.revalidateTimer = setTimeout(() => {
      this.revalidateTimer = null;
      void this.load(this.agent, this.lastOpts);
    }, REVALIDATE_DEBOUNCE_MS);
  }

  // --- lifecycle mutations (keyed by session id, cross-agent) ---

  async newSession(agent: string, opts: { title?: string; userId?: string } = {}): Promise<string> {
    const body: Record<string, unknown> = {};
    if (opts.title != null) body.title = opts.title;
    if (opts.userId != null) body.user_id = opts.userId;
    const res = await api.post<{ id: string }>(
      `/api/agents/${encodeURIComponent(agent)}/sessions/new`,
      body,
    );
    return res.id;
  }

  async rename(id: string, title: string): Promise<void> {
    await api.patch(`/api/sessions/${encodeURIComponent(id)}`, { title });
    this.rows = patchRow(this.rows, id, { title } as Partial<SessionRow>);
  }

  async setTopic(id: string, topic: string): Promise<void> {
    await api.patch(`/api/sessions/${encodeURIComponent(id)}/metadata`, { topic });
    const row = this.rows.find((r) => r.id === id);
    if (row) {
      this.rows = patchRow(this.rows, id, {
        metadata: { ...row.metadata, topic },
      } as Partial<SessionRow>);
    }
  }

  async complete(id: string): Promise<void> {
    await api.patch(`/api/sessions/${encodeURIComponent(id)}`, { status: 'completed' });
  }

  async cancel(id: string): Promise<void> {
    await api.post(`/api/sessions/${encodeURIComponent(id)}/cancel`);
  }

  async restart(id: string): Promise<void> {
    await api.post(`/api/sessions/${encodeURIComponent(id)}/restart`);
  }

  async pin(id: string, position?: number): Promise<void> {
    await api.post(
      `/api/sessions/${encodeURIComponent(id)}/pin`,
      position != null ? { position } : undefined,
    );
    this.rows = patchRow(this.rows, id, { pinned: true } as Partial<SessionRow>);
  }

  async unpin(id: string): Promise<void> {
    await api.post(`/api/sessions/${encodeURIComponent(id)}/unpin`);
    this.rows = patchRow(this.rows, id, {
      pinned: false,
      pin_position: null,
    } as Partial<SessionRow>);
  }

  async reorderPins(ids: string[]): Promise<void> {
    // Echo the new order locally so the drag settles instantly; the broadcast
    // reordered event confirms it.
    this.rows = reorderPinRows(this.rows, ids);
    await api.post('/api/sessions/pinned/reorder', { ids });
  }

  async setPrimary(id: string): Promise<void> {
    await api.post(`/api/sessions/${encodeURIComponent(id)}/set-primary`);
  }

  async clearPrimary(agent: string, userId: string): Promise<void> {
    const params = new URLSearchParams({ agent, user_id: userId });
    await api.post(`/api/sessions/clear-primary?${params.toString()}`);
  }

  /** Read a session's model / reasoning_effort / agent (the picker + effort seg). */
  async getSettings(id: string): Promise<SessionSettings> {
    return api.get<SessionSettings>(`/api/sessions/${encodeURIComponent(id)}/settings`);
  }

  /** Resolve a session's identity from its record: the agent it actually runs
   *  under and its metadata. Agent-agnostic (GET /api/sessions/{id}), so it works
   *  even when that agent isn't a configured chat adapter - the chat surface uses
   *  it to heal a stale/wrong deep-link `agent` param and to flag job artifacts. */
  async getInfo(id: string): Promise<{
    agent: string | null;
    metadata: Record<string, unknown>;
    contextLimit: number | null;
    cumulativeTokens: number | null;
  }> {
    const res = await api.get<{
      agent?: string | null;
      metadata?: Record<string, unknown>;
      context_limit?: number | null;
      context_limit_resolved?: number | null;
      cumulative_tokens?: number | null;
    }>(`/api/sessions/${encodeURIComponent(id)}`);
    return {
      agent: res.agent ?? null,
      metadata: res.metadata ?? {},
      // Durable context truth for the meter: session_info frames are live-only,
      // so a freshly loaded conversation falls back to the session record. Prefer
      // the RESOLVED limit (agent default until the first turn reports a provider
      // window) - the raw `context_limit` is null on a fresh session, which would
      // otherwise leave a brand-new chat with no meter at all.
      contextLimit: res.context_limit_resolved ?? res.context_limit ?? null,
      cumulativeTokens: res.cumulative_tokens ?? null,
    };
  }

  /** Persist a model or reasoning_effort change for a session; returns the merged
   *  settings the server echoes back. */
  async patchSettings(
    id: string,
    patch: Partial<Pick<SessionSettings, 'model' | 'reasoning_effort'>>,
  ): Promise<SessionSettings> {
    return api.patch<SessionSettings>(`/api/sessions/${encodeURIComponent(id)}/settings`, patch);
  }

  async dismissAttention(id: string, deliveryId?: string): Promise<void> {
    await api.post(
      `/api/sessions/${encodeURIComponent(id)}/dismiss-attention`,
      deliveryId ? { delivery_id: deliveryId } : undefined,
    );
    const row = this.rows.find((r) => r.id === id);
    const left = deliveryId ? (row?.pending_deliveries ?? []).filter((d) => d !== deliveryId) : [];
    this.rows = patchRow(this.rows, id, {
      needs_attention: left.length > 0,
      pending_deliveries: left,
    } as Partial<SessionRow>);
  }

  async markViewed(id: string, ts?: string): Promise<void> {
    await api.post(`/api/sessions/${encodeURIComponent(id)}/mark-viewed`, ts ? { ts } : undefined);
    this.rows = patchRow(this.rows, id, { unread: false } as Partial<SessionRow>);
  }
}

export const sessions = new SessionsStore();
