/**
 * Terminals store: PTY list/create/kill/restart/stdin plus per-terminal SSE
 * stream handles. The stream is bearer-authed, and EventSource can't set
 * headers, so it's hand-rolled over fetch + the named-frame parser (parseNamedSSE).
 *
 * Two orthogonal axes, deliberately kept separate: the server
 * `state` (the 6-value TerminalState machine) and a purely client-side `follow`
 * boolean per open terminal. A terminal can be state:'running' with follow:false
 * at the same time - "paused-follow" is not a server state, so it is not folded
 * into the state map. Exported as a class instance.
 */
import { api, authHeaders } from '$lib/api/client';
import { auth } from '$lib/stores/auth.svelte';
import { parseNamedSSE } from '$lib/api/sse';

// The 6 real backend states (terminal_store.TerminalState). No 'paused-follow'.
export type TerminalState =
  'starting' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'stream_lost';

const LIVE_STATES = new Set<TerminalState>(['starting', 'running']);

export interface Terminal {
  id: string;
  cmd: string;
  cwd: string | null;
  state: TerminalState;
  pid: number | null;
  exit_code: number | null;
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
  bytes_out: number;
  lines_out: number;
  last_line: string;
  parent_session_id: string | null;
  truncated: boolean;
  /** Only present on the restart response; never persisted server-side. */
  restarted_from?: string;
}

export interface TerminalStreamHandlers {
  onState?: (state: TerminalState) => void;
  /** replay=true marks the single backlog frame sent right after connecting. */
  onOutput?: (chunk: string, replay: boolean) => void;
  onExit?: (exitCode: number | null) => void;
  onStatus?: (connected: boolean) => void;
}

export interface TerminalStreamHandle {
  close(): void;
}

export interface CreateTerminalOpts {
  cmd: string;
  cwd?: string;
  parentSessionId?: string;
  env?: Record<string, string>;
}

function countLines(chunk: string): number {
  let n = 0;
  for (const ch of chunk) if (ch === '\n') n += 1;
  return n;
}

export class TerminalsStore {
  list = $state<Terminal[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);
  // Live state overlay (from terminal_state broadcasts + stream `state` frames),
  // read through stateOf() so a record's own state is the fallback.
  states = $state<Record<string, TerminalState>>({});
  // Client-only follow axis + queued-line counters for the "N new lines" pill.
  follow = $state<Record<string, boolean>>({});
  queuedLines = $state<Record<string, number>>({});

  stateOf(id: string): TerminalState | null {
    return this.states[id] ?? this.list.find((t) => t.id === id)?.state ?? null;
  }

  isLive(id: string): boolean {
    const state = this.stateOf(id);
    return state != null && LIVE_STATES.has(state);
  }

  async load(parentSessionId?: string): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      this.list = this.carryRestartedFrom(await this.fetchList(parentSessionId));
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }

  /** restarted_from rides only the one-shot restart() response, never a plain
   *  GET /api/terminals/, so a metrics-poll refresh would blank the chip. Carry
   *  the client-held value forward per id (a value on the fetched record wins). */
  private carryRestartedFrom(fresh: Terminal[]): Terminal[] {
    return fresh.map((t) => {
      const prior = this.list.find((p) => p.id === t.id)?.restarted_from;
      return prior && !t.restarted_from ? { ...t, restarted_from: prior } : t;
    });
  }

  private async fetchList(parentSessionId?: string): Promise<Terminal[]> {
    const qs = parentSessionId
      ? `?${new URLSearchParams({ parent_session_id: parentSessionId }).toString()}`
      : '';
    const res = await api.get<{ terminals: Terminal[] }>(`/api/terminals/${qs}`);
    // Newest first, stable across polls (the API order is store-internal); ISO
    // timestamps compare lexicographically, id breaks created_at ties.
    return res.terminals
      .slice()
      .sort(
        (a, b) =>
          (b.created_at ?? '').localeCompare(a.created_at ?? '') || b.id.localeCompare(a.id),
      );
  }

  async create(opts: CreateTerminalOpts): Promise<Terminal> {
    const body: Record<string, unknown> = { cmd: opts.cmd };
    if (opts.cwd) body.cwd = opts.cwd;
    if (opts.parentSessionId) body.parent_session_id = opts.parentSessionId;
    if (opts.env) body.env = opts.env;
    const term = await api.post<Terminal>('/api/terminals/', body);
    this.list = [term, ...this.list.filter((t) => t.id !== term.id)];
    return term;
  }

  async kill(id: string): Promise<void> {
    await api.post(`/api/terminals/${encodeURIComponent(id)}/kill`);
  }

  /** Restart spawns a NEW terminal id carrying restarted_from -> old id. */
  async restart(id: string): Promise<Terminal> {
    const term = await api.post<Terminal>(`/api/terminals/${encodeURIComponent(id)}/restart`);
    this.list = [term, ...this.list.filter((t) => t.id !== term.id)];
    return term;
  }

  async stdin(id: string, data: string): Promise<number> {
    const res = await api.post<{ status: string; bytes_written: number }>(
      `/api/terminals/${encodeURIComponent(id)}/stdin`,
      { data },
    );
    return res.bytes_written;
  }

  // --- client follow axis ---

  isFollowing(id: string): boolean {
    return this.follow[id] ?? true;
  }

  setFollow(id: string, follow: boolean): void {
    this.follow = { ...this.follow, [id]: follow };
    if (follow) this.clearQueued(id);
  }

  clearQueued(id: string): void {
    if (!this.queuedLines[id]) return;
    this.queuedLines = { ...this.queuedLines, [id]: 0 };
  }

  // --- SSE broadcast application ---

  applyTerminalState(data: Record<string, unknown>): void {
    const id = data.terminal_id as string | undefined;
    const state = data.state as TerminalState | undefined;
    if (!id || !state) return;
    this.setState(id, state);
  }

  private setState(id: string, state: TerminalState): void {
    this.states = { ...this.states, [id]: state };
    const idx = this.list.findIndex((t) => t.id === id);
    const current = idx === -1 ? undefined : this.list[idx];
    if (current && current.state !== state) {
      const next = this.list.slice();
      next[idx] = { ...current, state };
      this.list = next;
    }
  }

  private noteOutput(id: string, chunk: string): void {
    if (this.isFollowing(id)) return;
    const added = countLines(chunk);
    if (!added) return;
    this.queuedLines = { ...this.queuedLines, [id]: (this.queuedLines[id] ?? 0) + added };
  }

  /** Open a live SSE stream for one terminal. Frames arrive named (state /
   *  output / exit); a drop before exit marks the terminal stream_lost (a
   *  client-detected disconnect, matching the backend's own on-restart
   *  reconciliation). */
  stream(id: string, handlers: TerminalStreamHandlers = {}): TerminalStreamHandle {
    let closed = false;
    let exited = false;
    const controller = new AbortController();

    const run = async (): Promise<void> => {
      try {
        const resp = await fetch(`/api/terminals/${encodeURIComponent(id)}/stream`, {
          headers: authHeaders(),
          signal: controller.signal,
        });
        if (resp.status === 401) {
          auth.requireAuth();
          return;
        }
        if (!resp.ok) throw new Error(resp.statusText);
        handlers.onStatus?.(true);
        for await (const frame of parseNamedSSE(resp)) {
          const payload = (frame.data ?? {}) as Record<string, unknown>;
          if (frame.event === 'state') {
            const state = payload.state as TerminalState;
            this.setState(id, state);
            handlers.onState?.(state);
          } else if (frame.event === 'output') {
            const chunk = (payload.chunk as string) ?? '';
            this.noteOutput(id, chunk);
            handlers.onOutput?.(chunk, payload.replay === true);
          } else if (frame.event === 'exit') {
            exited = true;
            handlers.onExit?.((payload.exit_code as number | null) ?? null);
          }
        }
      } catch {
        // Aborted (close) or a transport drop; handled below.
      }
      handlers.onStatus?.(false);
      // A stream that ended without an exit frame (and that we didn't close) is a
      // lost connection, not a clean terminate.
      if (!closed && !exited && this.isLive(id)) this.setState(id, 'stream_lost');
    };

    void run();
    return {
      close() {
        closed = true;
        controller.abort();
      },
    };
  }
}

export const terminals = new TerminalsStore();
