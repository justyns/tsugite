/**
 * Pure per-type router for the cross-session SSE broadcast (GET /api/events).
 * Every global frame is {seq, type, data}; this maps `type` to a sink callback
 * and hands it the `data` payload. Keeping it a pure function (a sink of
 * closures, not store singletons) makes the fan-out node-testable and keeps
 * App.svelte's dispatch seam a one-liner.
 *
 * Note the own-tab / broadcast split: turn-end and
 * streaming events (stream_chunk/stream_complete/prompt_snapshot/final_result/
 * error/cancelled) are withheld from this broadcast on purpose - they arrive
 * only on the per-chat POST stream owned by the surface that ran the send. This
 * router therefore never sees them; the chat helper (api/chat.ts) owns that half.
 */
import type { SSEEvent } from '$lib/api/sse';

export interface ShellEventSink {
  /** Daemon restart / unreplayable gap - the shell hard-reloads. */
  onReconnect?: () => void;
  /** session_event: a mid-session progress tick for the sidebar rollup. */
  onSessionEvent?: (data: Record<string, unknown>) => void;
  /** session_update: a lifecycle transition (busy/titled/completed/...). */
  onSessionUpdate?: (data: Record<string, unknown>) => void;
  /** job_update: a full Job payload for the jobs board. */
  onJobUpdate?: (data: Record<string, unknown>) => void;
  /** schedule_update: {action, id} for the schedules list. */
  onScheduleUpdate?: (data: Record<string, unknown>) => void;
  /** terminal_state: {terminal_id, state} liveness for the terminals list. */
  onTerminalState?: (data: Record<string, unknown>) => void;
  /** agent_status: {agent} - config/running-count changed. */
  onAgentStatus?: (data: Record<string, unknown>) => void;
  /** history_update: {agent, session_id} - a turn settled; refetch history. */
  onHistoryUpdate?: (data: Record<string, unknown>) => void;
}

// SSE type -> sink method. A single source of truth so a new global event type
// is one line, and the router stays a pure lookup.
const ROUTES: Record<string, keyof ShellEventSink> = {
  reconnect: 'onReconnect',
  session_event: 'onSessionEvent',
  session_update: 'onSessionUpdate',
  job_update: 'onJobUpdate',
  schedule_update: 'onScheduleUpdate',
  terminal_state: 'onTerminalState',
  agent_status: 'onAgentStatus',
  history_update: 'onHistoryUpdate',
};

/** Dispatch one broadcast frame to its sink handler. Returns the sink method
 *  name it routed to (for tests / tracing), or null when the type is unhandled
 *  (hello / resync_required are consumed inside connectEvents, never here). */
export function routeShellEvent(
  event: SSEEvent,
  sink: ShellEventSink,
): keyof ShellEventSink | null {
  const method = ROUTES[event.type];
  if (!method) return null;
  const handler = sink[method];
  if (!handler) return method;
  if (method === 'onReconnect') (handler as () => void)();
  else
    (handler as (data: Record<string, unknown>) => void)(
      (event.data ?? {}) as Record<string, unknown>,
    );
  return method;
}
