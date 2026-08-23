/**
 * Per-session progress rollup reducer. Mirrors the daemon's
 * session_store._apply_event_to_progress / _progress_status_text (Python) so the
 * client can fold cross-session `session_event` broadcasts into the same
 * {turn_count, tool_count, status_text, last_event_time} shape the server serves
 * as row['progress'] - without refetching the whole session list on every tick.
 *
 * Pure and DOM-free (node-unit-tested). The sessions store wires it to $state.
 */

export interface Progress {
  turn_count: number;
  tool_count: number;
  status_text: string;
  last_event_time: string | null;
}

/** A folded event: the broadcast's `event_type` normalised to `type`, plus its
 *  data fields spread alongside. */
export interface ProgressEvent {
  type?: string;
  turn?: unknown;
  tool?: unknown;
  name?: unknown;
  agent?: unknown;
  message?: unknown;
  elapsed_seconds?: unknown;
  timestamp?: string;
  [key: string]: unknown;
}

// Turn-end markers reset the rollup to an empty/idle state. Kept in sync with
// session_store.SESSION_END_EVENT_TYPES.
const SESSION_END_TYPES = new Set([
  'session_complete',
  'session_error',
  'session_cancelled',
  'final_result',
  'error',
  'cancelled',
  'session_end',
]);

export function emptyProgress(): Progress {
  return { turn_count: 0, tool_count: 0, status_text: 'Starting...', last_event_time: null };
}

/** Only named-tool events count toward the tool counter; tool_call is skipped
 *  because its matching tool_result fires later for the same invocation. */
function isRealToolEvent(event: ProgressEvent): boolean {
  if (event.type === 'tool_result') return (event.tool || 'unknown') !== 'unknown';
  if (event.type === 'tool_invocation') return Boolean(event.name);
  return false;
}

/** Short status label for a mid-session progress event, or null when the event
 *  carries no display text. Mirrors _progress_status_text. */
export function progressStatusText(event: ProgressEvent): string | null {
  switch (event.type) {
    case 'session_start':
      return 'Starting...';
    case 'init':
      return event.agent ? `Agent: ${event.agent}` : 'Starting...';
    case 'turn_start':
      return event.turn != null ? `Turn ${event.turn}...` : 'Working...';
    case 'thought':
      return 'Thinking...';
    case 'reasoning_content':
      return 'Reasoning...';
    case 'tool_result':
      return isRealToolEvent(event) ? `Tool: ${event.tool}` : null;
    case 'tool_call':
      return event.tool ? `Tool: ${event.tool}` : null;
    case 'tool_invocation':
      return event.name ? `Tool: ${event.name}` : null;
    case 'code_execution':
      return 'Running code...';
    case 'model_request':
      return 'Waiting on LLM...';
    case 'hook_status':
      return typeof event.message === 'string' ? event.message : null;
    case 'llm_wait_progress':
      return event.elapsed_seconds
        ? `Waiting on LLM (${event.elapsed_seconds}s)`
        : 'Waiting on LLM...';
    default:
      return null;
  }
}

/** Fold one event into a progress rollup, returning a new object (the caller
 *  holds it in $state, so this stays immutable rather than mutating in place). */
export function applyEventToProgress(prev: Progress, event: ProgressEvent): Progress {
  const last_event_time = event.timestamp ?? prev.last_event_time;
  if (event.type && SESSION_END_TYPES.has(event.type)) {
    return { turn_count: 0, tool_count: 0, status_text: '', last_event_time };
  }
  const next: Progress = { ...prev, last_event_time };
  if (event.type === 'turn_start') {
    const turn = event.turn;
    if (typeof turn === 'number' && turn > prev.turn_count) next.turn_count = turn;
  } else if (isRealToolEvent(event)) {
    next.tool_count = prev.tool_count + 1;
  }
  const label = progressStatusText(event);
  if (label) next.status_text = label;
  return next;
}
