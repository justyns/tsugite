// Session lifecycle events that end a live progress trace. Kept in sync with
// the Python _SESSION_END_EVENT_TYPES set in tsugite/daemon/session_store.py.
export const SESSION_END_EVENTS = new Set(['session_complete', 'session_error', 'session_cancelled']);

// Events skipped during replay-on-select so we don't re-announce the session
// or surface the final result twice (loadHistory already provides it).
export const REPLAY_SKIP_EVENTS = new Set([...SESSION_END_EVENTS, 'session_start', 'final_result']);

// Map an incoming session event to a short human-readable status label.
// Mirrors Python `_progress_status_text` in tsugite/daemon/session_store.py.
export function progressStatusFor(evType, data) {
  if (evType === 'session_start') return 'Starting...';
  if (evType === 'init') return data.agent ? `Agent: ${data.agent}` : 'Starting...';
  if (evType === 'turn_start') return data.turn != null ? `Turn ${data.turn}...` : 'Working...';
  if (evType === 'thought') return 'Thinking...';
  if (evType === 'reasoning_content') return 'Reasoning...';
  if (evType === 'tool_result') {
    const tool = data.tool || 'unknown';
    return tool === 'unknown' ? null : `Tool: ${tool}`;
  }
  if (evType === 'hook_status') return data.message || null;
  return null;
}

// Normalize a server-side `progress` payload (snake_case) into the local cache shape (camelCase).
export function progressFromPayload(p) {
  if (!p) return null;
  return {
    turnCount: p.turn_count || 0,
    toolCount: p.tool_count || 0,
    statusText: p.status_text || 'Starting...',
    lastEventTime: p.last_event_time || null,
  };
}
