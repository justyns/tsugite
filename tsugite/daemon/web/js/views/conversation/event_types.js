export const SESSION_END_EVENTS = new Set(['session_complete', 'session_error', 'session_cancelled']);

// Turn-end events on a session that itself stays active. `error`/`cancelled`
// are the unprefixed names the HTTP adapter persists for interactive turns.
export const TURN_END_EVENTS = new Set(['final_result', 'error', 'cancelled']);

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
  if (evType === 'llm_wait_progress') {
    return data.elapsed_seconds ? `Waiting on LLM (${data.elapsed_seconds}s)` : 'Waiting on LLM...';
  }
  return null;
}

// Build the bubble for a `final_result` event. Returns null when there's
// nothing to render (no structured payload AND no string answer).
// Pre-stringifies `result_data` so the template doesn't re-run JSON.stringify
// on every Alpine re-render.
export function finalResultBubble({ result, result_data }) {
  if (result_data != null) {
    return { type: 'return_value', data: result_data, dataText: JSON.stringify(result_data, null, 2) };
  }
  if (result) return { type: 'agent', text: result };
  return null;
}

// Shared by history replay and live streaming so both paths group reasoning chunks identically.
export function appendReasoningChunk(bubbles, step, content) {
  const last = bubbles[bubbles.length - 1];
  if (last?.type === 'reasoning' && last.step === step) {
    last.text += content;
  } else {
    bubbles.push({ type: 'reasoning', text: content, step });
  }
}

export function attachReasoningTokens(bubbles, step, tokens) {
  for (let i = bubbles.length - 1; i >= 0; i--) {
    if (bubbles[i].type === 'reasoning' && bubbles[i].step === step) {
      bubbles[i].tokens = tokens;
      return;
    }
  }
}

// Empty `status_text` from the backend means "live progress cleared" (turn ended);
// preserve it as '' so sessionProgressLabel can render nothing instead of "Starting...".
export function progressFromPayload(p) {
  if (!p) return null;
  return {
    turnCount: p.turn_count || 0,
    toolCount: p.tool_count || 0,
    statusText: typeof p.status_text === 'string' ? p.status_text : 'Starting...',
    lastEventTime: p.last_event_time || null,
  };
}
