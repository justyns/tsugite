import { describe, expect, it } from 'vitest';
import { applyEventToProgress, emptyProgress, progressStatusText } from './progress';

describe('emptyProgress', () => {
  it('starts at zero counts with a Starting label', () => {
    expect(emptyProgress()).toEqual({
      turn_count: 0,
      tool_count: 0,
      status_text: 'Starting...',
      last_event_time: null,
    });
  });
});

describe('progressStatusText', () => {
  it('labels turns, tools, and llm waits like the server', () => {
    expect(progressStatusText({ type: 'turn_start', turn: 3 })).toBe('Turn 3...');
    expect(progressStatusText({ type: 'tool_call', tool: 'read_file' })).toBe('Tool: read_file');
    expect(progressStatusText({ type: 'llm_wait_progress', elapsed_seconds: 12 })).toBe(
      'Waiting on LLM (12s)',
    );
    expect(progressStatusText({ type: 'hook_status', message: 'running hooks' })).toBe(
      'running hooks',
    );
  });

  it('returns null for a tool_result with no real tool name', () => {
    expect(progressStatusText({ type: 'tool_result', tool: 'unknown' })).toBeNull();
    expect(progressStatusText({ type: 'tool_result', tool: 'grep' })).toBe('Tool: grep');
  });

  it('returns null for an unknown event type', () => {
    expect(progressStatusText({ type: 'mystery' })).toBeNull();
  });
});

describe('applyEventToProgress', () => {
  it('advances the turn counter only when the turn number grows', () => {
    let p = emptyProgress();
    p = applyEventToProgress(p, { type: 'turn_start', turn: 1 });
    expect(p.turn_count).toBe(1);
    p = applyEventToProgress(p, { type: 'turn_start', turn: 3 });
    expect(p.turn_count).toBe(3);
    // A stale/replayed lower turn does not regress the counter.
    p = applyEventToProgress(p, { type: 'turn_start', turn: 2 });
    expect(p.turn_count).toBe(3);
  });

  it('counts named tool_result events but not tool_call', () => {
    let p = emptyProgress();
    p = applyEventToProgress(p, { type: 'tool_call', tool: 'read_file' });
    expect(p.tool_count).toBe(0);
    p = applyEventToProgress(p, { type: 'tool_result', tool: 'read_file' });
    p = applyEventToProgress(p, { type: 'tool_result', tool: 'grep' });
    expect(p.tool_count).toBe(2);
    // unknown-named results are not counted
    p = applyEventToProgress(p, { type: 'tool_result', tool: 'unknown' });
    expect(p.tool_count).toBe(2);
  });

  it('resets to an idle rollup on a session-end event', () => {
    let p = emptyProgress();
    p = applyEventToProgress(p, { type: 'turn_start', turn: 5 });
    p = applyEventToProgress(p, { type: 'final_result', timestamp: '2026-07-14T00:00:00Z' });
    expect(p).toEqual({
      turn_count: 0,
      tool_count: 0,
      status_text: '',
      last_event_time: '2026-07-14T00:00:00Z',
    });
  });

  it('tracks last_event_time and keeps the prior label when an event has none', () => {
    let p = emptyProgress();
    p = applyEventToProgress(p, { type: 'turn_start', turn: 1, timestamp: 't1' });
    expect(p.status_text).toBe('Turn 1...');
    // an event with no label keeps the last status but updates the timestamp
    p = applyEventToProgress(p, { type: 'mystery', timestamp: 't2' });
    expect(p.status_text).toBe('Turn 1...');
    expect(p.last_event_time).toBe('t2');
  });

  it('does not mutate the input rollup', () => {
    const p = emptyProgress();
    const next = applyEventToProgress(p, { type: 'turn_start', turn: 2 });
    expect(p.turn_count).toBe(0);
    expect(next.turn_count).toBe(2);
  });
});
