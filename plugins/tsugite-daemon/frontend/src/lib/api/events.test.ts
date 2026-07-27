import { describe, expect, it, vi } from 'vitest';
import { routeShellEvent } from './events';

describe('routeShellEvent', () => {
  it('routes each global broadcast type to its sink method with the data payload', () => {
    const sink = {
      onSessionEvent: vi.fn(),
      onSessionUpdate: vi.fn(),
      onJobUpdate: vi.fn(),
      onScheduleUpdate: vi.fn(),
      onTerminalState: vi.fn(),
      onAgentStatus: vi.fn(),
      onHistoryUpdate: vi.fn(),
    };
    routeShellEvent(
      { type: 'session_event', data: { session_id: 's1', event_type: 'turn_start' } },
      sink,
    );
    routeShellEvent({ type: 'job_update', data: { job_id: 'j1' } }, sink);
    routeShellEvent(
      { type: 'terminal_state', data: { terminal_id: 't1', state: 'running' } },
      sink,
    );
    expect(sink.onSessionEvent).toHaveBeenCalledWith({
      session_id: 's1',
      event_type: 'turn_start',
    });
    expect(sink.onJobUpdate).toHaveBeenCalledWith({ job_id: 'j1' });
    expect(sink.onTerminalState).toHaveBeenCalledWith({ terminal_id: 't1', state: 'running' });
  });

  it('invokes the reconnect handler with no argument', () => {
    const onReconnect = vi.fn();
    expect(routeShellEvent({ type: 'reconnect' }, { onReconnect })).toBe('onReconnect');
    expect(onReconnect).toHaveBeenCalledWith();
  });

  it('defaults a missing data payload to an empty object', () => {
    const onAgentStatus = vi.fn();
    routeShellEvent({ type: 'agent_status' }, { onAgentStatus });
    expect(onAgentStatus).toHaveBeenCalledWith({});
  });

  it('returns null for an unrouted type (hello/resync are handled upstream)', () => {
    expect(routeShellEvent({ type: 'hello', data: {} }, {})).toBeNull();
    expect(routeShellEvent({ type: 'stream_chunk' }, {})).toBeNull();
  });

  it('returns the method name even when the sink omits that handler', () => {
    expect(routeShellEvent({ type: 'job_update', data: {} }, {})).toBe('onJobUpdate');
  });
});
