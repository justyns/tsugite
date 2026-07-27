import { describe, expect, it } from 'vitest';
import {
  elapsedSeconds,
  formatBytes,
  isLiveTerminal,
  terminalIndicator,
  terminalPill,
  terminalTabState,
} from './termState';
import type { TerminalState } from '$lib/stores/terminals.svelte';

const ALL_STATES: TerminalState[] = [
  'starting',
  'running',
  'succeeded',
  'failed',
  'cancelled',
  'stream_lost',
];

describe('terminalPill', () => {
  it('maps each backend state to the pill bucket + label', () => {
    expect(terminalPill('starting')).toMatchObject({ st: 'queued', label: 'starting', spin: true });
    expect(terminalPill('running')).toMatchObject({ st: 'running', label: 'running', spin: true });
    expect(terminalPill('cancelled')).toMatchObject({
      st: 'cancelled',
      label: 'killed',
      spin: false,
    });
    expect(terminalPill('stream_lost')).toMatchObject({
      st: 'stuck',
      label: 'stream lost',
      spin: false,
    });
  });

  it('shows the real exit code for terminated states, falling back per state', () => {
    expect(terminalPill('succeeded', 0).label).toBe('exit 0');
    expect(terminalPill('failed', 137).label).toBe('exit 137');
    // null exit code -> per-state fallback (succeeded=0, failed=1)
    expect(terminalPill('succeeded', null).label).toBe('exit 0');
    expect(terminalPill('failed', null).label).toBe('exit 1');
    expect(terminalPill('succeeded').label).toBe('exit 0');
  });

  it('only running/starting spin (never a terminated state)', () => {
    for (const s of ALL_STATES) {
      expect(terminalPill(s).spin).toBe(s === 'running' || s === 'starting');
    }
  });
});

describe('terminalIndicator', () => {
  it('spins live states and gives terminated states a static icon', () => {
    expect(terminalIndicator('running')).toEqual({ spin: true, icon: 'play' });
    expect(terminalIndicator('starting')).toEqual({ spin: true, icon: 'clock' });
    expect(terminalIndicator('failed')).toEqual({ spin: false, icon: 'x' });
    expect(terminalIndicator('stream_lost')).toEqual({ spin: false, icon: 'alert' });
  });
});

describe('terminalTabState', () => {
  it('folds terminal states onto the mux tab-dot vocabulary', () => {
    expect(terminalTabState('running')).toBe('busy');
    expect(terminalTabState('starting')).toBe('busy');
    expect(terminalTabState('succeeded')).toBe('done');
    expect(terminalTabState('failed')).toBe('error');
    expect(terminalTabState('stream_lost')).toBe('blocked');
    expect(terminalTabState('cancelled')).toBe('idle');
  });
});

describe('isLiveTerminal', () => {
  it('is true only for starting/running', () => {
    for (const s of ALL_STATES) {
      expect(isLiveTerminal(s)).toBe(s === 'starting' || s === 'running');
    }
  });
});

describe('formatBytes', () => {
  it('matches the B / KB / MB byte thresholds', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(915)).toBe('915 B');
    expect(formatBytes(1000)).toBe('1000 B'); // strictly > 1000 to switch units
    expect(formatBytes(1001)).toBe('1 KB');
    expect(formatBytes(48210)).toBe('48 KB');
    expect(formatBytes(912400)).toBe('912 KB');
    expect(formatBytes(2_100_000)).toBe('2.1 MB');
  });
});

describe('elapsedSeconds', () => {
  const created = '2026-07-14T10:00:00.000Z';

  it('measures created -> now while live', () => {
    const now = Date.parse('2026-07-14T10:01:30.000Z');
    expect(elapsedSeconds(created, null, now)).toBe(90);
  });

  it('freezes at created -> resolved once terminated (ignores now)', () => {
    const resolved = '2026-07-14T10:00:42.000Z';
    const now = Date.parse('2026-07-14T12:00:00.000Z');
    expect(elapsedSeconds(created, resolved, now)).toBe(42);
  });

  it('never goes negative and survives bad timestamps', () => {
    const now = Date.parse('2026-07-14T09:59:00.000Z'); // before created
    expect(elapsedSeconds(created, null, now)).toBe(0);
    expect(elapsedSeconds('not-a-date', null, now)).toBe(0);
  });
});
