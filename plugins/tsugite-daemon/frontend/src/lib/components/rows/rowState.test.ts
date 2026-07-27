import { describe, expect, it } from 'vitest';
import {
  buildSessionRowAriaLabel,
  checkStatePrefix,
  clampPct,
  sessionStateMeta,
  sourceTypeLabel,
  spaceStateMeta,
} from './rowState';

describe('sessionStateMeta', () => {
  it('maps needs-you to the warn color, the q icon, and the card wording', () => {
    expect(sessionStateMeta('needs-you')).toEqual({
      label: 'awaiting your input',
      color: 'var(--st-warn)',
      spin: false,
      icon: 'q',
    });
  });

  it('marks running and thinking as spinning states with distinct colors', () => {
    expect(sessionStateMeta('running')).toMatchObject({ spin: true, color: 'var(--st-ok)' });
    expect(sessionStateMeta('thinking')).toMatchObject({ spin: true, color: 'var(--st-info)' });
    expect(sessionStateMeta('running').icon).toBeUndefined();
  });

  it('gives every non-spinning state an icon', () => {
    for (const state of ['idle', 'done', 'failed', 'needs-you'] as const) {
      expect(sessionStateMeta(state).spin).toBe(false);
      expect(sessionStateMeta(state).icon).toBeDefined();
    }
  });
});

describe('sourceTypeLabel', () => {
  it('abbreviates research to res and passes the rest through unchanged', () => {
    expect(sourceTypeLabel('research')).toBe('res');
    expect(sourceTypeLabel('ops')).toBe('ops');
    expect(sourceTypeLabel('code')).toBe('code');
    expect(sourceTypeLabel('chat')).toBe('chat');
  });
});

describe('buildSessionRowAriaLabel', () => {
  it('joins the title and the state word', () => {
    expect(
      buildSessionRowAriaLabel({
        title: 'ops: nightly backup failing on prune',
        state: 'needs-you',
      }),
    ).toBe('ops: nightly backup failing on prune, awaiting your input');
  });

  it('appends unread when the row carries the unread marker', () => {
    expect(
      buildSessionRowAriaLabel({
        title: 'research: local whisper models',
        state: 'idle',
        isUnread: true,
      }),
    ).toBe('research: local whisper models, idle, unread');
  });

  it('omits the unread suffix by default', () => {
    expect(buildSessionRowAriaLabel({ title: 'chat: naming things', state: 'idle' })).toBe(
      'chat: naming things, idle',
    );
  });
});

describe('spaceStateMeta', () => {
  it('maps each of the four states to its label', () => {
    expect(spaceStateMeta('working').label).toBe('working');
    expect(spaceStateMeta('blocked').label).toBe('blocked');
    expect(spaceStateMeta('idle').label).toBe('idle');
    expect(spaceStateMeta('done').label).toBe('done');
  });

  it('only working spins; the rest carry a static icon', () => {
    expect(spaceStateMeta('working')).toMatchObject({ spin: true });
    expect(spaceStateMeta('working').icon).toBeUndefined();
    expect(spaceStateMeta('blocked')).toMatchObject({ spin: false, icon: 'q' });
    expect(spaceStateMeta('idle')).toMatchObject({ spin: false, icon: 'ring' });
    expect(spaceStateMeta('done')).toMatchObject({ spin: false, icon: 'check' });
  });
});

describe('clampPct', () => {
  it('passes normal percentages through', () => {
    expect(clampPct(41)).toBe(41);
    expect(clampPct(0)).toBe(0);
    expect(clampPct(100)).toBe(100);
  });

  it('clamps below zero and above 100', () => {
    expect(clampPct(-5)).toBe(0);
    expect(clampPct(140)).toBe(100);
  });

  it('treats NaN as 0 instead of poisoning the bar width', () => {
    expect(clampPct(NaN)).toBe(0);
  });
});

describe('checkStatePrefix', () => {
  it('maps every state to a screen-reader prefix', () => {
    expect(checkStatePrefix('pending')).toBe('Pending');
    expect(checkStatePrefix('active')).toBe('Verifying');
    expect(checkStatePrefix('pass')).toBe('Passed');
    expect(checkStatePrefix('fail')).toBe('Failed');
  });
});
