import { describe, expect, it } from 'vitest';
import type { SessionRow } from '$lib/stores/sessions.svelte';
import {
  sessionSourceType,
  sessionRowState,
  groupSessions,
  isFinishedSession,
  formatWhen,
  sessionTopic,
  sessionNeedsYou,
  needsYouSessions,
} from './sessionModel';

const base = (over: Partial<SessionRow> = {}): SessionRow =>
  ({
    id: 'sess-1',
    user_id: 'web-alice',
    label: 'Web: web-alice',
    source: 'web',
    status: 'active',
    state: 'active',
    created_at: '2026-07-14T15:00:00+00:00',
    last_active: '2026-07-14T15:00:00+00:00',
    parent_id: null,
    prompt: '',
    model: null,
    error: null,
    result: null,
    title: 'refactor sse',
    is_default: false,
    metadata: {},
    pinned: false,
    pin_position: null,
    last_viewed_at: null,
    superseded_by: null,
    unread: false,
    is_primary: false,
    busy: false,
    ...over,
  }) as SessionRow;

describe('sessionSourceType', () => {
  it('reads metadata.type and clamps to the 4 known categories', () => {
    expect(sessionSourceType(base({ metadata: { type: 'code' } }))).toBe('code');
    expect(sessionSourceType(base({ metadata: { type: 'ops' } }))).toBe('ops');
    expect(sessionSourceType(base({ metadata: { type: 'research' } }))).toBe('research');
  });
  it('defaults to chat when type is missing or unknown', () => {
    expect(sessionSourceType(base({ metadata: {} }))).toBe('chat');
    expect(sessionSourceType(base({ metadata: { type: 'wat' } }))).toBe('chat');
  });
});

describe('sessionRowState', () => {
  it('maps failed status to failed', () => {
    expect(sessionRowState(base({ status: 'failed' }))).toBe('failed');
  });
  it('maps completed/cancelled to done', () => {
    expect(sessionRowState(base({ status: 'completed' }))).toBe('done');
    expect(sessionRowState(base({ status: 'cancelled' }))).toBe('done');
  });
  it('needs-you wins over busy when a question is pending', () => {
    expect(sessionRowState(base({ busy: true }), { needsYou: true })).toBe('needs-you');
  });
  it('busy with an LLM-wait status_text reads as thinking, else running', () => {
    const waiting = base({
      busy: true,
      progress: { status_text: 'Waiting on LLM (12s)' } as never,
    });
    const tooling = base({ busy: true, progress: { status_text: 'tool: read_file' } as never });
    expect(sessionRowState(waiting)).toBe('thinking');
    expect(sessionRowState(tooling)).toBe('running');
  });
  it('falls back to idle for an active-but-quiet session', () => {
    expect(sessionRowState(base({ status: 'active', busy: false }))).toBe('idle');
  });
});

describe('isFinishedSession', () => {
  it('is true for the finished statuses (completed/failed/cancelled)', () => {
    expect(isFinishedSession(base({ status: 'completed' }))).toBe(true);
    expect(isFinishedSession(base({ status: 'failed' }))).toBe(true);
    expect(isFinishedSession(base({ status: 'cancelled' }))).toBe(true);
  });
  it('is false for a live session', () => {
    expect(isFinishedSession(base({ status: 'active' }))).toBe(false);
    expect(isFinishedSession(base({ status: 'running' }))).toBe(false);
  });
});

describe('groupSessions', () => {
  it('buckets into pinned / active / recent / ended', () => {
    const rows = [
      base({ id: 'p', pinned: true }),
      base({ id: 'busy', busy: true }),
      base({ id: 'attn', status: 'active' }),
      base({ id: 'quiet', status: 'active' }),
      base({ id: 'done', status: 'completed' }),
    ];
    const g = groupSessions(rows, { attn: new Set(['attn']) });
    expect(g.pinned.map((r) => r.id)).toEqual(['p']);
    // busy + needs-you land in active; a quiet live row is recent
    expect(g.active.map((r) => r.id).sort()).toEqual(['attn', 'busy']);
    expect(g.recent.map((r) => r.id)).toEqual(['quiet']);
    // a finished row leaves the recency flow for its own bucket
    expect(g.ended.map((r) => r.id)).toEqual(['done']);
  });
  it('sends every finished status to ended, out of recent/active', () => {
    const rows = [
      base({ id: 'c', status: 'completed' }),
      base({ id: 'f', status: 'failed' }),
      base({ id: 'x', status: 'cancelled' }),
    ];
    const g = groupSessions(rows, { attn: new Set() });
    expect(g.ended.map((r) => r.id)).toEqual(['c', 'f', 'x']);
    expect(g.recent).toHaveLength(0);
    expect(g.active).toHaveLength(0);
  });
  it('keeps a pinned finished session pinned - pinned wins over ended', () => {
    const rows = [base({ id: 'p', pinned: true, status: 'completed' })];
    const g = groupSessions(rows, { attn: new Set() });
    expect(g.pinned.map((r) => r.id)).toEqual(['p']);
    expect(g.ended).toHaveLength(0);
  });
  it('never sorts a finished session into active, even with a pending ask', () => {
    const rows = [base({ id: 'done', status: 'completed' })];
    const g = groupSessions(rows, { attn: new Set(['done']) });
    expect(g.ended.map((r) => r.id)).toEqual(['done']);
    expect(g.active).toHaveLength(0);
  });
  it('a pinned row never also appears in active/recent/ended', () => {
    const rows = [base({ id: 'p', pinned: true, busy: true })];
    const g = groupSessions(rows, { attn: new Set() });
    expect(g.pinned).toHaveLength(1);
    expect(g.active).toHaveLength(0);
    expect(g.recent).toHaveLength(0);
    expect(g.ended).toHaveLength(0);
  });
});

describe('formatWhen', () => {
  const now = new Date('2026-07-14T15:00:00+00:00').getTime();
  it('renders now / minutes / hours / date buckets', () => {
    expect(formatWhen('2026-07-14T14:59:50+00:00', now)).toBe('now');
    expect(formatWhen('2026-07-14T14:48:00+00:00', now)).toBe('12m');
    expect(formatWhen('2026-07-14T13:00:00+00:00', now)).toBe('2h');
    expect(formatWhen('2026-07-12T13:00:00+00:00', now)).toMatch(/jul 12/i);
  });
  it('is empty for a null timestamp', () => {
    expect(formatWhen(null, now)).toBe('');
  });
});

describe('sessionTopic', () => {
  it('prefers metadata.topic, else derives from the label', () => {
    expect(sessionTopic(base({ metadata: { topic: 'sse backoff' } }))).toBe('sse backoff');
    expect(sessionTopic(base({ metadata: {}, label: 'Web: web-alice' }))).toBe('Web: web-alice');
  });
});

describe('sessionNeedsYou', () => {
  it('reads the daemon status_text for an outstanding question', () => {
    expect(sessionNeedsYou(base({ progress: { status_text: 'Awaiting answer' } as never }))).toBe(
      true,
    );
    expect(sessionNeedsYou(base({ progress: { status_text: 'Tool: grep' } as never }))).toBe(false);
    expect(sessionNeedsYou(base())).toBe(false);
  });

  it('counts a durable needs-attention flag from a needs-ack delivery', () => {
    expect(sessionNeedsYou(base({ needs_attention: true }))).toBe(true);
    expect(sessionNeedsYou(base({ needs_attention: false }))).toBe(false);
  });

  it('counts a job parked on the person as needing you', () => {
    expect(sessionNeedsYou(base(), 1)).toBe(true);
    expect(sessionNeedsYou(base(), 0)).toBe(false);
  });

  it('groups a session with a parked job as active, not recent', () => {
    const rows = [base({ id: 'blocked' }), base({ id: 'quiet' })];
    const parked: Record<string, number> = { blocked: 1 };
    const attn = new Set(rows.filter((r) => sessionNeedsYou(r, parked[r.id])).map((r) => r.id));
    const g = groupSessions(rows, { attn });
    expect(g.active.map((r) => r.id)).toEqual(['blocked']);
    expect(g.recent.map((r) => r.id)).toEqual(['quiet']);
  });

  it('groups a needs-attention row as active, not recent', () => {
    const rows = [base({ id: 'ack', needs_attention: true }), base({ id: 'quiet' })];
    const attn = new Set(rows.filter((r) => sessionNeedsYou(r)).map((r) => r.id));
    const g = groupSessions(rows, { attn });
    expect(g.active.map((r) => r.id)).toEqual(['ack']);
    expect(g.recent.map((r) => r.id)).toEqual(['quiet']);
  });
});

describe('needsYouSessions', () => {
  const tally = (parked: Record<string, number>) =>
    new Map(Object.entries(parked).map(([id, n]) => [id, { open: n, parked: n }]));

  it('collects the rows waiting on the person, jobs folded in', () => {
    const rows = [
      base({ id: 'ack', needs_attention: true }),
      base({ id: 'asked', progress: { status_text: 'Awaiting answer' } as never }),
      base({ id: 'blocked' }),
      base({ id: 'quiet' }),
    ];
    expect(needsYouSessions(rows, tally({ blocked: 1 })).map((r) => r.id)).toEqual([
      'ack',
      'asked',
      'blocked',
    ]);
  });

  it('drops a finished session, whatever it was waiting for', () => {
    const rows = [
      base({ id: 'done', status: 'completed', needs_attention: true }),
      base({ id: 'failed', status: 'failed' }),
    ];
    expect(needsYouSessions(rows, tally({ failed: 2 }))).toEqual([]);
  });

  it('drops a compacted-away session, leaving its successor to speak for it', () => {
    const rows = [
      base({ id: 'old', needs_attention: true, superseded_by: 'new' }),
      base({ id: 'new', needs_attention: true }),
    ];
    expect(needsYouSessions(rows).map((r) => r.id)).toEqual(['new']);
  });

  it('ignores jobs entirely when given no tally', () => {
    expect(needsYouSessions([base({ id: 'blocked' })])).toEqual([]);
  });
});
