import { describe, expect, it } from 'vitest';
import type { SessionRow } from '$lib/stores/sessions.svelte';
import {
  sessionSourceType,
  sessionRowState,
  groupSessions,
  isFinishedSession,
  isResumableSession,
  formatWhen,
  sessionTopic,
  chatNeedsAnswer,
  sessionHasAttention,
  chatsNeedingAnswer,
  attentionSessions,
} from './sessionModel';
import { attentionRecord as rec } from './__fixtures__/sessionRow';

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

describe('chatNeedsAnswer', () => {
  it('is false with nothing open', () => {
    expect(chatNeedsAnswer(base())).toBe(false);
    expect(sessionHasAttention(base())).toBe(false);
  });

  it('counts an unanswered delivery card', () => {
    const row = base({ attention: [rec('delivery', { kind: 'needs_ack' })] });
    expect(chatNeedsAnswer(row)).toBe(true);
    expect(sessionHasAttention(row)).toBe(true);
  });

  it('counts an outstanding question', () => {
    const row = base({ attention: [rec('ask')] });
    expect(chatNeedsAnswer(row)).toBe(true);
  });

  it('leaves a parked job to Jobs, while the rail still shows its session', () => {
    const row = base({ attention: [rec('job', { kind: 'stuck' })] });
    expect(chatNeedsAnswer(row)).toBe(false);
    expect(sessionHasAttention(row)).toBe(true);
  });

  it('groups a session with anything outstanding as active, not recent', () => {
    const rows = [
      base({ id: 'blocked', attention: [rec('job', { kind: 'stuck' })] }),
      base({ id: 'quiet' }),
    ];
    const attn = new Set(attentionSessions(rows).map((r) => r.id));
    const g = groupSessions(rows, { attn });
    expect(g.active.map((r) => r.id)).toEqual(['blocked']);
    expect(g.recent.map((r) => r.id)).toEqual(['quiet']);
  });
});

describe('chatsNeedingAnswer', () => {
  it('collects the rows the person still owes an answer', () => {
    const rows = [
      base({ id: 'ack', attention: [rec('delivery', { kind: 'needs_ack' })] }),
      base({ id: 'asked', attention: [rec('ask')] }),
      base({ id: 'blocked', attention: [rec('job', { kind: 'stuck' })] }),
      base({ id: 'quiet' }),
    ];
    expect(chatsNeedingAnswer(rows).map((r) => r.id)).toEqual(['ack', 'asked']);
    expect(attentionSessions(rows).map((r) => r.id)).toEqual(['ack', 'asked', 'blocked']);
  });

  it('drops a finished session, whatever it was waiting for', () => {
    const rows = [
      base({ id: 'done', status: 'completed', attention: [rec('delivery')] }),
      base({ id: 'failed', status: 'failed', attention: [rec('job', { kind: 'stuck' })] }),
    ];
    expect(chatsNeedingAnswer(rows)).toEqual([]);
    expect(attentionSessions(rows)).toEqual([]);
  });

  it('drops a compacted-away session, leaving its successor to speak for it', () => {
    const rows = [
      base({ id: 'old', attention: [rec('delivery')], superseded_by: 'new' }),
      base({ id: 'new', attention: [rec('delivery')] }),
    ];
    expect(chatsNeedingAnswer(rows).map((r) => r.id)).toEqual(['new']);
  });
});

describe('isResumableSession', () => {
  it('is true for a completed background session created to be talked to', () => {
    expect(
      isResumableSession(base({ source: 'background', status: 'completed', resumable: true })),
    ).toBe(true);
  });
  it('is false for a one-shot schedule or job run', () => {
    expect(isResumableSession(base({ source: 'schedule', status: 'completed' }))).toBe(false);
    expect(isResumableSession(base({ source: 'spawned', status: 'completed' }))).toBe(false);
  });
  it('is false once the run failed or was cancelled - those need a restart', () => {
    expect(isResumableSession(base({ status: 'failed', resumable: true }))).toBe(false);
    expect(isResumableSession(base({ status: 'cancelled', resumable: true }))).toBe(false);
  });
});

describe('resumable sessions in the rail', () => {
  const resumable = (over: Partial<SessionRow> = {}) =>
    base({ id: 'chatty', source: 'background', status: 'completed', resumable: true, ...over });

  it('stays in the recency flow instead of the ended bucket', () => {
    const g = groupSessions([resumable(), base({ id: 'done', status: 'completed' })], {
      attn: new Set(),
    });
    expect(g.recent.map((r) => r.id)).toEqual(['chatty']);
    expect(g.ended.map((r) => r.id)).toEqual(['done']);
  });
  it('reads as idle, the same as an interactive session awaiting input', () => {
    expect(sessionRowState(resumable())).toBe('idle');
    expect(sessionRowState(base({ status: 'completed' }))).toBe('done');
  });
  it('still counts as finished, so it is never auto-selected as the open chat', () => {
    expect(isFinishedSession(resumable())).toBe(true);
  });
});
