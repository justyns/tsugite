import { describe, expect, it } from 'vitest';
import { resolveDefaultSession } from './defaultSession';
import type { SessionRow } from '$lib/stores/sessions.svelte';

function row(id: string, extra: Partial<SessionRow> = {}): SessionRow {
  return {
    id,
    user_id: 'u',
    label: id,
    source: 'web',
    status: 'active',
    state: 'idle',
    created_at: '2026-07-16T00:00:00Z',
    last_active: null,
    parent_id: null,
    prompt: '',
    model: null,
    error: null,
    result: null,
    title: id,
    is_default: false,
    metadata: {},
    pinned: false,
    pin_position: null,
    last_viewed_at: null,
    superseded_by: null,
    unread: false,
    is_primary: false,
    busy: false,
    ...extra,
  };
}

describe('resolveDefaultSession', () => {
  it('never defaults to a superseded session - the live successor wins', () => {
    const rows = [row('old', { superseded_by: 'live', is_primary: true }), row('live')];
    expect(resolveDefaultSession(rows)).toBe('live');
  });

  it('keeps an explicit preferId even when that session is superseded (view source)', () => {
    const rows = [row('old', { superseded_by: 'live' }), row('live')];
    expect(resolveDefaultSession(rows, 'old')).toBe('old');
  });

  it('prefers primary, then pinned, then the first row', () => {
    expect(
      resolveDefaultSession([row('a'), row('b', { pinned: true }), row('c', { is_primary: true })]),
    ).toBe('c');
    expect(resolveDefaultSession([row('a'), row('b', { pinned: true })])).toBe('b');
    expect(resolveDefaultSession([row('a'), row('b')])).toBe('a');
  });

  it('never defaults to a finished session - a live row wins even over a finished primary', () => {
    const rows = [row('done', { status: 'completed', is_primary: true }), row('live')];
    expect(resolveDefaultSession(rows)).toBe('live');
  });

  it('skips finished sessions of every status and falls through to the live row', () => {
    expect(resolveDefaultSession([row('c', { status: 'completed' }), row('live')])).toBe('live');
    expect(resolveDefaultSession([row('f', { status: 'failed' }), row('live')])).toBe('live');
    expect(resolveDefaultSession([row('x', { status: 'cancelled' }), row('live')])).toBe('live');
  });

  it('does not default to a finished session even when it is pinned', () => {
    const rows = [row('done', { status: 'completed', pinned: true }), row('live')];
    expect(resolveDefaultSession(rows)).toBe('live');
  });

  it('keeps an explicit preferId even when that session is finished (deep-link to read)', () => {
    const rows = [row('done', { status: 'completed' }), row('live')];
    expect(resolveDefaultSession(rows, 'done')).toBe('done');
  });

  it('returns null when every session is finished - nothing live to auto-select', () => {
    const rows = [row('a', { status: 'completed' }), row('b', { status: 'failed' })];
    expect(resolveDefaultSession(rows)).toBeNull();
  });

  it('returns null when every session is superseded - nothing live to auto-select', () => {
    const rows = [row('a', { superseded_by: 'x' }), row('b', { superseded_by: 'y' })];
    expect(resolveDefaultSession(rows)).toBeNull();
  });

  it('falls back to the live default when a persisted preferId is stale (PWA cold start, session gone)', () => {
    // The PWA restores the last-selected conversation id on cold start. When that
    // session no longer exists, the id must not be returned verbatim - it falls
    // through to the normal live default instead of selecting a ghost row.
    const rows = [row('live', { is_primary: true }), row('other')];
    expect(resolveDefaultSession(rows, 'ghost-that-was-deleted')).toBe('live');
  });

  it('honors a persisted preferId that still resolves to a live row (PWA cold start, session present)', () => {
    const rows = [row('live', { is_primary: true }), row('restored')];
    expect(resolveDefaultSession(rows, 'restored')).toBe('restored');
  });
});
