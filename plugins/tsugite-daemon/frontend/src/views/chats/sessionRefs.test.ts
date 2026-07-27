import { describe, expect, it } from 'vitest';
import type { SessionRow } from '$lib/stores/sessions.svelte';
import { buildSessionRefs } from './sessionRefs';

const NOW = Date.parse('2026-07-14T15:00:00+00:00');

const base = (over: Partial<SessionRow> = {}): SessionRow =>
  ({
    id: 'sess-1',
    user_id: 'web-alice',
    label: 'Web: web-alice',
    source: 'web',
    status: 'active',
    state: 'active',
    created_at: '2026-07-14T14:00:00+00:00',
    last_active: '2026-07-14T14:58:00+00:00',
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

describe('buildSessionRefs', () => {
  it('maps rows to session RefItems with a status · when detail', () => {
    const [item] = buildSessionRefs([base()], null, NOW);
    expect(item).toMatchObject({
      id: 'sess-1',
      kind: 'session',
      label: 'refactor sse',
      detail: 'active · 2m',
      group: 'Sessions',
    });
  });

  it('excludes the current chat (you do not reference the chat you are in)', () => {
    const rows = [base({ id: 'a' }), base({ id: 'b' }), base({ id: 'c' })];
    const ids = buildSessionRefs(rows, 'b', NOW).map((r) => r.id);
    expect(ids).toEqual(['a', 'c']);
  });

  it('falls back to "Untitled chat" when a row has no title', () => {
    const [item] = buildSessionRefs([base({ title: null })], null, NOW);
    expect(item?.label).toBe('Untitled chat');
  });

  it('caps at the 25 most-recent rows', () => {
    const rows = Array.from({ length: 40 }, (_, i) => base({ id: `s${i}` }));
    expect(buildSessionRefs(rows, null, NOW)).toHaveLength(25);
  });

  it('omits the detail when there is neither status nor a timestamp', () => {
    const [item] = buildSessionRefs([base({ status: '', last_active: null })], null, NOW);
    expect(item?.detail).toBeUndefined();
  });
});
