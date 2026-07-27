import { describe, expect, it } from 'vitest';
import {
  buildRows,
  matchItem,
  MAX_RESULTS_EMPTY,
  MAX_RESULTS_QUERY,
  type PaletteItem,
} from './palette-match';

function item(group: string, label: string, extra: Partial<PaletteItem> = {}): PaletteItem {
  return { group, icon: 'chat', label, meta: '', ...extra };
}

describe('matchItem', () => {
  it('scores a contiguous substring by earliness and highlights the run', () => {
    const m = matchItem('sse', 'refactor: sse reconnect backoff');
    expect(m).not.toBeNull();
    expect(m!.score).toBe(90); // 100 - index(10)
    expect(m!.highlight).toEqual([10, 13]);
  });

  it('ranks an earlier substring above a later one', () => {
    const early = matchItem('npm', 'npm test --watch')!;
    const late = matchItem('npm', 'run npm test')!;
    expect(early.score).toBeGreaterThan(late.score);
    expect(early.score).toBe(100); // index 0
  });

  it('is case-insensitive and highlights the original-case run', () => {
    const m = matchItem('SSE', 'refactor: SSE reconnect')!;
    expect(m.highlight).toEqual([10, 13]);
    expect('refactor: SSE reconnect'.slice(10, 13)).toBe('SSE');
  });

  it('falls back to a subsequence match with no highlight', () => {
    const m = matchItem('nbp', 'nightly backup prune');
    expect(m).not.toBeNull();
    expect(m!.highlight).toBeNull();
    expect(m!.score).toBe(6); // hits(3) * 2
  });

  it('scores a subsequence below any substring', () => {
    const sub = matchItem('nbp', 'nightly backup prune')!; // subsequence
    const substr = matchItem('backup', 'nightly backup prune')!; // contiguous
    expect(substr.score).toBeGreaterThan(sub.score);
  });

  it('requires subsequence characters in order', () => {
    expect(matchItem('pn', 'prune')).not.toBeNull(); // p..n in order
    expect(matchItem('np', 'prune')).toBeNull(); // n before p -> no match
  });

  it('returns null when a character is absent', () => {
    expect(matchItem('xyz', 'abc')).toBeNull();
  });

  it('treats an empty query as a neutral, unhighlighted match', () => {
    expect(matchItem('', 'anything')).toEqual({ score: 0, highlight: null });
  });
});

describe('buildRows - empty query (grouped)', () => {
  const items = [item('sessions', 'alpha'), item('sessions', 'beta'), item('jobs', 'gamma')];

  it('interleaves one header per group run, in source order', () => {
    const rows = buildRows(items, '');
    expect(rows.map((r) => (r.kind === 'group' ? `#${r.label}` : r.item.label))).toEqual([
      '#sessions',
      'alpha',
      'beta',
      '#jobs',
      'gamma',
    ]);
  });

  it('assigns sequential selectable indices to item rows only', () => {
    const rows = buildRows(items, '');
    const itemRows = rows.filter((r) => r.kind === 'item');
    expect(itemRows.map((r) => (r.kind === 'item' ? r.index : -1))).toEqual([0, 1, 2]);
  });

  it('carries no highlight when unfiltered', () => {
    const rows = buildRows(items, '');
    for (const r of rows) if (r.kind === 'item') expect(r.highlight).toBeNull();
  });

  it('treats a whitespace-only query as empty', () => {
    expect(buildRows(items, '   ')).toEqual(buildRows(items, ''));
  });

  it(`caps item rows at ${MAX_RESULTS_EMPTY}`, () => {
    const many = Array.from({ length: 30 }, (_, i) => item('g', `s${i}`));
    const rows = buildRows(many, '');
    const itemRows = rows.filter((r) => r.kind === 'item');
    expect(itemRows).toHaveLength(MAX_RESULTS_EMPTY);
    expect(rows.filter((r) => r.kind === 'group')).toHaveLength(1);
  });
});

describe('buildRows - with query (flat, ranked)', () => {
  const items = [
    item('sessions', 'refactor sse reconnect'),
    item('jobs', 'fix flaky sse test'),
    item('terminals', 'npm test sse watch'),
    item('files', 'unrelated readme'),
  ];

  it('drops non-matches and emits no group headers', () => {
    const rows = buildRows(items, 'sse');
    expect(rows.every((r) => r.kind === 'item')).toBe(true);
    expect(rows).toHaveLength(3);
  });

  it('sorts by score descending (earliest substring first)', () => {
    const rows = buildRows(items, 'sse');
    const labels = rows.map((r) => (r.kind === 'item' ? r.item.label : ''));
    // 'npm test sse watch' has 'sse' earliest? index: refactor sse(9), fix flaky sse(10), npm test sse(9)
    // earliest wins; ties keep source order.
    expect(labels[0]).toBe('refactor sse reconnect');
  });

  it('highlights the matched substring on filtered rows', () => {
    const first = buildRows(items, 'sse')[0];
    expect(first?.kind).toBe('item');
    if (first?.kind === 'item') expect(first.highlight).toEqual([9, 12]);
  });

  it('returns an empty array when nothing matches', () => {
    expect(buildRows(items, 'zzzzz')).toEqual([]);
  });

  it('keeps source order for equal scores (stable sort)', () => {
    const tie = [item('a', 'zzz sse'), item('b', 'yyy sse')]; // both substring at index 4
    const rows = buildRows(tie, 'sse');
    const labels = rows.map((r) => (r.kind === 'item' ? r.item.label : ''));
    expect(labels).toEqual(['zzz sse', 'yyy sse']);
  });

  it(`caps results at ${MAX_RESULTS_QUERY}`, () => {
    const many = Array.from({ length: 30 }, (_, i) => item('g', `sse candidate ${i}`));
    expect(buildRows(many, 'sse')).toHaveLength(MAX_RESULTS_QUERY);
  });

  it('reassigns contiguous selectable indices after filtering', () => {
    const rows = buildRows(items, 'sse');
    expect(rows.map((r) => (r.kind === 'item' ? r.index : -1))).toEqual([0, 1, 2]);
  });
});

describe('buildRows - session search (query-only, own group)', () => {
  const commands = [item('views', 'Chats'), item('actions', 'Settings')];
  const sess = (label: string, extra: Partial<PaletteItem> = {}): PaletteItem =>
    item('sessions', label, { href: `session:${label}`, ...extra });

  it('excludes sessions from the default (empty-query) list', () => {
    const rows = buildRows(commands, '', [sess('refactor sse')]);
    const labels = rows.map((r) => (r.kind === 'group' ? `#${r.label}` : r.item.label));
    expect(labels).not.toContain('refactor sse');
    expect(labels).not.toContain('#sessions');
  });

  it('surfaces matching sessions under a sessions header when querying', () => {
    const rows = buildRows(commands, 'sse', [
      sess('refactor sse reconnect'),
      sess('unrelated ops'),
    ]);
    const gi = rows.findIndex((r) => r.kind === 'group' && r.label === 'sessions');
    expect(gi).toBeGreaterThanOrEqual(0);
    const under = rows.slice(gi + 1).filter((r) => r.kind === 'item');
    expect(under.map((r) => (r.kind === 'item' ? r.item.label : ''))).toEqual([
      'refactor sse reconnect',
    ]);
  });

  it('emits no sessions header when nothing matches', () => {
    const rows = buildRows(commands, 'zzzzz', [sess('refactor sse')]);
    expect(rows.some((r) => r.kind === 'group' && r.label === 'sessions')).toBe(false);
  });

  it('ranks command matches before the sessions group', () => {
    const rows = buildRows(commands, 'chat', [sess('chatty session')]);
    const cmd = rows.findIndex((r) => r.kind === 'item' && r.item.label === 'Chats');
    const gi = rows.findIndex((r) => r.kind === 'group' && r.label === 'sessions');
    expect(cmd).toBeGreaterThanOrEqual(0);
    expect(cmd).toBeLessThan(gi);
  });

  it('assigns contiguous selectable indices across commands and sessions', () => {
    const rows = buildRows(commands, 'chat', [sess('chatty session')]);
    const idxs = rows
      .filter((r) => r.kind === 'item')
      .map((r) => (r.kind === 'item' ? r.index : -1));
    expect(idxs).toEqual([...idxs.keys()]);
  });

  it('caps session matches at the top 8 by recency', () => {
    const many = Array.from({ length: 20 }, (_, i) => sess(`sse candidate ${i}`));
    const rows = buildRows([], 'sse', many);
    expect(rows.filter((r) => r.kind === 'item')).toHaveLength(8);
  });

  it('preserves given session order on tied scores (live-first upstream)', () => {
    const rows = buildRows([], 'sse', [sess('zzz sse'), sess('yyy sse')]);
    const labels = rows
      .filter((r) => r.kind === 'item')
      .map((r) => (r.kind === 'item' ? r.item.label : ''));
    expect(labels).toEqual(['zzz sse', 'yyy sse']);
  });

  it('matches a session on its keywords (topic) with no highlight', () => {
    const rows = buildRows([], 'kubernetes', [
      sess('daily standup', { keywords: 'kubernetes rollout' }),
    ]);
    const row = rows.find((r) => r.kind === 'item');
    expect(row?.kind === 'item' && row.item.label).toBe('daily standup');
    expect(row?.kind === 'item' && row.highlight).toBeNull();
  });
});
