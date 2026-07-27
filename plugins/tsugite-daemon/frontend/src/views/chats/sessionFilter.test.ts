import { describe, expect, it } from 'vitest';
import {
  parseSessionFilter,
  sessionMatchesFilter,
  filterFreeText,
  type SessionFilterRow,
} from './sessionFilter';

const row = (over: Partial<SessionFilterRow> = {}): SessionFilterRow => ({
  id: 'sess-1',
  title: 'refactor sse backoff',
  topic: 'sse',
  label: 'Web: web-alice',
  agent: 'smoke',
  status: 'active',
  pinned: false,
  unread: false,
  isPrimary: false,
  needsYou: false,
  ...over,
});

describe('parseSessionFilter', () => {
  it('splits facet tokens from free text', () => {
    const f = parseSessionFilter('agent:smoke status:active reconnect');
    expect(f.agents).toEqual(['smoke']);
    expect(f.statuses).toEqual(['active']);
    expect(f.terms).toEqual(['reconnect']);
  });

  it('lowercases facet values but keeps the axis repeatable (OR within an axis)', () => {
    const f = parseSessionFilter('agent:Smoke agent:Ops status:Failed');
    expect(f.agents).toEqual(['smoke', 'ops']);
    expect(f.statuses).toEqual(['failed']);
  });

  it('collects is: flags and strips a leading # from a term', () => {
    const f = parseSessionFilter('is:pinned is:unread #sess-42 hello');
    expect(f.flags).toEqual(['pinned', 'unread']);
    expect(f.terms).toEqual(['sess-42', 'hello']);
  });

  it('is empty for blank / whitespace-only input', () => {
    const f = parseSessionFilter('   ');
    expect(f).toEqual({ agents: [], statuses: [], flags: [], terms: [] });
  });

  it('treats a bare colon or empty facet value as free text, not a facet', () => {
    const f = parseSessionFilter('status: :');
    expect(f.statuses).toEqual([]);
    // "status:" with no value and a lone ":" are meaningless facets -> free text
    expect(f.terms).toEqual(['status:', ':']);
  });
});

describe('sessionMatchesFilter', () => {
  it('ANDs across axes, ORs within an axis', () => {
    const f = parseSessionFilter('agent:smoke agent:ops status:active');
    expect(sessionMatchesFilter(row({ agent: 'smoke', status: 'active' }), f)).toBe(true);
    expect(sessionMatchesFilter(row({ agent: 'ops', status: 'active' }), f)).toBe(true);
    expect(sessionMatchesFilter(row({ agent: 'other', status: 'active' }), f)).toBe(false);
    expect(sessionMatchesFilter(row({ agent: 'smoke', status: 'completed' }), f)).toBe(false);
  });

  it('keeps finished sessions searchable by status: (completed/failed/cancelled)', () => {
    expect(
      sessionMatchesFilter(row({ status: 'completed' }), parseSessionFilter('status:completed')),
    ).toBe(true);
    expect(
      sessionMatchesFilter(row({ status: 'failed' }), parseSessionFilter('status:failed')),
    ).toBe(true);
    expect(
      sessionMatchesFilter(row({ status: 'active' }), parseSessionFilter('status:completed')),
    ).toBe(false);
  });

  it('matches every free-text term against the title/topic/label/id haystack', () => {
    const f = parseSessionFilter('reconnect backoff');
    expect(
      sessionMatchesFilter(row({ title: 'refactor sse backoff', topic: 'reconnect' }), f),
    ).toBe(true);
    expect(sessionMatchesFilter(row({ title: 'refactor sse', topic: 'x' }), f)).toBe(false);
  });

  it('honours is: flags (pinned/unread/primary/needs-you)', () => {
    expect(sessionMatchesFilter(row({ pinned: true }), parseSessionFilter('is:pinned'))).toBe(true);
    expect(sessionMatchesFilter(row({ pinned: false }), parseSessionFilter('is:pinned'))).toBe(
      false,
    );
    expect(sessionMatchesFilter(row({ needsYou: true }), parseSessionFilter('is:needs-you'))).toBe(
      true,
    );
    expect(sessionMatchesFilter(row({ isPrimary: true }), parseSessionFilter('is:primary'))).toBe(
      true,
    );
  });

  it('an empty filter matches everything', () => {
    expect(sessionMatchesFilter(row(), parseSessionFilter(''))).toBe(true);
  });
});

describe('filterFreeText', () => {
  it('rejoins just the free-text terms for the server ?q= merge', () => {
    expect(filterFreeText(parseSessionFilter('agent:smoke reconnect backoff'))).toBe(
      'reconnect backoff',
    );
    expect(filterFreeText(parseSessionFilter('agent:smoke status:active'))).toBe('');
  });
});
