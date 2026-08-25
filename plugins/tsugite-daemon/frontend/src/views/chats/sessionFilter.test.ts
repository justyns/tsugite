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
  status: 'active',
  pinned: false,
  unread: false,
  isPrimary: false,
  needsYou: false,
  ...over,
});

describe('parseSessionFilter', () => {
  it('splits facet tokens from free text', () => {
    const f = parseSessionFilter('status:active reconnect');
    expect(f.statuses).toEqual(['active']);
    expect(f.terms).toEqual(['reconnect']);
  });

  it('lowercases facet values but keeps the axis repeatable (OR within an axis)', () => {
    const f = parseSessionFilter('status:Failed status:Active');
    expect(f.statuses).toEqual(['failed', 'active']);
  });

  it('collects is: flags and strips a leading # from a term', () => {
    const f = parseSessionFilter('is:pinned is:unread #sess-42 hello');
    expect(f.flags).toEqual(['pinned', 'unread']);
    expect(f.terms).toEqual(['sess-42', 'hello']);
  });

  it('is empty for blank / whitespace-only input', () => {
    const f = parseSessionFilter('   ');
    expect(f).toEqual({ statuses: [], flags: [], terms: [] });
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
    const f = parseSessionFilter('status:active status:running is:pinned');
    expect(sessionMatchesFilter(row({ status: 'active', pinned: true }), f)).toBe(true);
    expect(sessionMatchesFilter(row({ status: 'running', pinned: true }), f)).toBe(true);
    expect(sessionMatchesFilter(row({ status: 'active', pinned: false }), f)).toBe(false);
    expect(sessionMatchesFilter(row({ status: 'completed', pinned: true }), f)).toBe(false);
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
    expect(filterFreeText(parseSessionFilter('status:active reconnect backoff'))).toBe(
      'reconnect backoff',
    );
    expect(filterFreeText(parseSessionFilter('is:pinned status:active'))).toBe('');
  });
});
