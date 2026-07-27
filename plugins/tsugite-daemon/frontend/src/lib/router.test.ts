import { describe, expect, it } from 'vitest';
import { buildHash, parseHash } from '$lib/router.svelte';
import type { Route } from '$lib/router.svelte';

describe('parseHash', () => {
  it('treats an empty hash as the empty view', () => {
    expect(parseHash('')).toEqual({ view: '', params: {} });
    expect(parseHash('#')).toEqual({ view: '', params: {} });
  });

  it('reads a bare view with or without the leading #', () => {
    expect(parseHash('#chats')).toEqual({ view: 'chats', params: {} });
    expect(parseHash('chats')).toEqual({ view: 'chats', params: {} });
  });

  it('parses query-ish params', () => {
    expect(parseHash('#chats?session=abc')).toEqual({
      view: 'chats',
      params: { session: 'abc' },
    });
  });

  it('parses multiple params', () => {
    expect(parseHash('#jobs?state=stuck&id=7')).toEqual({
      view: 'jobs',
      params: { state: 'stuck', id: '7' },
    });
  });

  it('url-decodes param values', () => {
    expect(parseHash('#files?path=a%2Fb.txt').params.path).toBe('a/b.txt');
  });
});

describe('buildHash', () => {
  it('serializes a bare view', () => {
    expect(buildHash('chats')).toBe('#chats');
  });

  it('omits an empty query', () => {
    expect(buildHash('chats', {})).toBe('#chats');
  });

  it('serializes params', () => {
    expect(buildHash('chats', { session: 'abc' })).toBe('#chats?session=abc');
  });

  it('url-encodes param values', () => {
    expect(buildHash('files', { path: 'a/b.txt' })).toBe('#files?path=a%2Fb.txt');
  });
});

describe('round trip', () => {
  it('parseHash(buildHash(...)) is identity', () => {
    const cases: Route[] = [
      { view: 'chats', params: {} },
      { view: 'jobs', params: { state: 'stuck' } },
      { view: 'files', params: { path: 'a/b.txt', q: 'hello world' } },
    ];
    for (const route of cases) {
      expect(parseHash(buildHash(route.view, route.params))).toEqual(route);
    }
  });
});
