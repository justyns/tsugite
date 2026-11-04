import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import {
  compilePatterns,
  LinkPreviewStore,
  matchResolver,
  RETRY_MS,
  type LinkPreview,
} from './linkPreviews.svelte';

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

const PR = 'https://forge.example.test/octo-org/demo/pulls/7';
const PATTERNS = [
  { plugin: 'forge', name: 'pulls', hosts: ['forge.example.test'], path: '/pulls/' },
];
const PREVIEW: LinkPreview = {
  label: 'demo#7',
  title: 'Add retry',
  url: PR,
  status: null,
  badges: [],
  fields: [],
  icon: null,
};

function store(now = { t: 0 }) {
  return new LinkPreviewStore(
    () => PATTERNS,
    () => now.t,
  );
}

const settle = () => new Promise((resolve) => setTimeout(resolve));

describe('LinkPreviewStore.lookup', () => {
  it('asks the daemon about a matching link once and serves the answer', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue(PREVIEW as never);
    const s = store();

    expect(s.lookup(PR)).toBeUndefined();
    expect(s.lookup(PR)).toBeUndefined();
    await settle();

    expect(s.lookup(PR)).toEqual(PREVIEW);
    expect(s.version).toBe(1);
    expect(get).toHaveBeenCalledExactlyOnceWith(`/api/link-preview?url=${encodeURIComponent(PR)}`);
  });

  it('never asks about a link no resolver matches', () => {
    const get = vi.spyOn(api, 'get');

    expect(store().lookup('https://example.test/x')).toBeUndefined();
    expect(get).not.toHaveBeenCalled();
  });

  it('does not ask again about a link that failed', async () => {
    const get = vi.spyOn(api, 'get').mockRejectedValue(new Error('no preview'));
    const s = store();

    s.lookup(PR);
    await settle();
    expect(s.lookup(PR)).toBeUndefined();
    await settle();

    expect(get).toHaveBeenCalledOnce();
  });

  it('refreshes a stale preview and serves the old one meanwhile', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue(PREVIEW as never);
    const now = { t: 0 };
    const s = store(now);
    s.lookup(PR);
    await settle();

    now.t += 61_000;
    get.mockResolvedValue({ ...PREVIEW, label: 'demo#7 merged' } as never);
    expect(s.lookup(PR)?.label).toBe('demo#7');
    await settle();

    expect(s.lookup(PR)?.label).toBe('demo#7 merged');
    expect(get).toHaveBeenCalledTimes(2);
  });

  it('asks once more after a timeout, then keeps the link plain', async () => {
    vi.useFakeTimers();
    const timeout = Object.assign(new Error('preview timed out'), { status: 504 });
    const get = vi.spyOn(api, 'get').mockRejectedValue(timeout);
    const s = store();

    s.lookup(PR);
    await vi.advanceTimersByTimeAsync(0);
    s.lookup(PR);
    expect(get).toHaveBeenCalledOnce();

    get.mockResolvedValueOnce(PREVIEW as never);
    await vi.advanceTimersByTimeAsync(RETRY_MS);
    expect(s.version).toBe(1);
    s.lookup(PR);
    await vi.advanceTimersByTimeAsync(0);
    expect(s.lookup(PR)).toEqual(PREVIEW);

    const t = store();
    t.lookup(PR);
    await vi.advanceTimersByTimeAsync(RETRY_MS);
    t.lookup(PR);
    await vi.advanceTimersByTimeAsync(RETRY_MS);
    t.lookup(PR);
    expect(get).toHaveBeenCalledTimes(4);
  });
});

const FORGE = {
  plugin: 'forge',
  name: 'pulls',
  hosts: ['forge.example.test'],
  path: '/pulls/\\d+$',
};

describe('matchResolver', () => {
  const compiled = compilePatterns([FORGE, { ...FORGE, hosts: ['example.test'], path: null }]);

  it('matches a configured host and path', () => {
    expect(matchResolver(compiled, 'https://forge.example.test/octo-org/demo/pulls/7')).toBe(true);
    expect(matchResolver(compiled, 'https://FORGE.example.test/octo-org/demo/pulls/7')).toBe(true);
  });

  it('skips a path the pattern does not match, or another host', () => {
    expect(matchResolver(compiled, 'https://forge.example.test/octo-org/demo/issues/7')).toBe(
      false,
    );
    expect(matchResolver(compiled, 'https://other.example.test/octo-org/demo/pulls/7')).toBe(false);
  });

  it('matches any path on a host with no path pattern', () => {
    expect(matchResolver(compiled, 'http://example.test/anything')).toBe(true);
  });

  it('matches only absolute http(s) links', () => {
    expect(matchResolver(compiled, 'ftp://example.test/x')).toBe(false);
    expect(matchResolver(compiled, '/relative/pulls/7')).toBe(false);
    expect(matchResolver(compiled, 'not a url')).toBe(false);
  });
});

describe('compilePatterns', () => {
  it('drops a pattern JavaScript cannot compile and keeps the rest', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const compiled = compilePatterns([{ ...FORGE, path: '(?P<n>\\d+)' }, FORGE]);

    expect(compiled).toHaveLength(1);
    expect(warn).toHaveBeenCalledOnce();
  });
});
