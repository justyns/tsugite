import { afterEach, describe, expect, test, vi } from 'vitest';
import { api } from '$lib/api/client';
import { daysAgoISO, UsageStore } from './usage.svelte';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('daysAgoISO', () => {
  test('subtracts whole UTC days and returns a bare ISO date', () => {
    const from = new Date('2026-07-14T08:30:00Z');
    expect(daysAgoISO(0, from)).toBe('2026-07-14');
    expect(daysAgoISO(30, from)).toBe('2026-06-14');
  });

  test('crosses a UTC month/year boundary correctly', () => {
    expect(daysAgoISO(5, new Date('2026-01-02T00:00:00Z'))).toBe('2025-12-28');
  });
});

describe('UsageStore.loadToday', () => {
  test('fetches /api/usage/total with a since=today (UTC) date and stores it independently of `total`', async () => {
    const store = new UsageStore();
    const spy = vi.spyOn(api, 'get').mockResolvedValue({
      runs: 4,
      total_tokens: 86754,
      total_cost: 2.14,
      input_tokens: 1,
      output_tokens: 1,
    });

    await store.loadToday();

    const todayIso = daysAgoISO(0);
    expect(spy).toHaveBeenCalledWith(`/api/usage/total?since=${todayIso}`);
    expect(store.today).toEqual({
      runs: 4,
      total_tokens: 86754,
      total_cost: 2.14,
      input_tokens: 1,
      output_tokens: 1,
    });
    // The range-scoped dashboard field is untouched by this call.
    expect(store.total).toBeNull();
  });

  test('a failed fetch is best-effort: it does not throw and leaves the previous value in place', async () => {
    const store = new UsageStore();
    store.today = {
      runs: 1,
      total_tokens: 1,
      total_cost: 1,
      input_tokens: 1,
      output_tokens: 1,
      cache_creation_tokens: 0,
      cache_read_tokens: 0,
    };
    vi.spyOn(api, 'get').mockRejectedValue(new Error('network down'));

    await expect(store.loadToday()).resolves.toBeUndefined();
    expect(store.today).toEqual({
      runs: 1,
      total_tokens: 1,
      total_cost: 1,
      input_tokens: 1,
      output_tokens: 1,
      cache_creation_tokens: 0,
      cache_read_tokens: 0,
    });
  });

  test('does not touch loading/error - those belong to the dashboard range load', async () => {
    const store = new UsageStore();
    vi.spyOn(api, 'get').mockResolvedValue({
      runs: 0,
      total_tokens: 0,
      total_cost: 0,
      input_tokens: 0,
      output_tokens: 0,
    });

    await store.loadToday();

    expect(store.loading).toBe(false);
    expect(store.error).toBeNull();
  });
});

describe('UsageStore.load request ordering', () => {
  test('a slow earlier range does not overwrite the newest one', async () => {
    const store = new UsageStore();
    const pending: { path: string; resolve: (value: unknown) => void }[] = [];
    vi.spyOn(api, 'get').mockImplementation(((path: string) => {
      if (path.startsWith('/api/usage/providers')) return Promise.resolve([]);
      return new Promise((resolve) => {
        pending.push({ path, resolve });
      });
    }) as never);

    function settle(since: string, runs: number): void {
      for (const req of pending.filter((p) => p.path.includes(`since=${since}`))) {
        req.resolve(
          req.path.startsWith('/api/usage/total')
            ? { runs, total_tokens: runs, total_cost: 0, input_tokens: 0, output_tokens: 0 }
            : [{ period: since, runs }],
        );
      }
    }

    const slow = store.load({ sinceDays: 90 });
    const fresh = store.load({ sinceDays: 7 });

    settle(daysAgoISO(7), 7);
    await fresh;
    expect(store.summary[0]!.period).toBe(daysAgoISO(7));

    settle(daysAgoISO(90), 90);
    await slow;

    expect(store.summary[0]!.period).toBe(daysAgoISO(7));
    expect(store.total!.runs).toBe(7);
    expect(store.range.sinceDays).toBe(7);
  });

  test('an earlier range that fails does not post its error over the newest one', async () => {
    const store = new UsageStore();
    const pending: { path: string; reject: (reason: unknown) => void }[] = [];
    vi.spyOn(api, 'get').mockImplementation(((path: string) => {
      if (path.startsWith('/api/usage/providers')) return Promise.resolve([]);
      if (path.includes(`since=${daysAgoISO(7)}`)) return Promise.resolve([]);
      return new Promise((_resolve, reject) => {
        pending.push({ path, reject });
      });
    }) as never);

    const slow = store.load({ sinceDays: 90 });
    await store.load({ sinceDays: 7 });

    for (const req of pending) req.reject(new Error('90-day read failed'));
    await slow;

    expect(store.error).toBeNull();
    expect(store.loading).toBe(false);
  });
});
