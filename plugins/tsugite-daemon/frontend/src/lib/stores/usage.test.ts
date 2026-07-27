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
