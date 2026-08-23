import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(async () => ({ entries: [] })) },
}));

import { api } from '$lib/api/client';
import { ActivityStore } from './activity.svelte';

const get = vi.mocked(api.get);

beforeEach(() => {
  vi.useFakeTimers();
  get.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('ActivityStore.load', () => {
  it('sends the type filter', async () => {
    await new ActivityStore().load({ types: 'job' });
    expect(get).toHaveBeenCalledWith('/api/activity?types=job');
  });

  it('asks for the unfiltered feed when no options are given', async () => {
    await new ActivityStore().load();
    expect(get).toHaveBeenCalledWith('/api/activity');
  });

  it('records a failed fetch instead of throwing', async () => {
    get.mockRejectedValueOnce(new Error('daemon down'));
    const store = new ActivityStore();
    await store.load();
    expect(store.error).toBe('daemon down');
    expect(store.loading).toBe(false);
  });
});

describe('ActivityStore.applyEvent', () => {
  it('coalesces a burst of broadcasts into one revision bump', () => {
    const store = new ActivityStore();
    store.applyEvent({ type: 'job_update' });
    store.applyEvent({ type: 'schedule_update' });
    store.applyEvent({ type: 'history_update' });
    expect(store.rev).toBe(0);
    vi.advanceTimersByTime(500);
    expect(store.rev).toBe(1);
  });

  it('only bumps for a session_update that ended a run', () => {
    const store = new ActivityStore();
    store.applyEvent({ type: 'session_update', data: { action: 'busy', id: 's1' } });
    vi.advanceTimersByTime(500);
    expect(store.rev).toBe(0);

    store.applyEvent({ type: 'session_update', data: { action: 'completed', id: 's1' } });
    vi.advanceTimersByTime(500);
    expect(store.rev).toBe(1);
  });

  it('ignores broadcasts that cannot add a feed row', () => {
    const store = new ActivityStore();
    store.applyEvent({ type: 'terminal_state' });
    store.applyEvent({ type: 'compaction_started' });
    vi.advanceTimersByTime(500);
    expect(store.rev).toBe(0);
  });
});
