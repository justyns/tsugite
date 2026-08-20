import { afterEach, describe, expect, test, vi } from 'vitest';
import { api } from '$lib/api/client';
import { JobsStore, type Job } from './jobs.svelte';

afterEach(() => vi.restoreAllMocks());

function job(id: string, extra: Partial<Job> = {}): Job {
  return { job_id: id, state: 'running', agent: 'odyn', prompt: '', ...extra } as unknown as Job;
}

describe('JobsStore.applyJobUpdate', () => {
  test('prepends an unseen job as the newest', () => {
    const store = new JobsStore();
    store.jobs = [job('job-1')];
    store.applyJobUpdate(job('job-2') as unknown as Record<string, unknown>);
    expect(store.jobs.map((j) => j.job_id)).toEqual(['job-2', 'job-1']);
  });

  test('replaces an existing job in place', () => {
    const store = new JobsStore();
    store.jobs = [job('job-1', { state: 'running' }), job('job-2')];
    store.applyJobUpdate(job('job-1', { state: 'done' }) as unknown as Record<string, unknown>);
    expect(store.jobs.map((j) => j.job_id)).toEqual(['job-1', 'job-2']);
    expect(store.jobs[0]!.state).toBe('done');
  });

  test('ignores a payload with no job id', () => {
    const store = new JobsStore();
    store.jobs = [job('job-1')];
    store.applyJobUpdate({ state: 'running' });
    expect(store.jobs).toHaveLength(1);
  });
});

describe('JobsStore derived views', () => {
  test('counts groups the board buckets', () => {
    const store = new JobsStore();
    store.jobs = [
      job('a', { state: 'running' }),
      job('b', { state: 'errored' }),
      job('c', { state: 'done' }),
    ];
    expect(store.counts).toEqual({ stuck: 1, active: 1, resolved: 1 });
  });
});

describe('JobsStore.load', () => {
  test('a slow earlier load cannot overwrite the result of a newer one', async () => {
    // The shell seeds the open jobs at boot while the Jobs board asks for the
    // full list, so two loads can be in flight at once.
    const store = new JobsStore();
    let finishSeed: (value: { jobs: Job[] }) => void = () => {};
    vi.spyOn(api, 'get')
      .mockImplementationOnce(() => new Promise((resolve) => (finishSeed = resolve)))
      .mockResolvedValueOnce({ jobs: [job('full')] });

    const seed = store.load({ state: 'open' });
    await store.load();
    finishSeed({ jobs: [job('seed')] });
    await seed;

    expect(store.jobs.map((j) => j.job_id)).toEqual(['full']);
    expect(store.loading).toBe(false);
  });
});
