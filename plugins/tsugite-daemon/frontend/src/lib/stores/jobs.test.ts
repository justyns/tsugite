import { describe, expect, test } from 'vitest';
import { JobsStore, type Job } from './jobs.svelte';

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
  test('filtered applies the filter grammar over the list', () => {
    const store = new JobsStore();
    store.jobs = [
      job('job-1', { state: 'running', agent: 'odyn' }),
      job('job-2', { state: 'done', agent: 'scout' }),
    ];
    store.filterText = 'state:running';
    expect(store.filtered.map((j) => j.job_id)).toEqual(['job-1']);
  });

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
