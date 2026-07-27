/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, beforeEach } from 'vitest';
import JobTile from './JobTile.svelte';
import { jobs, type Job } from '$lib/stores/jobs.svelte';
import { jobDrawerRequest } from '../jobs/jobDrawerSignal.svelte';

const storeJob = (over: Partial<Job>): Job =>
  ({ job_id: 'j', state: 'queued', ...over }) as unknown as Job;

beforeEach(() => {
  jobs.jobs = [];
  jobDrawerRequest.consume();
});

test('shows the live store state, not the spawn-time recorded snapshot', async () => {
  // The recorded block froze at "running"; the live store says the job finished.
  jobs.jobs = [storeJob({ job_id: 'j1', state: 'done' })];
  render(JobTile, { job: { job_id: 'j1', state: 'running', prompt: 'do x' } });
  await expect.element(page.getByText('done')).toBeInTheDocument();
  expect(page.getByText('running').query()).toBeNull();
});

test('falls back to the recorded state when the store has no such job', async () => {
  render(JobTile, { job: { job_id: 'j2', state: 'running', prompt: 'y' } });
  await expect.element(page.getByText('running')).toBeInTheDocument();
});

test('clicking open requests this job’s drawer', async () => {
  render(JobTile, { job: { job_id: 'j3', state: 'running' } });
  await page.getByRole('link', { name: 'Open in Jobs' }).click();
  expect(jobDrawerRequest.pending).toBe('j3');
});
