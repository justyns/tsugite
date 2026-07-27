/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import JobDrawer from './JobDrawer.svelte';
import type { Job } from '$lib/stores/jobs.svelte';

function job(partial: Partial<Job>): Job {
  return {
    job_id: 'job-1',
    parent_session_id: 'sess-parent',
    worker_session_id: null,
    verifier_session_id: null,
    state: 'stuck',
    prompt: 'rename agent frontmatter keys',
    verify_attempts: 1,
    max_attempts: 3,
    notify_when: 'stuck',
    error: null,
    error_detail: null,
    pending_question: null,
    attempts: [],
    acceptance_criteria: [],
    ac_results: [],
    result: null,
    agent: 'code-worker',
    model: 'sonnet-4.6',
    effort: null,
    model_ladder: null,
    ladder_index: null,
    verifier_model: null,
    repo: null,
    created_at: '2026-07-14T10:00:00Z',
    updated_at: '2026-07-14T11:00:00Z',
    resolved_at: null,
    spawned_by: 'user',
    executor: 'agent',
    worker_terminal_id: null,
    ...partial,
  } as Job;
}

const noop = {
  onClose: () => {},
  onRetry: () => {},
  onCancel: () => {},
  onMarkDone: () => {},
  onOpenChat: () => {},
  onOpenTerminal: () => {},
};

test('a stuck job offers retry, mark-done and cancel, each wired to its callback', async () => {
  const onRetry = vi.fn();
  const onMarkDone = vi.fn();
  const onCancel = vi.fn();
  await render(JobDrawer, {
    job: job({ state: 'stuck' }),
    now: Date.now(),
    ...noop,
    onRetry,
    onMarkDone,
    onCancel,
  });

  await page.getByTestId('job-retry').click();
  await page.getByTestId('job-mark-done').click();
  await page.getByTestId('job-cancel').click();
  expect(onRetry).toHaveBeenCalledOnce();
  expect(onMarkDone).toHaveBeenCalledOnce();
  expect(onCancel).toHaveBeenCalledOnce();
});

test('an errored job offers retry + cancel but not mark-done', async () => {
  await render(JobDrawer, {
    job: job({ state: 'errored', error: 'exit 1' }),
    now: Date.now(),
    ...noop,
  });
  await expect.element(page.getByTestId('job-retry')).toBeInTheDocument();
  await expect.element(page.getByTestId('job-cancel')).toBeInTheDocument();
  expect(page.getByTestId('job-mark-done').elements()).toHaveLength(0);
});

test('a resolved (done) job is read-only: no retry / cancel / mark-done', async () => {
  await render(JobDrawer, {
    job: job({ state: 'done', resolved_at: '2026-07-14T11:30:00Z' }),
    now: Date.now(),
    ...noop,
  });
  expect(page.getByTestId('job-retry').elements()).toHaveLength(0);
  expect(page.getByTestId('job-cancel').elements()).toHaveLength(0);
  expect(page.getByTestId('job-mark-done').elements()).toHaveLength(0);
});

test('the acceptance-criteria checklist shows pass/fail with the verifier reason', async () => {
  await render(JobDrawer, {
    job: job({
      state: 'awaiting_input',
      acceptance_criteria: ['backup completes', 'disk below 80%'],
      ac_results: [
        { ac_index: 0, ac_text: 'backup completes', pass: true, reason: '', attempt: 1 },
        {
          ac_index: 1,
          ac_text: 'disk below 80%',
          pass: false,
          reason: '84% after prune',
          attempt: 1,
        },
      ],
      pending_question: 'drop weeklies older than 90d?',
    }),
    now: Date.now(),
    ...noop,
  });
  await expect.element(page.getByText('acceptance criteria · 1/2')).toBeInTheDocument();
  await expect.element(page.getByText('84% after prune')).toBeInTheDocument();
});

test('an awaiting job routes answering to the parent chat surface', async () => {
  const onOpenChat = vi.fn();
  await render(JobDrawer, {
    job: job({
      state: 'awaiting_input',
      pending_question: 'which retention?',
      parent_session_id: 'sess-parent',
    }),
    now: Date.now(),
    ...noop,
    onOpenChat,
  });
  await expect.element(page.getByText('which retention?')).toBeInTheDocument();
  await page.getByTestId('job-link-chat').click();
  expect(onOpenChat).toHaveBeenCalledWith('sess-parent');
});

test('worker and verifier session links each route to their own chat; parent stays parent', async () => {
  const onOpenChat = vi.fn();
  await render(JobDrawer, {
    job: job({
      state: 'running',
      parent_session_id: 'sess-parent',
      worker_session_id: 'sess-worker',
      verifier_session_id: 'sess-verifier',
    }),
    now: Date.now(),
    ...noop,
    onOpenChat,
  });
  await page.getByTestId('job-link-chat').click();
  expect(onOpenChat).toHaveBeenLastCalledWith('sess-parent');
  await page.getByTestId('job-link-worker').click();
  expect(onOpenChat).toHaveBeenLastCalledWith('sess-worker');
  await page.getByTestId('job-link-verifier').click();
  expect(onOpenChat).toHaveBeenLastCalledWith('sess-verifier');
});

test('worker and verifier links are absent when their session ids are null', async () => {
  await render(JobDrawer, {
    job: job({ state: 'stuck' }),
    now: Date.now(),
    ...noop,
  });
  await expect.element(page.getByTestId('job-link-chat')).toBeInTheDocument();
  expect(page.getByTestId('job-link-worker').elements()).toHaveLength(0);
  expect(page.getByTestId('job-link-verifier').elements()).toHaveLength(0);
});

test('the worker-pty link appears only when a terminal id is resolved', async () => {
  const onOpenTerminal = vi.fn();
  const { rerender } = await render(JobDrawer, {
    job: job({ state: 'running' }),
    now: Date.now(),
    terminalId: null,
    ...noop,
    onOpenTerminal,
  });
  expect(page.getByTestId('job-link-terminal').elements()).toHaveLength(0);
  await rerender({
    job: job({ state: 'running' }),
    now: Date.now(),
    terminalId: 'term-9',
    ...noop,
    onOpenTerminal,
  });
  await page.getByTestId('job-link-terminal').click();
  expect(onOpenTerminal).toHaveBeenCalledWith('term-9');
});
