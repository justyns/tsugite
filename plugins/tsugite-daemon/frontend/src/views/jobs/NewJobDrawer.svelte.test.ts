/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import NewJobDrawer from './NewJobDrawer.svelte';

const base = {
  open: true,
  agents: ['smoke', 'ops-runner'],
  executors: ['agent'],
  onClose: () => {},
  onSubmit: () => {},
};

test('the command preview reflects the live form state', async () => {
  await render(NewJobDrawer, base);
  await page.getByTestId('new-job-prompt').fill('backfill usage rollups');
  const preview = page.getByTestId('new-job-preview');
  await expect.element(preview).toHaveTextContent('/job "backfill usage rollups"');
  await expect.element(preview).toHaveTextContent('--agent smoke');
  await expect.element(preview).toHaveTextContent('--max-attempts 3');
  // default notify "needs you" maps to the backend's stuck token
  await expect.element(preview).toHaveTextContent('--notify-when stuck');
});

test('submit emits the cleaned form with notify mapped to a backend value', async () => {
  const onSubmit = vi.fn();
  await render(NewJobDrawer, { ...base, onSubmit });
  await page.getByTestId('new-job-prompt').fill('do the thing');
  await page.getByRole('textbox', { name: 'Criterion 1' }).fill('tests pass');
  await page.getByTestId('new-job-submit').click();
  expect(onSubmit).toHaveBeenCalledOnce();
  expect(onSubmit.mock.calls[0]![0]).toEqual({
    agent: 'smoke',
    prompt: 'do the thing',
    acceptanceCriteria: ['tests pass'],
    maxAttempts: 3,
    executor: 'agent',
    notifyWhen: 'stuck',
  });
});

test('submit is blocked until a prompt is entered', async () => {
  const onSubmit = vi.fn();
  await render(NewJobDrawer, { ...base, onSubmit });
  await page.getByTestId('new-job-submit').click({ force: true });
  expect(onSubmit).not.toHaveBeenCalled();
});

test('adding a criterion grows the list', async () => {
  await render(NewJobDrawer, base);
  expect(page.getByRole('textbox', { name: /^Criterion/ }).elements()).toHaveLength(1);
  await page.getByTestId('new-job-ac-add').click();
  expect(page.getByRole('textbox', { name: /^Criterion/ }).elements()).toHaveLength(2);
});

test('the executor picker is hidden when only the default executor exists', async () => {
  await render(NewJobDrawer, base);
  expect(page.getByTestId('new-job-executor').elements()).toHaveLength(0);
});

test('the executor picker appears once more than one executor is registered', async () => {
  await render(NewJobDrawer, { ...base, executors: ['agent', 'docker'] });
  await expect.element(page.getByTestId('new-job-executor')).toBeInTheDocument();
});
