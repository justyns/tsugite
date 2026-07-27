/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import LocalEcho from './LocalEcho.svelte';

test('renders the command line, its output, and the local-only footer', async () => {
  const { container } = await render(LocalEcho, {
    command: '/status',
    output: 'Model: claude_code:haiku\nContext: 0 / 128,000 tokens',
    ok: true,
  });
  expect(container.querySelector('.echo-cmd')?.textContent).toContain('/status');
  expect(container.querySelector('.echo-out')?.textContent).toContain('Model: claude_code:haiku');
  // Preserves newlines from the command output.
  expect(container.querySelector('.echo-out')?.textContent).toContain(
    'Context: 0 / 128,000 tokens',
  );
  // The muted footer marks the echo as ephemeral / not sent to the model.
  expect(container.textContent).toContain('not saved');
  expect(container.textContent).toContain('not sent to model');
});

test('an error echo renders the message as an alert', async () => {
  const { container } = await render(LocalEcho, {
    command: '/model bogus',
    output: 'unknown provider: bogus',
    ok: false,
  });
  await expect.element(page.getByRole('alert')).toBeInTheDocument();
  expect(page.getByRole('alert').element().textContent).toContain('unknown provider: bogus');
  expect(container.querySelector('.is-err')).not.toBeNull();
});

test('empty output shows a minimal acknowledgment, never an empty block', async () => {
  const { container } = await render(LocalEcho, { command: '/compact', output: '', ok: true });
  expect(container.querySelector('.echo-cmd')?.textContent).toContain('/compact');
  // Not an empty gutter block: a "done" acknowledgment stands in for no output.
  expect(container.querySelector('.echo-out')?.textContent?.trim()).toBeTruthy();
  expect(container.textContent).toContain('done');
});

test('an action affordance renders as a link (the /job open-jobs button)', async () => {
  render(LocalEcho, {
    command: '/job build the thing',
    output: 'Job started',
    ok: true,
    action: { label: 'Open jobs', href: '#jobs' },
  });
  const link = page.getByRole('link', { name: 'Open jobs' });
  await expect.element(link).toBeInTheDocument();
  await expect.element(link).toHaveAttribute('href', '#jobs');
});
