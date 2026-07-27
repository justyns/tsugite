/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import ExecBlock from './ExecBlock.svelte';

test('header toggles the output on click', async () => {
  render(ExecBlock, { command: 'echo hello', status: 'done', exitCode: 0, output: 'hello' });
  const header = page.getByRole('button', { name: /echo hello/ });
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');
  await header.click();
  await expect.element(header).toHaveAttribute('aria-expanded', 'true');
  await header.click();
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');
});

test('header toggles via Enter and Space (role=button keyboard path)', async () => {
  render(ExecBlock, { command: 'echo hello', status: 'done', exitCode: 0, output: 'hello' });
  const header = page.getByRole('button', { name: /echo hello/ });
  (header.element() as HTMLElement).focus();

  await userEvent.keyboard('{Enter}');
  await expect.element(header).toHaveAttribute('aria-expanded', 'true');

  await userEvent.keyboard(' ');
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');
});

test('renders native tool-call arguments in the disclosure body', async () => {
  render(ExecBlock, {
    command: 'session_metadata',
    status: 'done',
    args: { key: 'topic', value: 'ledger totals' },
    output: 'metadata updated',
    open: true,
  });
  await expect.element(page.getByText('topic', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('ledger totals')).toBeInTheDocument();
  await expect.element(page.getByText('metadata updated')).toBeInTheDocument();
});

test('the title bar previews the primary argument at a glance, truncated', async () => {
  const { container } = await render(ExecBlock, {
    command: 'read_file',
    status: 'done',
    args: { path: 'ops/alpha.md', start_line: 1 },
  });
  expect(container.querySelector('.argspv')?.textContent).toBe('ops/alpha.md +1');

  await render(ExecBlock, {
    command: 'run',
    status: 'done',
    args: { cmd: 'x'.repeat(200) },
  });
  const previews = container.ownerDocument.querySelectorAll('.argspv');
  const long = previews[previews.length - 1]!.textContent!;
  expect(long.length).toBeLessThanOrEqual(80);
  expect(long.endsWith('…')).toBe(true);
});

test('an ended block shows neither a spinner nor an exit code (closed-neutral)', async () => {
  render(ExecBlock, { command: 'slow_tool', status: 'ended' });
  await expect.element(page.getByText('running')).not.toBeInTheDocument();
  await expect.element(page.getByText(/exit \d/)).not.toBeInTheDocument();
  await expect.element(page.getByText('ended')).toBeInTheDocument();
});

test('auto-collapses when a running block finishes', async () => {
  const { rerender } = await render(ExecBlock, {
    command: 'npm test',
    status: 'running',
    output: 'watching…',
    open: true,
  });
  const header = page.getByRole('button', { name: /npm test/ });
  await expect.element(header).toHaveAttribute('aria-expanded', 'true');

  await rerender({ status: 'done', exitCode: 0, open: false });
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');
});

test('a manual toggle wins until the open prop next changes', async () => {
  const { rerender } = await render(ExecBlock, {
    command: 'npm test',
    status: 'running',
    output: 'watching…',
    open: true,
  });
  const header = page.getByRole('button', { name: /npm test/ });
  await header.click();
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');

  await rerender({ status: 'done', exitCode: 0, open: false });
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');
  await header.click();
  await expect.element(header).toHaveAttribute('aria-expanded', 'true');
});

test('the open affordance fires its callback without toggling the header', async () => {
  const onOpenExternal = vi.fn();
  render(ExecBlock, {
    command: 'npm test',
    status: 'running',
    meta: '0:07',
    output: 'watching…',
    onOpenExternal,
  });
  const header = page.getByRole('button', { name: /npm test/ });
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');

  await page.getByRole('button', { name: 'Open in Terminals' }).click();
  expect(onOpenExternal).toHaveBeenCalledTimes(1);
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');
});
