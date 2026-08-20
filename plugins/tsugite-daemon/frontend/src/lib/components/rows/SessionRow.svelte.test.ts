/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import SessionRow from './SessionRow.svelte';

test('renders title, relative time, source tag, and description', async () => {
  render(SessionRow, {
    title: 'refactor: sse reconnect backoff',
    when: 'now',
    description: 'streaming a reply',
    state: 'running',
    sourceType: 'code',
  });
  await expect.element(page.getByText('refactor: sse reconnect backoff')).toBeInTheDocument();
  await expect.element(page.getByText('now')).toBeInTheDocument();
  await expect.element(page.getByText('code')).toBeInTheDocument();
  await expect.element(page.getByText('streaming a reply')).toBeInTheDocument();
});

test('a needs-you row carries the state in its accessible name, not just color', async () => {
  render(SessionRow, {
    title: 'ops: nightly backup failing on prune',
    when: '12m',
    description: 'job blocked on a retention question',
    state: 'needs-you',
    sourceType: 'ops',
  });
  await expect
    .element(
      page.getByRole('button', {
        name: 'ops: nightly backup failing on prune, awaiting your input',
      }),
    )
    .toBeInTheDocument();
});

test('the active row is marked aria-current for screen readers', async () => {
  render(SessionRow, {
    title: 'refactor: sse reconnect backoff',
    when: 'now',
    state: 'running',
    sourceType: 'code',
    isActive: true,
  });
  await expect
    .element(page.getByRole('button', { name: /refactor: sse reconnect backoff/ }))
    .toHaveAttribute('aria-current', 'true');
});

test('clicking the row fires onSelect', async () => {
  const onSelect = vi.fn();
  render(SessionRow, {
    title: 'chat: naming things',
    when: '1d',
    state: 'idle',
    sourceType: 'chat',
    onSelect,
  });
  await userEvent.click(page.getByRole('button', { name: /chat: naming things/ }));
  expect(onSelect).toHaveBeenCalledOnce();
});

test('Enter and Space activate the focused row from the keyboard', async () => {
  const onSelect = vi.fn();
  render(SessionRow, {
    title: 'chat: naming things',
    when: '1d',
    state: 'idle',
    sourceType: 'chat',
    onSelect,
  });
  await userEvent.click(page.getByRole('button', { name: /chat: naming things/ }));
  onSelect.mockClear();
  await userEvent.keyboard('{Enter}');
  expect(onSelect).toHaveBeenCalledOnce();
  await userEvent.keyboard(' ');
  expect(onSelect).toHaveBeenCalledTimes(2);
});

test('an active-job count renders as a badge with an accessible count', async () => {
  render(SessionRow, {
    title: 'refactor: sse reconnect backoff',
    when: 'now',
    state: 'running',
    sourceType: 'code',
    activeJobCount: 1,
  });
  await expect.element(page.getByText('1▸')).toBeInTheDocument();
  await expect.element(page.getByLabelText('1 active job')).toBeInTheDocument();
});

test('an unread idle row shows the accent dot instead of the idle glyph, and bolds the title', async () => {
  const { container } = await render(SessionRow, {
    title: 'research: local whisper models',
    when: '2h',
    state: 'idle',
    sourceType: 'research',
    isUnread: true,
  });
  expect(container.querySelector('.t-srow')!.className).toContain('is-unread');
  expect(container.querySelector('.ind .t-dot')).not.toBeNull();
});

test('a running-but-unread row keeps the ambient spinner in the indicator slot', async () => {
  const { container } = await render(SessionRow, {
    title: 'refactor: sse reconnect backoff',
    when: 'now',
    state: 'running',
    sourceType: 'code',
    isUnread: true,
  });
  expect(container.querySelector('.ind .t-spin')).not.toBeNull();
  expect(container.querySelector('.ind .t-dot')).toBeNull();
});

test('a waiting-on count renders in the marker cluster with an accessible label', async () => {
  render(SessionRow, {
    title: 'ops: quarterly rollup',
    when: '4m',
    state: 'idle',
    sourceType: 'ops',
    waitingOnCount: 2,
  });
  await expect.element(page.getByLabelText('waiting on 2 sessions')).toBeInTheDocument();
  await expect.element(page.getByText('2', { exact: true })).toBeInTheDocument();
});

test('the waiting-on label is singular for one session', async () => {
  render(SessionRow, {
    title: 'ops: quarterly rollup',
    when: '4m',
    state: 'idle',
    sourceType: 'ops',
    waitingOnCount: 1,
  });
  await expect.element(page.getByLabelText('waiting on 1 session')).toBeInTheDocument();
});
