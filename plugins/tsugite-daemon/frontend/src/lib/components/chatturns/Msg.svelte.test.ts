/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import { createRawSnippet } from 'svelte';
import Msg from './Msg.svelte';

const body = createRawSnippet(() => ({ render: () => '<p>hello world</p>' }));

test('a user turn exposes copy + edit-and-fork actions', async () => {
  const onEditFork = vi.fn();
  // pinnedActs keeps the hover toolbar visible so it is actionable in a headless run.
  render(Msg, {
    role: 'user',
    who: 'you',
    at: '14:22',
    children: body,
    onEditFork,
    pinnedActs: true,
  });

  await expect.element(page.getByRole('button', { name: 'Copy message' })).toBeInTheDocument();
  await page.getByRole('button', { name: 'Edit and fork from here' }).click();
  expect(onEditFork).toHaveBeenCalledTimes(1);
});

test('an ai turn exposes a retry action instead of edit-and-fork', async () => {
  const onRetry = vi.fn();
  render(Msg, {
    role: 'ai',
    who: 'tsugite',
    at: '14:23',
    children: body,
    onRetry,
    pinnedActs: true,
  });

  await expect
    .element(page.getByRole('button', { name: 'Edit and fork from here' }))
    .not.toBeInTheDocument();
  await page.getByRole('button', { name: 'Retry this response' }).click();
  expect(onRetry).toHaveBeenCalledTimes(1);
});

test('a failed ai turn shows a prominent Retry button (not hover-gated) that re-sends', async () => {
  const onRetry = vi.fn();
  // retryFailed marks the turn as errored; pinnedActs is NOT set, so the button
  // must be visible on its own - a failed turn cannot rely on hovering the exact
  // action-bar spot to recover.
  render(Msg, {
    role: 'ai',
    who: 'tsugite',
    at: '14:40',
    children: body,
    onRetry,
    retryFailed: true,
  });

  const retry = page.getByTestId('chat-retry');
  await expect.element(retry).toBeVisible();
  await expect.element(retry).toHaveTextContent('Retry');
  await retry.click();
  expect(onRetry).toHaveBeenCalledTimes(1);
});

test('a non-error ai turn keeps the hover-only retry icon, not the prominent button', async () => {
  render(Msg, {
    role: 'ai',
    who: 'tsugite',
    at: '14:41',
    children: body,
    onRetry: vi.fn(),
    pinnedActs: true,
  });
  // The regenerate affordance for a healthy turn stays the small action-bar icon.
  await expect
    .element(page.getByRole('button', { name: 'Retry this response' }))
    .toBeInTheDocument();
  expect(page.getByTestId('chat-retry').query()).toBeNull();
});

test('a streaming turn marks its body as a busy live region', async () => {
  const { container } = await render(Msg, {
    role: 'ai',
    who: 'tsugite',
    at: '14:31',
    children: body,
    streaming: true,
  });
  const bod = container.querySelector('.bod');
  expect(bod?.getAttribute('aria-busy')).toBe('true');
  expect(bod?.getAttribute('aria-live')).toBe('polite');
  expect(bod?.textContent).toContain('hello world');
});

test('a settled turn is not a live region', async () => {
  const { container } = await render(Msg, {
    role: 'ai',
    who: 'tsugite',
    at: '14:23',
    children: body,
  });
  const bod = container.querySelector('.bod');
  expect(bod?.hasAttribute('aria-busy')).toBe(false);
});
