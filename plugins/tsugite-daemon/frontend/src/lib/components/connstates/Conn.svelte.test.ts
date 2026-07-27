/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import Conn from './Conn.svelte';

test('announces state via role=status', async () => {
  const { container } = await render(Conn, { state: 'on' });
  expect(container.querySelector('[role="status"]')).toBeTruthy();
});

test('reconnect attempt count is visible while reconnecting', async () => {
  await render(Conn, { state: 're', reconnectAttempt: 2 });
  await expect.element(page.getByText('(2)')).toBeVisible();
});

test('reconnect attempt count is hidden (CSS) once connected again', async () => {
  await render(Conn, { state: 'on', reconnectAttempt: 2 });
  // still in the DOM (all three labels always are, per the CSS
  // switch), but display:none under data-st="on" - not visible to the user.
  await expect.element(page.getByText('(2)')).not.toBeVisible();
});

test('reconnect attempt count is hidden (CSS) once given up', async () => {
  await render(Conn, { state: 'off', reconnectAttempt: 2 });
  await expect.element(page.getByText('(2)')).not.toBeVisible();
});

test('retry now is reachable only when offline and a handler is provided', async () => {
  const onRetry = vi.fn();
  await render(Conn, { state: 'off', onRetry });
  const retry = page.getByRole('button', { name: 'retry now' });
  await expect.element(retry).toBeInTheDocument();
  await retry.click();
  expect(onRetry).toHaveBeenCalledTimes(1);
});

test('no retry button when onRetry is not provided', async () => {
  const { container } = await render(Conn, { state: 'off' });
  expect(container.querySelector('button')).toBeNull();
});

test('retry button is not exposed to the accessibility tree while merely reconnecting', async () => {
  const onRetry = vi.fn();
  await render(Conn, { state: 're', onRetry });
  // display:none removes it from the a11y tree entirely, so the role query
  // itself must fail to resolve - a stronger guarantee than "not visible".
  await expect.element(page.getByRole('button', { name: 'retry now' })).not.toBeInTheDocument();
});
