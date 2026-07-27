/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { beforeEach, expect, test } from 'vitest';
import Toasts from './Toasts.svelte';
import { toasts } from './toast-store.svelte';

beforeEach(() => {
  toasts.items.length = 0;
});

test('renders the container as a status region with one Toast per queued item', async () => {
  toasts.push('ok', 'Job done', { body: '5/5 criteria passed', sticky: true });
  toasts.push('warn', 'Job needs an answer', { sticky: true });
  const { container } = await render(Toasts);
  expect(container.querySelector('.t-toasts')?.getAttribute('role')).toBe('status');
  expect(container.querySelectorAll('.t-toast')).toHaveLength(2);
  await expect.element(page.getByText('Job done')).toBeInTheDocument();
  await expect.element(page.getByText('Job needs an answer')).toBeInTheDocument();
});

test('dismissing one toast removes only that one from the stack', async () => {
  toasts.push('ok', 'first', { sticky: true });
  toasts.push('ok', 'second', { sticky: true });
  await render(Toasts);

  await userEvent.click(page.getByRole('button', { name: 'Dismiss' }).nth(0));

  await expect.element(page.getByText('first')).not.toBeInTheDocument();
  await expect.element(page.getByText('second')).toBeInTheDocument();
});

test('reflects the MAX_TOASTS cap - only the newest four render', async () => {
  for (let i = 0; i < 6; i++) toasts.push('info', `toast ${i}`, { sticky: true });
  const { container } = await render(Toasts);
  expect(container.querySelectorAll('.t-toast')).toHaveLength(4);
  await expect.element(page.getByText('toast 5')).toBeInTheDocument();
  await expect.element(page.getByText('toast 0')).not.toBeInTheDocument();
});
