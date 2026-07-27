/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { afterEach, expect, test, vi } from 'vitest';
import Drawer from './Drawer.svelte';

afterEach(cleanup);

const body = createRawSnippet(() => ({ render: () => `<div>detail body</div>` }));
const statusPill = createRawSnippet(() => ({
  render: () => `<span class="t-pill">awaiting</span>`,
}));

test('renders the title and status', async () => {
  await render(Drawer, { open: true, title: 'nightly backup', status: statusPill, children: body });
  await expect.element(page.getByRole('heading', { name: 'nightly backup' })).toBeInTheDocument();
  await expect.element(page.getByText('awaiting')).toBeInTheDocument();
});

test('the close button calls onclose', async () => {
  const onclose = vi.fn();
  await render(Drawer, { open: true, title: 'x', onclose, children: body });
  await userEvent.click(page.getByRole('button', { name: 'Close detail' }));
  expect(onclose).toHaveBeenCalledTimes(1);
});

test('Escape closes', async () => {
  const onclose = vi.fn();
  await render(Drawer, { open: true, title: 'x', onclose, children: body });
  const close = document.querySelector('.t-drawer .t-btn') as HTMLElement;
  close.focus();
  close.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  expect(onclose).toHaveBeenCalledTimes(1);
});

test('is inert while closed and interactive when open', async () => {
  const screen = await render(Drawer, { open: false, title: 'x', children: body });
  const aside = document.querySelector('.t-drawer') as HTMLElement;
  expect(aside.hasAttribute('inert')).toBe(true);
  await screen.rerender({ open: true });
  expect(aside.hasAttribute('inert')).toBe(false);
});

test('moves focus to the close button when opened', async () => {
  const screen = await render(Drawer, { open: false, title: 'x', children: body });
  await screen.rerender({ open: true });
  const close = document.querySelector('.t-drawer .t-btn') as HTMLElement;
  expect(document.activeElement).toBe(close);
});

test('a specimen that mounts already-open does not steal focus', async () => {
  const before = document.activeElement;
  await render(Drawer, { open: true, title: 'x', children: body });
  expect(document.activeElement).toBe(before);
});
