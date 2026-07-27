/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import KeyStrip from './KeyStrip.svelte';
import { conn } from '$lib/stores/conn.svelte';

afterEach(() => {
  conn.status = 'connecting';
});

test('the settings trigger fires its callback', async () => {
  const onOpenSettings = vi.fn();
  // The trigger is width:100% (it fills the nav rail); the standalone test mount
  // has no width, so a real click fires the handler without a layout dependency.
  const { container } = await render(KeyStrip, { onOpenSettings });
  const trigger = container.querySelector<HTMLButtonElement>('[data-testid="settings-trigger"]');
  trigger?.click();
  expect(onOpenSettings).toHaveBeenCalledOnce();
});

test('usage placeholders are overridden by props when data arrives', async () => {
  await render(KeyStrip, {
    onOpenSettings: () => {},
    cost: '$1.84',
    tokens: '412k',
    model: 'sonnet-4.6',
    effort: 'med',
  });
  await expect.element(page.getByText('$1.84')).toBeInTheDocument();
  await expect.element(page.getByText('412k')).toBeInTheDocument();
  await expect.element(page.getByText(/sonnet-4\.6/)).toBeInTheDocument();
});

test('the conn chip mirrors the store status', async () => {
  conn.status = 'reconnecting';
  const { container } = await render(KeyStrip, { onOpenSettings: () => {} });
  expect(container.querySelector('.t-conn')?.getAttribute('data-st')).toBe('re');
});
