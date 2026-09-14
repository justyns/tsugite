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
    model: 'anthropic:claude-sonnet-4-6',
  });
  await expect.element(page.getByText('$1.84')).toBeInTheDocument();
  await expect.element(page.getByText('412k')).toBeInTheDocument();
  await expect.element(page.getByText('claude-sonnet-4-6')).toBeInTheDocument();
});

test('the conn chip mirrors the store status', async () => {
  conn.status = 'reconnecting';
  const { container } = await render(KeyStrip, { onOpenSettings: () => {} });
  expect(container.querySelector('.t-conn')?.getAttribute('data-st')).toBe('re');
});

test('no session means no model line', async () => {
  // NavRail mounts this without a model whenever no session is selected.
  const { container } = await render(KeyStrip, {
    onOpenSettings: () => {},
    cost: '$0.00',
    tokens: '0',
  });
  await expect.element(page.getByText('$0.00')).toBeInTheDocument();
  expect(container.querySelector('.model')).toBeNull();
});
