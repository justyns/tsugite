/// <reference types="@vitest/browser/context" />
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import AppView from './AppView.svelte';

const base = {
  name: 'Deploy Monitor',
  source: 'deploy-server',
  iconChar: 'D',
  iconColor: '#1F8A5B',
};

test('mode / reload / border controls fire their callbacks', async () => {
  const onMode = vi.fn();
  const onReload = vi.fn();
  const onBorderToggle = vi.fn();
  const screen = await render(AppView, { ...base, onMode, onReload, onBorderToggle });

  await screen.getByRole('button', { name: 'pip' }).click();
  expect(onMode).toHaveBeenCalledWith('pip');

  await screen.getByRole('button', { name: 'Reload' }).click();
  expect(onReload).toHaveBeenCalledTimes(1);

  await screen.getByRole('button', { name: 'Border preference' }).click();
  expect(onBorderToggle).toHaveBeenCalledTimes(1);
});

test('active mode is reflected with aria-pressed', async () => {
  const screen = await render(AppView, { ...base, mode: 'inline' });
  await expect
    .element(screen.getByRole('button', { name: 'inline' }))
    .toHaveAttribute('aria-pressed', 'true');
  await expect
    .element(screen.getByRole('button', { name: 'pip' }))
    .toHaveAttribute('aria-pressed', 'false');
});

test('init lifecycle shows the handshake', async () => {
  const screen = await render(AppView, { ...base, life: 'init' });
  await expect.element(screen.getByText(/ui\/initialize/)).toBeVisible();
});

test('ready lifecycle hides the handshake', async () => {
  const screen = await render(AppView, { ...base, life: 'ready' });
  await expect.element(screen.getByText(/ui\/initialize/)).not.toBeVisible();
});
