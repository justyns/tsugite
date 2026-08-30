/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Think from './Think.svelte';

test('the reasoning toggle starts collapsed and expands on click', async () => {
  render(Think, { label: 'thought for 6s', tokens: 1024, content: 'heartbeat gap reasoning' });
  const toggle = page.getByRole('button', { name: /thought for 6s/ });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
  await toggle.click();
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
});

test('respects an initial open prop', async () => {
  render(Think, { label: 'thought for 2s', content: 'already visible', open: true });
  const toggle = page.getByRole('button', { name: /thought for 2s/ });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
});

test('follows the open prop when it changes', async () => {
  const { rerender } = await render(Think, {
    label: 'thought for 4s',
    content: 'reconsidering',
    open: true,
  });
  const toggle = page.getByRole('button', { name: /thought for 4s/ });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');

  await rerender({ open: false });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
});

test('a manual toggle wins until the open prop next changes', async () => {
  const { rerender } = await render(Think, {
    label: 'thought for 9s',
    content: 'reconsidering',
    open: true,
  });
  const toggle = page.getByRole('button', { name: /thought for 9s/ });
  await toggle.click();
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');

  await rerender({ open: false });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
  await toggle.click();
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
});
