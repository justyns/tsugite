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
