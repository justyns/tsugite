/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import ThemeSeg from './ThemeSeg.svelte';
import { theme } from '$lib/stores/theme.svelte';

test('reflects the active theme and switches it through the store', async () => {
  theme.set('mocha');
  await render(ThemeSeg);
  await expect
    .element(page.getByRole('button', { name: 'mocha' }))
    .toHaveAttribute('aria-pressed', 'true');

  await page.getByRole('button', { name: 'latte' }).click();
  expect(theme.current).toBe('latte');
  await expect
    .element(page.getByRole('button', { name: 'latte' }))
    .toHaveAttribute('aria-pressed', 'true');
  theme.set('mocha');
});

test('only the active option is a tab stop (roving tabindex)', async () => {
  theme.set('mocha');
  const { container } = await render(ThemeSeg);
  const tabindexes = [...container.querySelectorAll('button')].map((b) => b.tabIndex);
  expect(tabindexes[0]).toBe(0);
  expect(tabindexes.slice(1).every((t) => t === -1)).toBe(true);
});
