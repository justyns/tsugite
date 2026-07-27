/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test } from 'vitest';
import Surface from './Surface.svelte';
import { terminals } from '$lib/stores/terminals.svelte';
import { routeHistory } from '$lib/router.svelte';

// Reset to a desktop width so a narrow-viewport test never leaks into the next
// file (the browser instance is shared and viewport persists).
afterEach(async () => {
  await page.viewport(1440, 900);
});

test('a dead-link content screen still offers a phone back to the terminal list', async () => {
  // Deep link to a terminalId that is not in the list -> the "not available" screen,
  // which has no canvas header. At phone width it must still expose a way back, and
  // back clears ?terminalId to the #terminals list.
  await page.viewport(390, 780);
  routeHistory.prev = null;
  terminals.list = [];
  location.hash = '#terminals?terminalId=gone';
  render(Surface, { params: { terminalId: 'gone' } });
  await expect.element(page.getByText('Terminal not available')).toBeInTheDocument();
  await expect.element(page.getByTestId('phone-back')).toBeVisible();
  await page.getByTestId('phone-back').click();
  expect(location.hash).toBe('#terminals');
});

test('at desktop width the terminal-unavailable screen has no phone back', async () => {
  await page.viewport(1440, 900);
  terminals.list = [];
  render(Surface, { params: { terminalId: 'gone' } });
  await expect.element(page.getByText('Terminal not available')).toBeInTheDocument();
  await expect.element(page.getByTestId('phone-back')).not.toBeVisible();
});
