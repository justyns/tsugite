/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import TopBar from './TopBar.svelte';

test('the palette trigger fires its callback', async () => {
  const onOpenPalette = vi.fn();
  await render(TopBar, { onOpenPalette });
  await page.getByRole('button', { name: 'Command palette' }).click();
  expect(onOpenPalette).toHaveBeenCalledOnce();
});

test('the settings trigger fires its callback', async () => {
  const onOpenSettings = vi.fn();
  await render(TopBar, { onOpenPalette: () => {}, onOpenSettings });
  // Settings sits in the top bar only on mobile (the rail footer that hosts it
  // on desktop is hidden below 640px); fire the DOM click so the wiring is
  // proven regardless of the runner's viewport.
  const btn = page.getByRole('button', { name: 'Settings' }).element() as HTMLElement;
  btn.click();
  expect(onOpenSettings).toHaveBeenCalledOnce();
});

// The theme control itself (reflect + switch + roving tabindex) is covered by
// ThemeSeg.svelte.test.ts; the top bar hides it below 640px, so it isn't
// clickable at the browser runner's default viewport.
test('renders the brand wordmark', async () => {
  await render(TopBar, { onOpenPalette: () => {} });
  await expect.element(page.getByText('tsugite')).toBeInTheDocument();
});
