/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { beforeEach, expect, test } from 'vitest';
import SettingsDrawer from './SettingsDrawer.svelte';
import { autoAttachStore } from '$lib/stores/autoAttach.svelte';
import { hardLineBreaks } from '$lib/stores/hardLineBreaks.svelte';

beforeEach(() => {
  localStorage.removeItem('tsugite_auto_follow');
});

test('renders the old-UI parity fields', async () => {
  await render(SettingsDrawer, { open: true, onclose: () => {} });
  await expect.element(page.getByLabelText('Access token')).toBeInTheDocument();
  await expect.element(page.getByLabelText('User ID')).toBeInTheDocument();
  await expect.element(page.getByRole('group', { name: 'Theme' })).toBeInTheDocument();
  await expect
    .element(page.getByRole('switch', { name: 'Auto-follow new output' }))
    .toBeInTheDocument();
});

test('toggling auto-follow persists the pref to localStorage', async () => {
  await render(SettingsDrawer, { open: true, onclose: () => {} });
  // Defaults on; one click turns it off and writes the pref.
  await page.getByRole('switch', { name: 'Auto-follow new output' }).click();
  expect(localStorage.getItem('tsugite_auto_follow')).toBe('false');
});

test('toggling a context provider auto-attach persists through its store', async () => {
  autoAttachStore('tsugite_geo_autoattach').set(false);
  await render(SettingsDrawer, { open: true, onclose: () => {} });
  await page.getByRole('switch', { name: 'Auto-attach my location to messages' }).click();
  expect(localStorage.getItem('tsugite_geo_autoattach')).toBe('true');
});

test('hard line breaks are on out of the box, and turning them off persists', async () => {
  hardLineBreaks.set(true);
  await render(SettingsDrawer, { open: true, onclose: () => {} });
  const toggle = page.getByRole('switch', { name: 'Render soft line breaks as hard line breaks' });
  await expect.element(toggle).toHaveAttribute('aria-checked', 'true');

  await toggle.click();
  expect(hardLineBreaks.enabled).toBe(false);
  expect(localStorage.getItem('tsugite_hard_line_breaks')).toBe('false');
});
