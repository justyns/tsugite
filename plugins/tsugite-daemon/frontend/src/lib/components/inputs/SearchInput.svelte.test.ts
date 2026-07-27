/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, afterEach } from 'vitest';
import SearchInput from './SearchInput.svelte';

afterEach(() => {
  // Keydown listeners are window-scoped; make sure a stray focused element
  // from one test can't leak into the next.
  (document.activeElement as HTMLElement | null)?.blur?.();
});

test('renders a search box with the icon + hint chip', async () => {
  await render(SearchInput, { ariaLabel: 'Search sessions', placeholder: 'search sessions…' });
  const input = page.getByRole('searchbox', { name: 'Search sessions' });
  await expect.element(input).toBeInTheDocument();
  await expect.element(input).toHaveAttribute('placeholder', 'search sessions…');
});

test('typing updates the bound value', async () => {
  await render(SearchInput, { ariaLabel: 'Search sessions', value: '' });
  const input = page.getByRole('searchbox', { name: 'Search sessions' });
  await input.fill('stuck jobs');
  await expect.element(input).toHaveValue('stuck jobs');
});

test('pressing the shortcut key focuses the input from anywhere on the page', async () => {
  document.body.focus();
  await render(SearchInput, { ariaLabel: 'Search sessions', shortcutKey: '/' });
  const input = page.getByRole('searchbox', { name: 'Search sessions' });
  await expect.element(input).not.toHaveFocus();

  await userEvent.keyboard('/');
  await expect.element(input).toHaveFocus();
});

test('the shortcut key does not hijack focus while typing in another field', async () => {
  await render(SearchInput, { ariaLabel: 'Search sessions', shortcutKey: '/' });
  const decoy = document.createElement('input');
  document.body.appendChild(decoy);
  decoy.focus();

  await userEvent.keyboard('/');
  expect(document.activeElement).toBe(decoy);

  decoy.remove();
});

test('search inputs opt out of password managers', async () => {
  const { container } = await render(SearchInput, { ariaLabel: 'filter' });
  const input = container.querySelector('input')!;
  // type=search keeps Chromium's built-in manager away; the data-* opt-outs
  // and autocomplete=off cover the password-manager extensions.
  expect(input.getAttribute('type')).toBe('search');
  expect(input.getAttribute('data-1p-ignore')).not.toBeNull();
  expect(input.getAttribute('data-lpignore')).toBe('true');
  expect(input.getAttribute('data-bwignore')).not.toBeNull();
  expect(input.getAttribute('autocomplete')).toBe('off');
});
