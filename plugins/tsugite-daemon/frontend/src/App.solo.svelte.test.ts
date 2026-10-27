/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import App from './App.svelte';
import { auth } from '$lib/stores/auth.svelte';
import { sessions } from '$lib/stores/sessions.svelte';
import { sessionRow } from './views/chats/__fixtures__/sessionRow';

afterEach(() => {
  location.hash = '';
  auth.gated = true;
  sessions.rows = [];
  vi.restoreAllMocks();
});

test('solo mode ignores the palette and new-chat shortcuts', async () => {
  auth.gated = false;
  sessions.rows = [sessionRow('s1')];
  vi.spyOn(sessions, 'load').mockResolvedValue();
  const create = vi.spyOn(sessions, 'newSession');
  location.hash = '#solo?sessionId=s1';
  await render(App);
  await expect.element(page.getByRole('main', { name: 'Chat' })).toBeInTheDocument();
  await userEvent.keyboard('{Control>}k{/Control}');
  await userEvent.keyboard('{Control>}{Shift>}o{/Shift}{/Control}');
  await expect
    .element(page.getByRole('combobox', { name: 'Command palette search' }))
    .not.toBeInTheDocument();
  expect(location.hash).toBe('#solo?sessionId=s1');
  expect(create).not.toHaveBeenCalled();
});
