/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { beforeEach, expect, test } from 'vitest';
import TokenPane from './TokenPane.svelte';
import { auth } from '$lib/stores/auth.svelte';

beforeEach(() => {
  auth.token = '';
  auth.gated = true;
  localStorage.removeItem('tsugite_token');
});

test('entering a token and connecting saves it and clears the gate', async () => {
  await render(TokenPane);
  await page.getByLabelText('Access token').fill('secret-123');
  await page.getByRole('button', { name: 'Connect' }).click();
  expect(auth.token).toBe('secret-123');
  expect(auth.gated).toBe(false);
  expect(localStorage.getItem('tsugite_token')).toBe('secret-123');
});

test('an empty token neither saves nor clears the gate', async () => {
  await render(TokenPane);
  await page.getByRole('button', { name: 'Connect' }).click();
  expect(auth.gated).toBe(true);
  expect(localStorage.getItem('tsugite_token')).toBeNull();
});
