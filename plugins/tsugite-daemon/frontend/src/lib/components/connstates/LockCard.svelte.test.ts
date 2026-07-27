/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import LockCard from './LockCard.svelte';

test('submitting with a passphrase calls onUnlock with the entered value', async () => {
  const onUnlock = vi.fn();
  await render(LockCard, { onUnlock });

  await page.getByLabelText('passphrase').fill('correct horse battery staple');
  await page.getByRole('button', { name: 'Unlock store' }).click();

  expect(onUnlock).toHaveBeenCalledExactlyOnceWith('correct horse battery staple');
});

test('required passphrase blocks submit when empty', async () => {
  const onUnlock = vi.fn();
  await render(LockCard, { onUnlock });

  await page.getByRole('button', { name: 'Unlock store' }).click();

  expect(onUnlock).not.toHaveBeenCalled();
});
