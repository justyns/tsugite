/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import SetSecretModal from './SetSecretModal.svelte';

test('rotate mode locks the name field and pre-fills it', async () => {
  await render(SetSecretModal, { mode: 'rotate', name: 'OPENAI_API_KEY' });
  const nameInput = page.getByLabelText('name');
  await expect.element(nameInput).toHaveValue('OPENAI_API_KEY');
  await expect.element(nameInput).toHaveAttribute('readonly');
});

test('add mode leaves the name field editable and empty', async () => {
  await render(SetSecretModal, { mode: 'add' });
  const nameInput = page.getByLabelText('name');
  await expect.element(nameInput).toHaveValue('');
  await expect.element(nameInput).not.toHaveAttribute('readonly');
});

test('show/hide toggles the value field between password and text', async () => {
  await render(SetSecretModal, { mode: 'add' });
  const valueInput = page.getByLabelText('value');
  await expect.element(valueInput).toHaveAttribute('type', 'password');

  await page.getByRole('button', { name: 'show' }).click();
  await expect.element(valueInput).toHaveAttribute('type', 'text');

  await page.getByRole('button', { name: 'hide' }).click();
  await expect.element(valueInput).toHaveAttribute('type', 'password');
});

test('rotate: saving calls onSave with the locked name and typed value', async () => {
  const onSave = vi.fn();
  await render(SetSecretModal, { mode: 'rotate', name: 'OPENAI_API_KEY', onSave });

  await page.getByLabelText('value').fill('sk-new-value');
  await page.getByRole('button', { name: 'Rotate value' }).click();

  expect(onSave).toHaveBeenCalledExactlyOnceWith({
    name: 'OPENAI_API_KEY',
    value: 'sk-new-value',
  });
});

test('add: saving calls onSave with the typed name', async () => {
  const onSave = vi.fn();
  await render(SetSecretModal, { mode: 'add', onSave });

  await page.getByLabelText('name').fill('NEW_SECRET');
  await page.getByLabelText('value').fill('v');
  await page.getByRole('button', { name: 'Add secret' }).click();

  expect(onSave).toHaveBeenCalledExactlyOnceWith({
    name: 'NEW_SECRET',
    value: 'v',
  });
});

test('cancel calls onCancel without calling onSave', async () => {
  const onCancel = vi.fn();
  const onSave = vi.fn();
  await render(SetSecretModal, { mode: 'add', onCancel, onSave });

  await page.getByRole('button', { name: 'Cancel' }).click();

  expect(onCancel).toHaveBeenCalledTimes(1);
  expect(onSave).not.toHaveBeenCalled();
});

test('Escape calls onCancel', async () => {
  const onCancel = vi.fn();
  const { container } = await render(SetSecretModal, { mode: 'add', onCancel });

  const dialog = container.querySelector('[role="dialog"]') as HTMLElement;
  dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

  expect(onCancel).toHaveBeenCalledTimes(1);
});

test('Tab cycles from the last focusable element back to the first (focus trap)', async () => {
  const { container } = await render(SetSecretModal, { mode: 'rotate', name: 'OPENAI_API_KEY' });
  const dialog = container.querySelector('[role="dialog"]') as HTMLElement;
  const focusables = dialog.querySelectorAll<HTMLElement>('button, input, select');
  const first = focusables[0]!;
  const last = focusables[focusables.length - 1]!;

  last.focus();
  expect(document.activeElement).toBe(last);
  last.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }));
  expect(document.activeElement).toBe(first);

  first.dispatchEvent(
    new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true, cancelable: true }),
  );
  expect(document.activeElement).toBe(last);
});
