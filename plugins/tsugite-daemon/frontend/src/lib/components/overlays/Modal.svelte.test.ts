/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { afterEach, expect, test, vi } from 'vitest';
import Modal from './Modal.svelte';

afterEach(cleanup);

const body = createRawSnippet(() => ({
  render: () => `<span>this can't be resumed</span>`,
}));

// createRawSnippet requires a single root, so the two actions are wrapped; the
// focus trap still finds both buttons as descendants of the dialog.
const confirmFooter = () =>
  createRawSnippet(() => ({
    render: () =>
      `<span><button type="button" data-autofocus data-testid="safe">Keep running</button>` +
      `<button type="button" class="t-btn--danger" data-testid="danger">Cancel job</button></span>`,
  }));

test('initial focus lands on the safe (data-autofocus) action when opened', async () => {
  const screen = await render(Modal, {
    open: false,
    title: 'Cancel job?',
    tone: 'danger',
    children: body,
    footer: confirmFooter(),
  });
  await screen.rerender({ open: true });
  const safe = page.getByTestId('safe').element();
  expect(document.activeElement).toBe(safe);
});

test('a specimen that mounts already-open does not steal focus', async () => {
  const before = document.activeElement;
  await render(Modal, { open: true, title: 'x', children: body, footer: confirmFooter() });
  expect(document.activeElement).toBe(before);
});

test('Escape cancels', async () => {
  const onclose = vi.fn();
  await render(Modal, { open: true, title: 'x', onclose, children: body, footer: confirmFooter() });
  const safe = page.getByTestId('safe').element() as HTMLElement;
  safe.focus();
  safe.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  expect(onclose).toHaveBeenCalledTimes(1);
});

test('Tab from the last action wraps to the first', async () => {
  await render(Modal, { open: true, title: 'x', children: body, footer: confirmFooter() });
  const safe = page.getByTestId('safe').element() as HTMLElement;
  const danger = page.getByTestId('danger').element() as HTMLElement;
  danger.focus();
  danger.dispatchEvent(
    new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }),
  );
  expect(document.activeElement).toBe(safe);
});

test('Shift+Tab from the first action wraps to the last', async () => {
  await render(Modal, { open: true, title: 'x', children: body, footer: confirmFooter() });
  const safe = page.getByTestId('safe').element() as HTMLElement;
  const danger = page.getByTestId('danger').element() as HTMLElement;
  safe.focus();
  safe.dispatchEvent(
    new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true, cancelable: true }),
  );
  expect(document.activeElement).toBe(danger);
});

test('backdrop click cancels; a click on the dialog does not', async () => {
  const onclose = vi.fn();
  await render(Modal, { open: true, title: 'x', onclose, children: body, footer: confirmFooter() });
  const dialog = page.getByRole('dialog').element() as HTMLElement;
  const scrim = dialog.parentElement as HTMLElement;
  dialog.click();
  expect(onclose).not.toHaveBeenCalled();
  scrim.click();
  expect(onclose).toHaveBeenCalledTimes(1);
});

test('danger tone renders the alert icon; default tone omits it', async () => {
  const screen = await render(Modal, {
    open: true,
    title: 'Cancel job?',
    tone: 'danger',
    children: body,
    footer: confirmFooter(),
  });
  let dialog = page.getByRole('dialog').element();
  expect(dialog.querySelector('svg.ic--danger')).not.toBeNull();

  await screen.rerender({ tone: 'default' });
  dialog = page.getByRole('dialog').element();
  expect(dialog.querySelector('svg.ic--danger')).toBeNull();
});
