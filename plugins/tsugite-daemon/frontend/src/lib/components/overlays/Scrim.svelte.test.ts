/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { afterEach, expect, test, vi } from 'vitest';
import Scrim from './Scrim.svelte';

afterEach(cleanup);

const child = createRawSnippet(() => ({
  render: () => `<button data-testid="inside" type="button">x</button>`,
}));

test('toggles is-open with the open prop', async () => {
  const screen = await render(Scrim, { open: false, children: child });
  const scrim = document.querySelector('.t-scrim') as HTMLElement;
  expect(scrim.classList.contains('is-open')).toBe(false);
  await screen.rerender({ open: true });
  expect(scrim.classList.contains('is-open')).toBe(true);
});

test('backdrop click closes; a click on a child does not', async () => {
  const onclose = vi.fn();
  await render(Scrim, { open: true, onclose, children: child });
  const scrim = document.querySelector('.t-scrim') as HTMLElement;
  const inside = page.getByTestId('inside').element() as HTMLElement;
  inside.click();
  expect(onclose).not.toHaveBeenCalled();
  scrim.click();
  expect(onclose).toHaveBeenCalledTimes(1);
});
