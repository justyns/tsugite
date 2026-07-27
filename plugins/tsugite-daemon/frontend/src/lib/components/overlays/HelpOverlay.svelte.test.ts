/// <reference types="@vitest/browser/context" />
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, expect, test } from 'vitest';
import HelpOverlay from './HelpOverlay.svelte';

afterEach(cleanup);

const scrim = () => document.querySelector('.t-scrim') as HTMLElement;

test('open renders the grouped shortcuts as keycap chips; the prop toggles it', async () => {
  const screen = await render(HelpOverlay, { open: true });
  expect(scrim().classList.contains('is-open')).toBe(true);

  // One row from each group proves the sectioned render off the SHORTCUTS source.
  const text = document.querySelector('.t-modal')!.textContent ?? '';
  expect(text).toContain('Command palette');
  expect(text).toContain('Next tab');
  expect(text).toContain('Stop generating');

  // Keys render as .t-kbd chips, with the 'Mod' token resolved for the platform
  // (Ctrl in the Linux test browser, ⌘ on mac).
  const chips = [...document.querySelectorAll('.t-modal .t-kbd')];
  expect(chips.length).toBeGreaterThan(0);
  expect(chips.some((c) => c.textContent === 'Ctrl' || c.textContent === '⌘')).toBe(true);

  await screen.rerender({ open: false });
  expect(scrim().classList.contains('is-open')).toBe(false);
});
