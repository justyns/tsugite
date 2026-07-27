/// <reference types="@vitest/browser/context" />
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Spin from './Spin.svelte';

// The frame cadence (96ms), wrap, and reduced-motion glyph are the shared
// startSpin() driver's contract - covered by buttons/spin.test.ts. These stay at
// the component boundary: what the rendered <span> looks like.

test('renders a decorative, aria-hidden spinner glyph', async () => {
  const { container } = await render(Spin);
  const el = container.querySelector('.t-spin');
  expect(el?.getAttribute('aria-hidden')).toBe('true');
  expect(el?.textContent).toBeTruthy();
});

test('an explicit color prop rides the --spin-c variable', async () => {
  const { container } = await render(Spin, { color: 'var(--st-warn)' });
  expect(container.querySelector('.t-spin')?.getAttribute('style')).toBe(
    '--spin-c: var(--st-warn);',
  );
});

test('freezes on a static glyph under prefers-reduced-motion', async () => {
  const original = window.matchMedia;
  window.matchMedia = ((query: string) => ({
    matches: query.includes('reduce'),
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as typeof window.matchMedia;

  try {
    const { container } = await render(Spin);
    expect(container.querySelector('.t-spin')?.textContent).toBe('∙');
  } finally {
    window.matchMedia = original;
  }
});
