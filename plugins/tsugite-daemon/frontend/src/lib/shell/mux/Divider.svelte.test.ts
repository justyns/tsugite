/// <reference types="@vitest/browser/context" />
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, expect, test } from 'vitest';
import Divider from './Divider.svelte';

afterEach(cleanup);

function sepOf(container: HTMLElement): HTMLElement {
  return container.querySelector<HTMLElement>('[role="separator"]')!;
}

// WAI-ARIA: aria-valuenow must never sit outside [aria-valuemin, aria-valuemax].
// The share handed in (a pane's percent of its pair) can exceed that band in an
// over-split layout, so the divider has to clamp it into its own declared range.
test('aria-valuenow is clamped into the declared min/max even for out-of-range input', async () => {
  const { container, rerender } = await render(Divider, {
    dir: 'row' as const,
    splitId: 's',
    index: 0,
    valueNow: 161, // wildly over range (e.g. the corrupted-pair case)
  });
  const sep = sepOf(container);
  const min = Number(sep.getAttribute('aria-valuemin'));
  const max = Number(sep.getAttribute('aria-valuemax'));
  const high = Number(sep.getAttribute('aria-valuenow'));
  expect(high).toBeLessThanOrEqual(max);
  expect(high).toBeGreaterThanOrEqual(min);

  await rerender({ dir: 'row' as const, splitId: 's', index: 0, valueNow: 2 }); // below the 5% floor
  const low = Number(sepOf(container).getAttribute('aria-valuenow'));
  expect(low).toBeGreaterThanOrEqual(min);
  expect(low).toBeLessThanOrEqual(max);

  await rerender({ dir: 'row' as const, splitId: 's', index: 0, valueNow: 42 }); // in range: untouched
  expect(Number(sepOf(container).getAttribute('aria-valuenow'))).toBe(42);
});
