/// <reference types="vitest/browser" />
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Stat from './Stat.svelte';

test('renders value + label, no delta by default', async () => {
  await render(Stat, { value: '12', label: 'open' });
  await expect.element(page.getByText('12')).toBeInTheDocument();
  await expect.element(page.getByText('open')).toBeInTheDocument();
});

test('tone colors the value via a semantic token, not a hardcoded color', async () => {
  const { container } = await render(Stat, { value: '1', label: 'blocked', tone: 'warn' });
  const v = container.querySelector('.v') as HTMLElement;
  expect(v.dataset.tone).toBe('warn');
  // Prove the color rides the --st-warn token: pin it on an ancestor and the
  // value's computed color must follow.
  container.style.setProperty('--st-warn', 'rgb(1, 2, 3)');
  expect(getComputedStyle(v).color).toBe('rgb(1, 2, 3)');
});

test('delta is opt-in', async () => {
  const { container, rerender } = await render(Stat, { value: '42', label: 'done today' });
  expect(container.querySelector('.d')).toBeNull();

  await rerender({ value: '42', label: 'done today', delta: '+6 vs yesterday' });
  expect(container.querySelector('.d')?.textContent).toBe('+6 vs yesterday');
});
