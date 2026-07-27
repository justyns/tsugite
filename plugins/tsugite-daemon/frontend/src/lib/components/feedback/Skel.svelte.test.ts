/// <reference types="@vitest/browser/context" />
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Skel from './Skel.svelte';

test('defaults to 4 rows with varied widths, announced as loading', async () => {
  const { container } = await render(Skel);
  const status = container.querySelector('.t-skel');
  expect(status?.getAttribute('role')).toBe('status');
  expect(status?.getAttribute('aria-label')).toBe('Loading');
  const bars = container.querySelectorAll('.t-skel i');
  expect(bars).toHaveLength(4);
  expect(Array.from(bars).map((b) => (b as HTMLElement).style.getPropertyValue('--w'))).toEqual([
    '72%',
    '88%',
    '55%',
    '80%',
  ]);
});

test('bars are decorative; the status role carries the announcement', async () => {
  const { container } = await render(Skel);
  const bars = container.querySelectorAll('.t-skel i');
  bars.forEach((b) => expect(b.getAttribute('aria-hidden')).toBe('true'));
});

test('rows and label props are honored', async () => {
  const { container } = await render(Skel, { rows: 2, label: 'Loading sessions' });
  expect(container.querySelectorAll('.t-skel i')).toHaveLength(2);
  expect(container.querySelector('.t-skel')?.getAttribute('aria-label')).toBe('Loading sessions');
});

test('the width pattern cycles for row counts beyond the base palette', async () => {
  const { container } = await render(Skel, { rows: 8 });
  const widths = Array.from(container.querySelectorAll('.t-skel i')).map((b) =>
    b.getAttribute('style'),
  );
  expect(widths[6]).toBe(widths[0]);
  expect(widths[7]).toBe(widths[1]);
});
