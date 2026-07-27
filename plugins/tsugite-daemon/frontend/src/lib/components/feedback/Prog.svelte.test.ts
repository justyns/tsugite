/// <reference types="@vitest/browser/context" />
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Prog from './Prog.svelte';

test('determinate: reports value via aria and sets the fill width', async () => {
  const { container } = await render(Prog, { value: 64, label: 'attempt progress' });
  const el = container.querySelector('.t-prog');
  expect(el?.getAttribute('role')).toBe('progressbar');
  expect(el?.getAttribute('aria-label')).toBe('attempt progress');
  expect(el?.getAttribute('aria-valuemin')).toBe('0');
  expect(el?.getAttribute('aria-valuemax')).toBe('100');
  expect(el?.getAttribute('aria-valuenow')).toBe('64');
  expect(el?.classList.contains('t-prog--ind')).toBe(false);
  expect(el?.querySelector('i')?.getAttribute('style')).toBe('--w: 64%;');
});

test('indeterminate: omitting value drops the aria-value* triad and adds the sweep modifier', async () => {
  const { container } = await render(Prog, { label: 'working' });
  const el = container.querySelector('.t-prog');
  expect(el?.classList.contains('t-prog--ind')).toBe(true);
  expect(el?.hasAttribute('aria-valuenow')).toBe(false);
  expect(el?.hasAttribute('aria-valuemin')).toBe(false);
  expect(el?.hasAttribute('aria-valuemax')).toBe(false);
  expect(el?.querySelector('i')?.hasAttribute('style')).toBe(false);
});

test('clamps out-of-range values instead of overflowing the bar', async () => {
  const { container } = await render(Prog, { value: 140, label: 'attempt progress' });
  expect(container.querySelector('.t-prog')?.getAttribute('aria-valuenow')).toBe('100');
  const { container: c2 } = await render(Prog, { value: -20, label: 'attempt progress' });
  expect(c2.querySelector('.t-prog')?.getAttribute('aria-valuenow')).toBe('0');
});
