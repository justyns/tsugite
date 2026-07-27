/// <reference types="vitest/browser" />
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Meter from './Meter.svelte';

test('renders meter semantics and the visible readout', async () => {
  await render(Meter, {
    value: 9200,
    max: 200000,
    label: 'Context 9.2k of 200k tokens',
    displayText: '9.2k/200k',
  });
  const meter = page.getByRole('meter', { name: 'Context 9.2k of 200k tokens' });
  await expect.element(meter).toBeInTheDocument();
  await expect.element(meter).toHaveAttribute('aria-valuenow', '9200');
  await expect.element(meter).toHaveAttribute('aria-valuemin', '0');
  await expect.element(meter).toHaveAttribute('aria-valuemax', '200000');
  // Screen readers announce the friendly readout, not the raw 9200.
  await expect.element(meter).toHaveAttribute('aria-valuetext', '9.2k/200k');
  await expect.element(page.getByText('9.2k/200k')).toBeInTheDocument();
});

test('fill width is proportional and clamped to the [0,100] range', async () => {
  const { container, rerender } = await render(Meter, {
    value: 9200,
    max: 200000,
    label: 'ctx',
    displayText: '9.2k/200k',
  });
  const bar = () => container.querySelector('.bar i') as HTMLElement;
  expect(bar().style.getPropertyValue('--w')).toBe('4.6%');

  await rerender({ value: 500000, max: 200000, label: 'ctx', displayText: 'over' });
  expect(bar().style.getPropertyValue('--w')).toBe('100%');

  await rerender({ value: -50, max: 200000, label: 'ctx', displayText: 'under' });
  expect(bar().style.getPropertyValue('--w')).toBe('0%');

  // degenerate range (max <= min) must not divide by zero / render NaN%
  await rerender({ value: 5, max: 0, min: 0, label: 'ctx', displayText: 'degenerate' });
  expect(bar().style.getPropertyValue('--w')).toBe('0%');
});

test('is-warn modifier is opt-in and off by default', async () => {
  const { container, rerender } = await render(Meter, {
    value: 10,
    max: 100,
    label: 'ctx',
    displayText: '10/100',
  });
  expect(container.querySelector('.t-meter')!.className).not.toContain('is-warn');

  await rerender({ value: 10, max: 100, label: 'ctx', displayText: '10/100', warn: true });
  expect(container.querySelector('.t-meter')!.className).toContain('is-warn');
});
