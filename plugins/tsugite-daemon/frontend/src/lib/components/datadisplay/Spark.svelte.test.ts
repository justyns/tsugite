/// <reference types="vitest/browser" />
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Spark from './Spark.svelte';

test('renders one bar per point, labelled as an image for a11y', async () => {
  const { container } = await render(Spark, {
    label: 'last 3 runs: 1 failed',
    points: [{ height: 7 }, { height: 9, status: 'fail' }, { height: 5, status: 'skip' }],
  });
  await expect
    .element(page.getByRole('img', { name: 'last 3 runs: 1 failed' }))
    .toBeInTheDocument();

  const bars = Array.from(container.querySelectorAll('.t-spark i')) as HTMLElement[];
  expect(bars).toHaveLength(3);
  expect(bars[0]!.classList.contains('f')).toBe(false);
  expect(bars[0]!.classList.contains('s')).toBe(false);
  expect(bars[0]!.style.getPropertyValue('--h')).toBe('7px');
  expect(bars[1]!.classList.contains('f')).toBe(true);
  expect(bars[2]!.classList.contains('s')).toBe(true);
});
