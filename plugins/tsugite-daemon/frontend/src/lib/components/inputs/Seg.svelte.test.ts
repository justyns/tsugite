/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Seg from './Seg.svelte';

test('renders one button per option, marking the active one via aria-pressed', async () => {
  await render(Seg, {
    ariaLabel: 'Reasoning effort',
    options: ['low', 'med', 'high'],
    value: 'med',
  });
  await expect
    .element(page.getByRole('button', { name: 'low' }))
    .toHaveAttribute('aria-pressed', 'false');
  await expect
    .element(page.getByRole('button', { name: 'med' }))
    .toHaveAttribute('aria-pressed', 'true');
  await expect.element(page.getByRole('group', { name: 'Reasoning effort' })).toBeInTheDocument();
});

test('clicking a segment selects it', async () => {
  await render(Seg, {
    ariaLabel: 'Reasoning effort',
    options: ['low', 'med', 'high'],
    value: 'med',
  });
  await page.getByRole('button', { name: 'high' }).click();
  await expect
    .element(page.getByRole('button', { name: 'high' }))
    .toHaveAttribute('aria-pressed', 'true');
  await expect
    .element(page.getByRole('button', { name: 'med' }))
    .toHaveAttribute('aria-pressed', 'false');
});

test('arrow keys move focus and selection, wrapping at the ends', async () => {
  await render(Seg, {
    ariaLabel: 'Reasoning effort',
    options: ['low', 'med', 'high'],
    value: 'low',
  });
  const low = page.getByRole('button', { name: 'low' });
  await low.element().focus();

  await userEvent.keyboard('{ArrowRight}');
  await expect.element(page.getByRole('button', { name: 'med' })).toHaveFocus();
  await expect
    .element(page.getByRole('button', { name: 'med' }))
    .toHaveAttribute('aria-pressed', 'true');

  await userEvent.keyboard('{ArrowLeft}{ArrowLeft}');
  await expect.element(page.getByRole('button', { name: 'high' })).toHaveFocus();
  await expect
    .element(page.getByRole('button', { name: 'high' }))
    .toHaveAttribute('aria-pressed', 'true');

  await userEvent.keyboard('{Home}');
  await expect
    .element(page.getByRole('button', { name: 'low' }))
    .toHaveAttribute('aria-pressed', 'true');
  await userEvent.keyboard('{End}');
  await expect
    .element(page.getByRole('button', { name: 'high' }))
    .toHaveAttribute('aria-pressed', 'true');
});

test('only the active segment is a tab stop (roving tabindex)', async () => {
  const { container } = await render(Seg, {
    ariaLabel: 'Reasoning effort',
    options: ['low', 'med', 'high'],
    value: 'med',
  });
  const buttons = [...container.querySelectorAll('button')];
  expect(buttons.map((b) => b.tabIndex)).toEqual([-1, 0, -1]);
});
