/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import Picker, { type PickItem } from './Picker.svelte';
import { TESTID } from '$lib/testids';

afterEach(cleanup);

// Source order deliberately differs from 'apple' match order (pineapple scores
// below apple pie), so the filter test also proves score-ordering.
const items: PickItem[] = [
  { value: 'pine', label: 'pineapple', detail: 'fruit' },
  { value: 'pie', label: 'apple pie', detail: 'dessert' },
  { value: 'bread', label: 'banana bread' },
];

const noop = () => {};

test('renders every item as a listbox option', async () => {
  render(Picker, { items, onPick: noop, onClose: noop });
  await expect.element(page.getByTestId(TESTID.picker)).toBeInTheDocument();
  expect(page.getByRole('option').elements()).toHaveLength(3);
  await expect.element(page.getByText('pineapple')).toBeInTheDocument();
  await expect.element(page.getByText('apple pie')).toBeInTheDocument();
});

test('typing filters to matches and orders them by score', async () => {
  render(Picker, { items, onPick: noop, onClose: noop });
  const input = page.getByRole('combobox');
  await input.click();
  await userEvent.type(input, 'apple');
  const options = page.getByRole('option');
  expect(options.elements()).toHaveLength(2); // banana bread dropped
  // apple pie (substring at 0) outranks pineapple (substring at 4) despite
  // coming second in source order.
  await expect.element(options.first()).toHaveTextContent('apple pie');
  await expect.element(page.getByText('banana bread')).not.toBeInTheDocument();
});

test('ArrowDown then Enter picks the highlighted value', async () => {
  const onPick = vi.fn();
  render(Picker, { items, onPick, onClose: noop });
  await page.getByRole('combobox').click();
  await userEvent.keyboard('{ArrowDown}{Enter}');
  expect(onPick).toHaveBeenCalledTimes(1);
  expect(onPick).toHaveBeenCalledWith('pie');
});

test('clicking a row picks its value', async () => {
  const onPick = vi.fn();
  render(Picker, { items, onPick, onClose: noop });
  await page.getByTestId(TESTID.pickerOption('pine')).click();
  expect(onPick).toHaveBeenCalledTimes(1);
  expect(onPick).toHaveBeenCalledWith('pine');
});

test('Escape fires onClose', async () => {
  const onClose = vi.fn();
  render(Picker, { items, onPick: noop, onClose });
  await page.getByRole('combobox').click();
  await userEvent.keyboard('{Escape}');
  expect(onClose).toHaveBeenCalledTimes(1);
});

test('scrim click fires onClose; a click inside the dialog does not', async () => {
  const onClose = vi.fn();
  render(Picker, { items, onPick: noop, onClose });
  const dialog = page.getByTestId(TESTID.picker).element() as HTMLElement;
  const scrim = dialog.parentElement as HTMLElement;
  dialog.click();
  expect(onClose).not.toHaveBeenCalled();
  scrim.click();
  expect(onClose).toHaveBeenCalledTimes(1);
});
