/// <reference types="@vitest/browser/context" />
import { userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import GenUI from './GenUI.svelte';

const choices = ['Roll forward now', 'Wait for the job to finish', 'Hold — decide after review'];

test('clicking a choice picks it and marks aria-checked', async () => {
  const onPick = vi.fn();
  const screen = await render(GenUI, { question: 'How to sequence?', choices, onPick });
  await screen.getByRole('radio', { name: /Wait for the job/ }).click();
  expect(onPick).toHaveBeenCalledWith(1);
  await expect
    .element(screen.getByRole('radio', { name: /Wait for the job/ }))
    .toHaveAttribute('aria-checked', 'true');
});

test('number keys pick the matching choice', async () => {
  const onPick = vi.fn();
  const screen = await render(GenUI, { question: 'Q', choices, onPick });
  (screen.getByRole('radio', { name: /Roll forward/ }).element() as HTMLElement).focus();
  await userEvent.keyboard('3');
  expect(onPick).toHaveBeenCalledWith(2);
});

test('arrow keys move roving focus', async () => {
  const screen = await render(GenUI, { question: 'Q', choices });
  (screen.getByRole('radio', { name: /Roll forward/ }).element() as HTMLElement).focus();
  await userEvent.keyboard('{ArrowDown}');
  await expect.element(screen.getByRole('radio', { name: /Wait for the job/ })).toHaveFocus();
});

test('resolved state marks the selection and blocks further picks', async () => {
  const onPick = vi.fn();
  const screen = await render(GenUI, { question: 'Q', choices, selected: 0, onPick });
  await expect
    .element(screen.getByRole('radio', { name: /Roll forward/ }))
    .toHaveAttribute('aria-checked', 'true');
  (screen.getByRole('radio', { name: /Roll forward/ }).element() as HTMLElement).focus();
  await userEvent.keyboard('2');
  expect(onPick).not.toHaveBeenCalled();
});
