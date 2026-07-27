/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { expect, test } from 'vitest';
import Field from './Field.svelte';

/** A minimal `<input>` snippet standing in for a real control (e.g. Input.svelte). */
function inputSnippet() {
  return createRawSnippet<[string | undefined]>((getDescribedBy) => ({
    render: () => `<input aria-describedby="${getDescribedBy() ?? ''}" />`,
  }));
}

test('label is wired to the control id', async () => {
  const { container } = await render(Field, {
    id: 'whk',
    label: 'webhook secret',
    children: inputSnippet(),
  });
  const label = container.querySelector('label')!;
  expect(label.textContent).toBe('webhook secret');
  expect(label.getAttribute('for')).toBe('whk');
});

test('error message gets an id and is exposed to the control via the children snippet param', async () => {
  const { container } = await render(Field, {
    id: 'whk',
    label: 'webhook secret',
    error: 'signature check failed on last delivery',
    children: inputSnippet(),
  });
  await expect
    .element(page.getByText('signature check failed on last delivery'))
    .toBeInTheDocument();

  const input = container.querySelector('input')!;
  const msg = container.querySelector('.msg')!;
  expect(msg.id).toBeTruthy();
  expect(input.getAttribute('aria-describedby')).toBe(msg.id);
});

test('hint renders when there is no error, sharing the same describedBy contract', async () => {
  const { container } = await render(Field, {
    id: 'model',
    label: 'model',
    hint: 'used for all new sessions',
    children: inputSnippet(),
  });
  const input = container.querySelector('input')!;
  const hint = container.querySelector('.hint')!;
  expect(hint.textContent).toBe('used for all new sessions');
  expect(input.getAttribute('aria-describedby')).toBe(hint.id);
});

test('error takes priority over hint when both are set', async () => {
  const { container } = await render(Field, {
    id: 'x',
    label: 'x',
    hint: 'a neutral hint',
    error: 'an error',
    children: inputSnippet(),
  });
  expect(container.querySelector('.hint')).toBeNull();
  expect(container.querySelector('.msg')!.textContent).toContain('an error');
});

test('renders no message and no describedBy when neither hint nor error is set', async () => {
  const { container } = await render(Field, {
    id: 'plain',
    label: 'plain field',
    children: inputSnippet(),
  });
  expect(container.querySelector('.msg')).toBeNull();
  expect(container.querySelector('.hint')).toBeNull();
  expect(container.querySelector('input')!.getAttribute('aria-describedby')).toBe('');
});
