/// <reference types="@vitest/browser/context" />
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Caret from './Caret.svelte';

test('renders the blinking token caret as a decorative, aria-hidden element', async () => {
  const { container } = await render(Caret);
  const el = container.querySelector('.t-caret');
  expect(el).not.toBeNull();
  expect(el?.getAttribute('aria-hidden')).toBe('true');
});
