/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { afterEach, expect, test, vi } from 'vitest';
import Button from './Button.svelte';

afterEach(cleanup);

const label = (text: string) => createRawSnippet(() => ({ render: () => `<span>${text}</span>` }));

test('an enabled button calls onclick when clicked', async () => {
  const onclick = vi.fn();
  render(Button, { onclick, children: label('Retry') });
  await page.getByRole('button', { name: 'Retry' }).click();
  expect(onclick).toHaveBeenCalledOnce();
});

test('a loading button is aria-busy, hides its icon, and swallows clicks', async () => {
  const onclick = vi.fn();
  const icon = createRawSnippet(() => ({
    render: () => `<svg class="ic" data-testid="icn"></svg>`,
  }));
  render(Button, { onclick, loading: true, icon, children: label('Retry') });

  const btn = page.getByRole('button', { name: 'Retry' });
  await expect.element(btn).toHaveAttribute('aria-busy', 'true');
  await expect.element(page.getByTestId('icn')).not.toBeInTheDocument();

  // `pointer-events:none` (the CSS loading treatment) legitimately blocks a
  // real pointer click, but does nothing for keyboard activation - that's
  // what the JS guard in handleClick is for. A native, non-actionability-
  // checked click exercises that guard directly instead of hanging on
  // Playwright's "receives pointer events" check.
  (btn.element() as HTMLElement).click();
  expect(onclick).not.toHaveBeenCalled();
});

test('a disabled button is natively inert', async () => {
  const onclick = vi.fn();
  render(Button, { onclick, disabled: true, children: label('Retry') });
  const btn = page.getByRole('button', { name: 'Retry' });
  await expect.element(btn).toBeDisabled();
  (btn.element() as HTMLElement).click();
  expect(onclick).not.toHaveBeenCalled();
});

test('an icon-only button exposes its aria-label as the accessible name', async () => {
  const icon = createRawSnippet(() => ({ render: () => `<svg class="ic"></svg>` }));
  render(Button, { iconOnly: true, icon, 'aria-label': 'More actions' });
  await expect.element(page.getByRole('button', { name: 'More actions' })).toBeInTheDocument();
});
