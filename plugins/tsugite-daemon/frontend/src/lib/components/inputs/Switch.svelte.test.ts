/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Switch from './Switch.svelte';

test('renders as a labelled switch reflecting the checked prop', async () => {
  await render(Switch, { ariaLabel: 'Enabled', checked: true });
  await expect
    .element(page.getByRole('switch', { name: 'Enabled' }))
    .toHaveAttribute('aria-checked', 'true');
});

test('starts unchecked when checked is false', async () => {
  await render(Switch, { ariaLabel: 'Disabled', checked: false });
  await expect
    .element(page.getByRole('switch', { name: 'Disabled' }))
    .toHaveAttribute('aria-checked', 'false');
});

test('clicking toggles aria-checked', async () => {
  await render(Switch, { ariaLabel: 'Enabled', checked: false });
  const sw = page.getByRole('switch', { name: 'Enabled' });
  await sw.click();
  await expect.element(sw).toHaveAttribute('aria-checked', 'true');
  await sw.click();
  await expect.element(sw).toHaveAttribute('aria-checked', 'false');
});

test('Space toggles the switch via native button semantics', async () => {
  await render(Switch, { ariaLabel: 'Enabled', checked: false });
  const sw = page.getByRole('switch', { name: 'Enabled' });
  await sw.element().focus();
  await userEvent.keyboard(' ');
  await expect.element(sw).toHaveAttribute('aria-checked', 'true');
});
