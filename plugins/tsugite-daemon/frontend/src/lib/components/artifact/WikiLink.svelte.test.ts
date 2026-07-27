/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import WikiLink from './WikiLink.svelte';

test('a missing target carries a non-colour "missing page" cue in its name', async () => {
  render(WikiLink, { props: { page: 'household-systems', missing: true } });

  // accessible name includes the missing marker so state is not colour-only
  await expect
    .element(page.getByRole('button', { name: /household-systems.*missing page/i }))
    .toBeInTheDocument();
});

test('navigating fires onNavigate with the page name', async () => {
  const onNavigate = vi.fn();
  render(WikiLink, { props: { page: 'backup-retention', onNavigate } });

  await page.getByRole('button', { name: /backup-retention/ }).click();
  expect(onNavigate).toHaveBeenCalledWith('backup-retention');
});
