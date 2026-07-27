/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import VerdictBar from './VerdictBar.svelte';

test('pending state exposes both review actions', async () => {
  const onApprove = vi.fn();
  const onRequestChanges = vi.fn();
  render(VerdictBar, { props: { state: 'pending', onApprove, onRequestChanges } });

  await page.getByRole('button', { name: 'Request changes' }).click();
  await page.getByRole('button', { name: 'Approve plan' }).click();

  expect(onRequestChanges).toHaveBeenCalledOnce();
  expect(onApprove).toHaveBeenCalledOnce();
});

test('resolved verdict is announced and hides the actions', async () => {
  render(VerdictBar, { props: { state: 'approved' } });

  await expect.element(page.getByRole('status')).toHaveTextContent('Approved');
  await expect.element(page.getByRole('button', { name: 'Approve plan' })).not.toBeInTheDocument();
});
