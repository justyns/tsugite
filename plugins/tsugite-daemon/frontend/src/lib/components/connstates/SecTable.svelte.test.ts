/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import SecTable, { type SecretRow } from './SecTable.svelte';

const rows: SecretRow[] = [
  {
    name: 'OPENAI_API_KEY',
    provenance: 'process env',
    scope: 'env',
  },
  { name: 'TAILSCALE_AUTHKEY', provenance: 'process env', scope: 'env' },
];

test('rows render name and scope', async () => {
  await render(SecTable, { rows });
  await expect.element(page.getByText('OPENAI_API_KEY')).toBeInTheDocument();
  await expect.element(page.getByText('TAILSCALE_AUTHKEY')).toBeInTheDocument();
});

test('the value column never renders the actual secret, only a mask', async () => {
  const { container } = await render(SecTable, { rows });
  expect(container.textContent).not.toContain('sk-');
  expect(container.querySelectorAll('.sec-mask')).toHaveLength(2);
});

test('rotate calls onRotate with that row', async () => {
  const onRotate = vi.fn();
  await render(SecTable, { rows, onRotate });

  const rotateButtons = page.getByRole('button', { name: 'Rotate' });
  await rotateButtons.nth(0).click();

  expect(onRotate).toHaveBeenCalledExactlyOnceWith(rows[0]);
});

test('delete calls onDelete with that row', async () => {
  const onDelete = vi.fn();
  await render(SecTable, { rows, onDelete });

  await page.getByRole('button', { name: 'Delete TAILSCALE_AUTHKEY' }).click();

  expect(onDelete).toHaveBeenCalledExactlyOnceWith(rows[1]);
});
