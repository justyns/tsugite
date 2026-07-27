/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { TESTID } from '$lib/testids';
import { secrets } from '$lib/stores/secrets.svelte';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import View from './View.svelte';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), post: vi.fn(), del: vi.fn() },
}));

import { api } from '$lib/api/client';

const addDialog = () => page.getByRole('dialog', { name: 'Add secret' });
const rotateDialog = (name: string) => page.getByRole('dialog', { name: `Rotate ${name}` });

afterEach(cleanup);

beforeEach(() => {
  secrets.names = [];
  secrets.loading = false;
  secrets.error = null;
  toasts.items = [];
  vi.mocked(api.get).mockReset();
  vi.mocked(api.post).mockReset();
  vi.mocked(api.del).mockReset();
});

test('shows a loading pane before the initial fetch resolves, then the table', async () => {
  let resolveGet!: (value: { secrets: string[] }) => void;
  vi.mocked(api.get).mockReturnValue(
    new Promise((resolve) => {
      resolveGet = resolve;
    }),
  );

  const { container } = await render(View);
  expect(container.querySelector('[aria-busy="true"]')).not.toBeNull();

  resolveGet({ secrets: ['OPENAI_API_KEY'] });
  await expect.element(page.getByText('OPENAI_API_KEY')).toBeInTheDocument();
  expect(container.querySelector('[aria-busy="true"]')).toBeNull();
});

test('shows an error pane with the message on a failed fetch; retry re-fetches', async () => {
  vi.mocked(api.get).mockRejectedValueOnce(new Error('network down'));
  await render(View);

  await expect.element(page.getByText('network down')).toBeInTheDocument();

  vi.mocked(api.get).mockResolvedValueOnce({ secrets: ['A_SECRET'] });
  await page.getByRole('button', { name: 'Retry' }).click();

  await expect.element(page.getByText('A_SECRET')).toBeInTheDocument();
});

test('shows capability messaging in the empty state, no fake unlock gate', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: [] });
  await render(View);

  await expect.element(page.getByText('No secrets stored')).toBeInTheDocument();
  // The empty state must not claim the store is "locked" - no such concept
  // exists server-side.
  await expect.element(page.getByText('locked', { exact: false })).not.toBeInTheDocument();
  await expect.element(page.getByLabelText('passphrase')).not.toBeInTheDocument();
});

test('renders a populated table with names visible and values always masked', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: ['OPENAI_API_KEY', 'TAILSCALE_AUTHKEY'] });
  const { container } = await render(View);

  await expect.element(page.getByText('OPENAI_API_KEY')).toBeInTheDocument();
  await expect.element(page.getByText('TAILSCALE_AUTHKEY')).toBeInTheDocument();
  expect(container.querySelectorAll('.sec-mask')).toHaveLength(2);
  expect(container.textContent).not.toContain('sk-');
});

test('rows carry no reference-tracking chatter', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: ['OPENAI_API_KEY'] });
  const { container } = await render(View);

  await expect.element(page.getByText('OPENAI_API_KEY')).toBeInTheDocument();
  expect(container.textContent).not.toMatch(/unused|referenced/i);
});

test('add: saves via the upsert endpoint and lists the new name', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: [] });
  vi.mocked(api.post).mockResolvedValueOnce({ status: 'ok', name: 'NEW_SECRET' });
  await render(View);
  await expect.element(page.getByText('No secrets stored')).toBeInTheDocument();

  await page.getByTestId(TESTID.secretsAdd).click();
  await addDialog().getByLabelText('name').fill('NEW_SECRET');
  await addDialog().getByLabelText('value').fill('sk-abc');
  await addDialog().getByRole('button', { name: 'Add secret' }).click();

  expect(api.post).toHaveBeenCalledExactlyOnceWith('/api/secrets/NEW_SECRET', { value: 'sk-abc' });
  await expect.element(page.getByText('NEW_SECRET')).toBeInTheDocument();
  await expect.element(addDialog()).not.toBeInTheDocument();
});

test('rotate: name is locked to the row and upserts the same name', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: ['OPENAI_API_KEY'] });
  vi.mocked(api.post).mockResolvedValueOnce({ status: 'ok', name: 'OPENAI_API_KEY' });
  await render(View);
  await expect.element(page.getByText('OPENAI_API_KEY')).toBeInTheDocument();

  await page.getByRole('button', { name: 'Rotate' }).click();
  const dialog = rotateDialog('OPENAI_API_KEY');
  await expect.element(dialog.getByLabelText('name')).toHaveValue('OPENAI_API_KEY');
  await dialog.getByLabelText('value').fill('sk-new-value');
  await dialog.getByRole('button', { name: 'Rotate value' }).click();

  expect(api.post).toHaveBeenCalledExactlyOnceWith('/api/secrets/OPENAI_API_KEY', {
    value: 'sk-new-value',
  });
});

test('delete: destructive confirm removes the row via DELETE', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: ['TAILSCALE_AUTHKEY'] });
  vi.mocked(api.del).mockResolvedValueOnce({ status: 'ok', name: 'TAILSCALE_AUTHKEY' });
  await render(View);
  await expect.element(page.getByText('TAILSCALE_AUTHKEY')).toBeInTheDocument();

  await page.getByRole('button', { name: 'Delete TAILSCALE_AUTHKEY' }).click();
  const dialog = page.getByRole('dialog', { name: 'Delete TAILSCALE_AUTHKEY?' });
  await expect.element(dialog).toBeInTheDocument();
  await dialog.getByRole('button', { name: 'Delete secret' }).click();

  expect(api.del).toHaveBeenCalledExactlyOnceWith('/api/secrets/TAILSCALE_AUTHKEY');
  await expect.element(page.getByText('No secrets stored')).toBeInTheDocument();
});

test('delete: cancel leaves the row untouched and never calls DELETE', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: ['TAILSCALE_AUTHKEY'] });
  await render(View);

  await page.getByRole('button', { name: 'Delete TAILSCALE_AUTHKEY' }).click();
  const dialog = page.getByRole('dialog', { name: 'Delete TAILSCALE_AUTHKEY?' });
  await dialog.getByRole('button', { name: 'Cancel' }).click();

  expect(api.del).not.toHaveBeenCalled();
  await expect.element(page.getByText('TAILSCALE_AUTHKEY')).toBeInTheDocument();
});

test('save failure keeps the modal open, keeps typed input, and toasts the real backend error', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: [] });
  const err = Object.assign(new Error('backend does not support writing'), { status: 400 });
  vi.mocked(api.post).mockRejectedValueOnce(err);
  await render(View);
  await expect.element(page.getByText('No secrets stored')).toBeInTheDocument();

  await page.getByTestId(TESTID.secretsAdd).click();
  await addDialog().getByLabelText('name').fill('X');
  await addDialog().getByLabelText('value').fill('v');
  await addDialog().getByRole('button', { name: 'Add secret' }).click();

  await expect.element(addDialog().getByLabelText('name')).toHaveValue('X');
  expect(
    toasts.items.some((t) => t.variant === 'err' && t.body === 'backend does not support writing'),
  ).toBe(true);
});

test('cancel add: closes without calling POST', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ secrets: [] });
  await render(View);
  await expect.element(page.getByText('No secrets stored')).toBeInTheDocument();

  await page.getByTestId(TESTID.secretsAdd).click();
  await addDialog().getByRole('button', { name: 'Cancel' }).click();

  expect(api.post).not.toHaveBeenCalled();
  await expect.element(addDialog()).not.toBeInTheDocument();
});
