/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import AnnThread from './AnnThread.svelte';

test('open thread fires reply and resolve', async () => {
  const onReply = vi.fn();
  const onResolve = vi.fn();
  render(AnnThread, {
    props: {
      author: 'you',
      anchor: 'cap total at 40 GB',
      when: '2m',
      body: 'pick one policy',
      status: 'open',
      onReply,
      onResolve,
    },
  });

  await page.getByRole('button', { name: 'Reply' }).click();
  await page.getByRole('button', { name: 'Resolve' }).click();
  expect(onReply).toHaveBeenCalledOnce();
  expect(onResolve).toHaveBeenCalledOnce();
});

test('editing saves the edited text', async () => {
  const onSave = vi.fn();
  render(AnnThread, {
    props: { author: 'you', when: 'now', body: 'original', status: 'editing', onSave },
  });

  const box = page.getByRole('textbox', { name: 'Edit annotation' });
  await box.fill('revised note');
  await page.getByRole('button', { name: 'Save' }).click();
  expect(onSave).toHaveBeenCalledWith('revised note');
});

test('resolved thread signals its state in text', async () => {
  render(AnnThread, {
    props: { author: 'ada', when: '1h', body: 'confirmed', status: 'resolved' },
  });

  await expect.element(page.getByText('Resolved', { exact: true })).toBeInTheDocument();
});
