/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { afterEach, expect, test, vi } from 'vitest';
import Chip from './Chip.svelte';

afterEach(cleanup);

const label = (text: string) => createRawSnippet(() => ({ render: () => `<span>${text}</span>` }));

test('a removable chip calls onRemove when its remove button is clicked', async () => {
  const onRemove = vi.fn();
  render(Chip, {
    removable: true,
    removeLabel: 'Remove sse-reconnect.md',
    onRemove,
    children: label('file: sse-reconnect.md'),
  });
  await page.getByRole('button', { name: 'Remove sse-reconnect.md' }).click();
  expect(onRemove).toHaveBeenCalledOnce();
});

test('a non-removable chip renders no button at all', async () => {
  render(Chip, { children: label('env') });
  await expect.element(page.getByRole('button')).not.toBeInTheDocument();
});
