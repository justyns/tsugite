/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { expect, test } from 'vitest';
import Badge from './Badge.svelte';

const count = (text: string) => createRawSnippet(() => ({ render: () => `<span>${text}</span>` }));

test('a labelled badge exposes its name to the accessibility tree', async () => {
  await render(Badge, { label: '3 jobs running', children: count('3') });
  await expect.element(page.getByRole('img', { name: '3 jobs running' })).toBeInTheDocument();
});

test('an unlabelled badge exposes no role to the accessibility tree', async () => {
  await render(Badge, { children: count('12') });
  expect(page.getByRole('img').elements()).toHaveLength(0);
});
