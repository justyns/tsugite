/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import HookGroup from './HookGroup.svelte';

const hooks = ['lint', 'fmt', 'notify'].map((n) => ({
  kind: 'exec' as const,
  command: `hook:${n}`,
  status: 'done' as const,
  exitCode: 0,
  output: `${n} output`,
  meta: '10ms',
}));

test('folded row expands to each hook with its output', async () => {
  render(HookGroup, { hooks, meta: '30ms' });
  const header = page.getByRole('button', { name: /3 hooks/ });
  await expect.element(header).toHaveAttribute('aria-expanded', 'false');
  await expect.element(page.getByText('hook:lint')).not.toBeInTheDocument();
  await header.click();
  await expect.element(header).toHaveAttribute('aria-expanded', 'true');
  for (const n of ['lint', 'fmt', 'notify']) {
    await expect.element(page.getByText(`hook:${n}`)).toBeInTheDocument();
    await expect.element(page.getByText(`${n} output`)).toBeInTheDocument();
  }
});
