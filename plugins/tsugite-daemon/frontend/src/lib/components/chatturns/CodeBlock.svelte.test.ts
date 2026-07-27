/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import CodeBlock from './CodeBlock.svelte';

const code = 'const a = 1\nconst b = 2';

test('copy invokes onCopy with the raw code text', async () => {
  const onCopy = vi.fn();
  render(CodeBlock, { code, lang: 'ts', filename: 'a.ts', onCopy });
  await page.getByRole('button', { name: 'Copy code' }).click();
  expect(onCopy).toHaveBeenCalledWith(code);
});

test('the collapse control toggles disclosure state and label', async () => {
  render(CodeBlock, { code, lang: 'ts', filename: 'a.ts' });
  const collapse = page.getByRole('button', { name: 'Collapse code' });
  await expect.element(collapse).toHaveAttribute('aria-expanded', 'true');
  await collapse.click();

  const expand = page.getByRole('button', { name: 'Expand code' });
  await expect.element(expand).toHaveAttribute('aria-expanded', 'false');
});

test('renders a real line count from the code', async () => {
  render(CodeBlock, { code, lang: 'ts', filename: 'a.ts' });
  await expect.element(page.getByText('2 lines')).toBeInTheDocument();
});

test('clicking the collapsed code peek expands it', async () => {
  const { container } = await render(CodeBlock, { code, lang: 'ts', collapsed: true });
  await expect
    .element(page.getByRole('button', { name: 'Expand code' }))
    .toHaveAttribute('aria-expanded', 'false');

  (container.querySelector('.pre-expand') as HTMLElement).click();
  await expect
    .element(page.getByRole('button', { name: 'Collapse code' }))
    .toHaveAttribute('aria-expanded', 'true');
  // Expanded: the overlay is gone so the code text stays selectable.
  expect(container.querySelector('.pre-expand')).toBeNull();
});

test('individual tool calls render as exec disclosure rows with their own output', async () => {
  render(CodeBlock, {
    code: 'run("ls")\nread_note("x")',
    lang: 'python',
    calls: [
      { tool: 'run', status: 'done', output: 'file-a\nfile-b' },
      { tool: 'read_note', status: 'error', output: 'no such note' },
    ],
  });
  const runRow = page.getByRole('button', { name: /^run/ });
  await expect.element(runRow).toBeInTheDocument();
  await expect.element(page.getByText('exit 1')).toBeInTheDocument();

  await runRow.click();
  await expect.element(page.getByText('file-a file-b')).toBeInTheDocument();
});

test('auto-collapses when the run finishes; a manual toggle wins until the prop flips', async () => {
  const { rerender } = await render(CodeBlock, {
    code,
    lang: 'python',
    running: true,
    collapsed: false,
  });
  await expect
    .element(page.getByRole('button', { name: 'Collapse code' }))
    .toHaveAttribute('aria-expanded', 'true');
  await expect.element(page.getByText('running')).toBeInTheDocument();

  await rerender({ running: false, collapsed: true });
  const expand = page.getByRole('button', { name: 'Expand code' });
  await expect.element(expand).toHaveAttribute('aria-expanded', 'false');

  await expand.click();
  await expect
    .element(page.getByRole('button', { name: 'Collapse code' }))
    .toHaveAttribute('aria-expanded', 'true');
});

test('a long result expands on click and collapses back via the strip', async () => {
  const long = Array.from({ length: 40 }, (_, i) => `line ${i}`).join('\n');
  render(CodeBlock, { code: 'run()', lang: 'python', output: long });
  const toggle = page.getByRole('button', { name: 'expand output' });
  await expect.element(toggle).toBeInTheDocument();

  await toggle.click();
  await expect.element(page.getByRole('button', { name: 'collapse output' })).toBeInTheDocument();

  await page.getByRole('button', { name: 'collapse output' }).click();
  await expect.element(page.getByRole('button', { name: 'expand output' })).toBeInTheDocument();
});

test('a short result offers no expand affordance', async () => {
  render(CodeBlock, { code: 'run()', lang: 'python', output: 'ok' });
  await expect.element(page.getByText('ok')).toBeInTheDocument();
  expect(document.querySelector('.out-toggle')).toBeNull();
});
