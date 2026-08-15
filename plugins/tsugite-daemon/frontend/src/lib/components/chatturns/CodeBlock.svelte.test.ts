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

test('a tsu_group renders its own calls under a label, leaving ungrouped ones loose', async () => {
  const { container } = await render(CodeBlock, {
    code,
    calls: [
      { tool: 'read_file', status: 'done' as const },
      { tool: 'http_request', status: 'done' as const, groupId: 'g1' },
    ],
    groups: [{ id: 'g1', title: 'fetch open issues', meta: '412ms' }],
  });

  const group = container.querySelector('.t-code-group')!;
  await expect.element(page.getByText('fetch open issues')).toBeInTheDocument();
  expect(group.textContent).toContain('http_request');
  expect(group.textContent).not.toContain('read_file');
});

test('a failed group shows a failed marker and its error text', async () => {
  const { container } = await render(CodeBlock, {
    code,
    calls: [{ tool: 'http_request', status: 'error' as const, groupId: 'g1' }],
    groups: [{ id: 'g1', title: 'fetch open issues', success: false, error: 'HTTPError: 401' }],
  });

  expect(container.querySelector('.t-code-group')!.classList).toContain('is-err');
  await expect.element(page.getByText('failed')).toBeInTheDocument();
  await expect.element(page.getByText('HTTPError: 401')).toBeInTheDocument();
});

test('a group renders where it ran, not after every loose call', async () => {
  const { container } = await render(CodeBlock, {
    code,
    calls: [
      { tool: 'before', status: 'done' as const },
      { tool: 'inside', status: 'done' as const, groupId: 'g1' },
      { tool: 'after', status: 'done' as const },
    ],
    groups: [{ id: 'g1', title: 'the middle' }],
  });

  const order = [...container.querySelectorAll('.t-code-calls > *')].map((el) =>
    el.classList.contains('t-code-group') ? 'GROUP' : el.textContent!.trim().split(/\s+/)[0],
  );
  expect(order).toEqual(['before', 'GROUP', 'after']);
});

test('a group that wrapped no tool calls still shows its heading', async () => {
  await render(CodeBlock, { code, calls: [], groups: [{ id: 'g1', title: 'crunch numbers' }] });
  await expect.element(page.getByText('crunch numbers')).toBeInTheDocument();
});

test('a call naming an unknown group is still rendered', async () => {
  await render(CodeBlock, {
    code,
    calls: [{ tool: 'orphan_call', status: 'done' as const, groupId: 'ghost' }],
    groups: [],
  });
  await expect.element(page.getByText('orphan_call')).toBeInTheDocument();
});

test('a model-authored group title is escaped, never markup', async () => {
  const { container } = await render(CodeBlock, {
    code,
    calls: [],
    groups: [{ id: 'g1', title: '<img src=x onerror="alert(1)">' }],
  });
  expect(container.querySelector('.grp-title')!.textContent).toBe('<img src=x onerror="alert(1)">');
  expect(container.querySelector('img')).toBeNull();
});

test('a finished block folds its tool calls and output away', async () => {
  const { container } = await render(CodeBlock, {
    code,
    collapsed: true,
    calls: [{ tool: 'read_file', status: 'done' as const }],
    output: 'lots of output',
  });
  // By container, not by text: the folded summary names the tool too.
  expect(container.querySelector('.t-code-calls')).not.toBeVisible();
  await expect.element(page.getByText('lots of output')).not.toBeVisible();
});

test('a running block keeps its calls on screen', async () => {
  await render(CodeBlock, {
    code,
    running: true,
    collapsed: false,
    calls: [{ tool: 'read_file', status: 'running' as const }],
  });
  await expect.element(page.getByText('read_file')).toBeVisible();
});

test('a block whose call failed stays open even when told to collapse', async () => {
  await render(CodeBlock, {
    code,
    collapsed: true,
    calls: [{ tool: 'http_request', status: 'error' as const }],
  });
  await expect.element(page.getByText('http_request')).toBeVisible();
});

test('a block whose group failed stays open even when told to collapse', async () => {
  await render(CodeBlock, {
    code,
    collapsed: true,
    calls: [{ tool: 'http_request', status: 'done' as const, groupId: 'g1' }],
    groups: [{ id: 'g1', title: 'fetch', success: false, error: 'HTTPError: 401' }],
  });
  await expect.element(page.getByText('HTTPError: 401')).toBeVisible();
});

test('the folded header says how many tools ran', async () => {
  await render(CodeBlock, {
    code,
    collapsed: true,
    calls: [
      { tool: 'read_file', status: 'done' as const },
      { tool: 'write_file', status: 'done' as const },
    ],
  });
  await expect.element(page.getByText('2 tools')).toBeVisible();
});

test('a folded block clips its code instead of showing every line', async () => {
  const { container } = await render(CodeBlock, {
    code: 'one = 1\ntwo = 2\nthree = 3\nfour = 4',
    lang: 'python',
    collapsed: true,
  });

  // Height in px is not comparable here: the app's border-box reset is absent in
  // this environment, so the peek's exact size is verified against a live daemon.
  const pre = container.querySelector('.t-code.is-collapsed pre') as HTMLElement;
  expect(pre.scrollHeight).toBeGreaterThan(pre.clientHeight);
});

test('a folded block summarises its groups instead of peeking at code', async () => {
  const { container } = await render(CodeBlock, {
    code: 'read_file(path="a")\nread_file(path="b")',
    lang: 'python',
    collapsed: true,
    calls: [{ tool: 'read_file', status: 'done' as const, groupId: 'g1' }],
    groups: [
      { id: 'g1', title: 'read the docs' },
      { id: 'g2', title: 'crunch numbers' },
    ],
  });

  await expect.element(page.getByText('read the docs · crunch numbers')).toBeVisible();
  expect(container.querySelector('.pre-wrap')).toBeNull();
});

test('expanding a summarised block reveals the code again', async () => {
  const { container } = await render(CodeBlock, {
    code: 'read_file(path="a")',
    lang: 'python',
    collapsed: true,
    groups: [{ id: 'g1', title: 'read the docs' }],
  });
  expect(container.querySelector('.pre-wrap')).toBeNull();

  await page.getByRole('button', { name: 'Expand code' }).click();
  expect(container.querySelector('.pre-wrap')).not.toBeNull();
});

test('a folded block with no groups summarises its tool names', async () => {
  const { container } = await render(CodeBlock, {
    code: 'read_file(path="a")',
    lang: 'python',
    collapsed: true,
    calls: [
      { tool: 'read_file', status: 'done' as const },
      { tool: 'read_file', status: 'done' as const },
      { tool: 'write_file', status: 'done' as const },
    ],
  });

  // Repeats collapse: three calls, two distinct tools.
  await expect.element(page.getByText('read_file · write_file')).toBeVisible();
  expect(container.querySelector('.pre-wrap')).toBeNull();
});

test('a folded block with neither groups nor calls peeks at its code', async () => {
  const { container } = await render(CodeBlock, {
    code: 'total = 1 + 1',
    lang: 'python',
    collapsed: true,
  });
  expect(container.querySelector('.pre-wrap')).not.toBeNull();
});
