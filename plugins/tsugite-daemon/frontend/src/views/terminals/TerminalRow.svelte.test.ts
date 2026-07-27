/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import TerminalRow from './TerminalRow.svelte';
import { readSurfaceDrag } from '$lib/shell/mux/drag';
import type { Terminal } from '$lib/stores/terminals.svelte';

afterEach(cleanup);

const NOW = Date.parse('2026-07-14T10:01:30.000Z');

function term(overrides: Partial<Terminal> = {}): Terminal {
  return {
    id: 'term-abc123',
    cmd: 'npm test --watch',
    cwd: null,
    state: 'running',
    pid: 4210,
    exit_code: null,
    created_at: '2026-07-14T10:00:00.000Z',
    updated_at: '2026-07-14T10:01:00.000Z',
    resolved_at: null,
    bytes_out: 2048,
    lines_out: 642,
    last_line: 'PASS reconnect.spec.ts',
    parent_session_id: null,
    truncated: false,
    ...overrides,
  };
}

test('renders command, line count, elapsed, and last-line preview', async () => {
  render(TerminalRow, { term: term(), st: 'running', now: NOW });
  await expect.element(page.getByText('npm test --watch')).toBeInTheDocument();
  await expect.element(page.getByText('642 ln')).toBeInTheDocument();
  await expect.element(page.getByText('01:30')).toBeInTheDocument(); // created -> now
  await expect.element(page.getByText('PASS reconnect.spec.ts')).toBeInTheDocument();
});

test('falls back to a placeholder when there is no output yet', async () => {
  render(TerminalRow, { term: term({ last_line: '' }), st: 'starting', now: NOW });
  await expect.element(page.getByText('no output yet')).toBeInTheDocument();
});

test('clicking the row selects it', async () => {
  const onSelect = vi.fn();
  render(TerminalRow, { term: term(), st: 'running', now: NOW, onSelect });
  await page.getByRole('option').click();
  expect(onSelect).toHaveBeenCalledOnce();
});

test('is a mux drag source carrying a {kind:terminal} surface ref', async () => {
  render(TerminalRow, { term: term(), st: 'running', now: NOW });
  const el = page.getByRole('option').element();

  const dt = new DataTransfer();
  el.dispatchEvent(new DragEvent('dragstart', { dataTransfer: dt, bubbles: true }));

  const ref = readSurfaceDrag(dt);
  expect(ref).toMatchObject({
    kind: 'terminal',
    params: { id: 'term-abc123' },
    title: 'npm test --watch',
    state: 'busy', // running -> busy tab dot
  });
});

test('a stream-lost terminal flags attention on the row', async () => {
  render(TerminalRow, { term: term(), st: 'stream_lost', now: NOW });
  const el = page.getByRole('option').element();
  expect(el.classList.contains('is-attn')).toBe(true);
});
