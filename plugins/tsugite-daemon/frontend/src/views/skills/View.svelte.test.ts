/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), put: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
import View from './View.svelte';

afterEach(cleanup);

beforeEach(() => {
  vi.mocked(api.get).mockReset();
  agentsMeta.skillFiles = [];
  agentsMeta.skillIssues = [];
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

interface FileFixture {
  path: string;
  name: string;
  source: string;
  readonly: boolean;
  description: string;
}
interface IssueFixture {
  name: string | null;
  source: string;
  path: string;
  severity: string;
  message: string;
}

function mockCatalog(files: FileFixture[], issues: IssueFixture[] = []) {
  vi.mocked(api.get).mockImplementation(async (path: string) => {
    if (path === '/api/skill-files') return { files };
    if (path === '/api/skills/issues') return { issues };
    if (path.startsWith('/api/skill-files/content')) {
      const target = decodeURIComponent(path.split('path=')[1] ?? '');
      const file = files.find((f) => f.path === target);
      return {
        path: target,
        content: `---\nname: ${file?.name}\n---\nbody`,
        readonly: !!file?.readonly,
      };
    }
    throw new Error(`unexpected path: ${path}`);
  });
}

function file(overrides: Partial<FileFixture> = {}): FileFixture {
  return {
    path: '/skills/pdf-extract/SKILL.md',
    name: 'pdf-extract',
    source: 'project',
    readonly: false,
    description: 'tables + ocr from pdf attachments',
    ...overrides,
  };
}

test('shows a loading pane while the initial fetch is in flight', async () => {
  const gate = deferred<{ files: FileFixture[]; issues: IssueFixture[] }>();
  vi.mocked(api.get).mockReturnValue(gate.promise);
  const { container } = await render(View);
  await expect.element(page.getByText('Skills')).toBeInTheDocument();
  expect(container.querySelector('.t-skel')).not.toBeNull();
  gate.resolve({ files: [], issues: [] });
});

test('renders a truthful empty state when no skills are discovered', async () => {
  mockCatalog([]);
  render(View);
  await expect.element(page.getByText('No skills discovered')).toBeInTheDocument();
});

test('renders an error pane with retry on a fetch failure, and retry re-fetches', async () => {
  vi.mocked(api.get).mockRejectedValue(new Error('Bad Gateway'));
  render(View);
  await expect.element(page.getByText('Bad Gateway')).toBeInTheDocument();

  mockCatalog([file({ name: 'log-triage', path: '/a/SKILL.md' })]);
  await page.getByRole('button', { name: /retry/i }).click();
  await expect.element(page.getByRole('option', { name: /log-triage/ })).toBeInTheDocument();
});

test('joins skill-files and skills/issues into one catalog row with the right status pill', async () => {
  mockCatalog(
    [file({ name: 'pdf_extract', path: '/skills/pdf-extract/SKILL.md' })],
    [
      {
        name: 'pdf_extract',
        source: 'scan',
        path: '/skills/pdf-extract/SKILL.md',
        severity: 'warning',
        message: "missing 'description' field (recommended)",
      },
    ],
  );
  const { container } = await render(View);
  // The auto-selected detail pane carries the joined status pill.
  await expect.element(page.getByText('1 warning')).toBeInTheDocument();

  expect(container.querySelectorAll('.skl-row')).toHaveLength(1);
  // Exact text, not just toContain: a template reformat once collapsed the
  // space before the middot here ("1 skill· 1 with issues") and passed a
  // looser assertion - see skillCatalog.ts's catalogHeading() doc comment.
  expect(container.querySelector('.count')?.textContent).toBe('1 skill · 1 with issues');
});

test('an issue with no matching file row still surfaces as its own row, not dropped', async () => {
  mockCatalog(
    [],
    [
      {
        name: 'orphan-secret',
        source: 'scan',
        path: '/elsewhere/orphan-secret/SKILL.md',
        severity: 'warning',
        message: "missing 'description' field (recommended)",
      },
    ],
  );
  const { container } = await render(View);
  await expect.element(page.getByRole('option', { name: /orphan-secret/ })).toBeInTheDocument();
  expect(container.querySelectorAll('.skl-row')).toHaveLength(1);
  // Its detail pane explains the unreadable path instead of faking a source view.
  await expect.element(page.getByText(/diagnostic issue above/)).toBeInTheDocument();
});

test('never renders an enable/disable toggle (no such field exists in the backend)', async () => {
  mockCatalog([file()]);
  const { container } = await render(View);
  await expect.element(page.getByRole('option', { name: /pdf-extract/ })).toBeInTheDocument();
  expect(container.querySelector('[role="switch"]')).toBeNull();
  expect(page.getByRole('checkbox').elements()).toHaveLength(0);
});

test('search filters the visible roster by name/description/source', async () => {
  mockCatalog([
    file({ name: 'pdf-extract', path: '/a/SKILL.md', description: 'tables + ocr' }),
    file({
      name: 'web-search',
      path: '/b/SKILL.md',
      source: 'builtin',
      description: 'searx pipeline',
    }),
  ]);
  const { container } = await render(View);
  await expect.element(page.getByRole('option', { name: /web-search/ })).toBeInTheDocument();
  expect(container.querySelectorAll('.skl-row')).toHaveLength(2);

  await page.getByRole('searchbox', { name: 'Filter skills' }).fill('pdf');
  await expect.element(page.getByText('No skills match your filter')).not.toBeInTheDocument();
  expect(container.querySelectorAll('.skl-row')).toHaveLength(1);
  expect(container.querySelector('.skl-row')?.textContent).toContain('pdf-extract');
});

test('selecting a row shows its detail pane with diagnostics and the raw SKILL.md source', async () => {
  mockCatalog(
    [
      file({ name: 'log-triage', path: '/skills/log-triage/SKILL.md' }),
      file({ name: 'pdf_extract', path: '/skills/pdf-extract/SKILL.md' }),
    ],
    [
      {
        name: 'pdf_extract',
        source: 'scan',
        path: '/skills/pdf-extract/SKILL.md',
        severity: 'error',
        message: 'boom',
      },
    ],
  );
  await render(View);
  await expect.element(page.getByRole('option', { name: /pdf_extract/ })).toBeInTheDocument();

  await page.getByRole('option', { name: /pdf_extract/ }).click();
  const detail = page.getByTestId('skills-drawer');
  await expect.element(detail).toHaveTextContent('boom');
  // The source loads on select into a full-height pane - no extra click.
  await expect.element(detail).toHaveTextContent('name: pdf_extract');
  await expect.element(detail).toHaveTextContent('body');
});
