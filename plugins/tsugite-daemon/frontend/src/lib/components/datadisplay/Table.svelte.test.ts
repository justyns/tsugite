/// <reference types="vitest/browser" />
import { page, userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { expect, test, vi } from 'vitest';
import Table from './Table.svelte';
import type { TableColumn, TableRow } from './Table.svelte';

const columns: TableColumn[] = [
  { key: 'name', label: 'name' },
  { key: 'state', label: 'state' },
  { key: 'updated', label: 'updated', sortable: true },
];

const rows: TableRow[] = [
  {
    id: 1,
    cells: [
      { content: 'usage-rollup' },
      { content: 'done' },
      { content: '2m', tone: 'c3', mono: true },
    ],
  },
  {
    id: 2,
    selected: true,
    cells: [
      { content: 'nightly-backup' },
      { content: 'errored' },
      { content: '6h', tone: 'c3', mono: true },
    ],
  },
  {
    id: 3,
    off: true,
    cells: [
      { content: 'model-cache-warm' },
      { content: 'off' },
      { content: '3d', tone: 'c3', mono: true },
    ],
  },
];

test('renders a column header per column', async () => {
  await render(Table, { columns, rows });
  await expect.element(page.getByRole('columnheader', { name: 'name' })).toBeInTheDocument();
  await expect.element(page.getByRole('columnheader', { name: 'state' })).toBeInTheDocument();
  await expect.element(page.getByRole('columnheader', { name: 'updated' })).toBeInTheDocument();
});

test('only the sortable column gets aria-sort; others get none of it', async () => {
  await render(Table, { columns, rows, sort: { key: 'updated', dir: 'descending' } });
  await expect
    .element(page.getByRole('columnheader', { name: 'updated' }))
    .toHaveAttribute('aria-sort', 'descending');
  await expect
    .element(page.getByRole('columnheader', { name: 'name' }))
    .not.toHaveAttribute('aria-sort');
  await expect
    .element(page.getByRole('columnheader', { name: 'state' }))
    .not.toHaveAttribute('aria-sort');
});

test('a sortable column not currently sorted reports aria-sort="none"', async () => {
  await render(Table, { columns, rows, sort: null });
  await expect
    .element(page.getByRole('columnheader', { name: 'updated' }))
    .toHaveAttribute('aria-sort', 'none');
});

test('clicking a sortable header button reports the column key', async () => {
  const onSort = vi.fn();
  await render(Table, { columns, rows, onSort });
  await page.getByRole('button', { name: 'updated' }).click();
  expect(onSort).toHaveBeenCalledExactlyOnceWith('updated');
});

test('sortable header is keyboard-operable (native button semantics)', async () => {
  const onSort = vi.fn();
  await render(Table, { columns, rows, onSort });
  const btnEl = page.getByRole('button', { name: 'updated' }).element() as HTMLElement;
  btnEl.focus();
  await userEvent.keyboard('{Enter}');
  expect(onSort).toHaveBeenCalledExactlyOnceWith('updated');
});

test('non-sortable columns render as plain headers with no button', async () => {
  await render(Table, { columns, rows });
  await expect.element(page.getByRole('button', { name: 'name' })).not.toBeInTheDocument();
});

test('row state classes: selected and off rows carry their state marker classes', async () => {
  const { container } = await render(Table, { columns, rows });
  const trs = container.querySelectorAll('tbody tr');
  expect(trs).toHaveLength(3);

  const [normalRow, selectedRow, offRow] = Array.from(trs) as HTMLElement[];
  expect(normalRow!.className).not.toContain('is-selected');
  expect(normalRow!.hasAttribute('aria-selected')).toBe(false);

  expect(selectedRow!.className).toContain('is-selected');
  expect(selectedRow!.getAttribute('aria-selected')).toBe('true');

  expect(offRow!.className).toContain('is-off');
  expect(offRow!.getAttribute('aria-disabled')).toBe('true');
});

test('cell content accepts a snippet for rich content like a pill', async () => {
  const pillSnippet = createRawSnippet(() => ({
    render: () => `<span class="t-pill" data-st="errored">errored</span>`,
  }));
  const richRows: TableRow[] = [
    {
      id: 1,
      cells: [{ content: 'nightly-backup' }, { content: pillSnippet }],
    },
  ];
  await render(Table, {
    columns: [
      { key: 'name', label: 'name' },
      { key: 'state', label: 'state' },
    ],
    rows: richRows,
  });
  await expect.element(page.getByText('errored')).toBeInTheDocument();
});
