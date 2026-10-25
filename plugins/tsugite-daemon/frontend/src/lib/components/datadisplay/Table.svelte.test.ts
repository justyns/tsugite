/// <reference types="vitest/browser" />
import { page, userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';
import { createRawSnippet } from 'svelte';
import { expect, test, vi } from 'vitest';
import GenericTable from './Table.svelte';
import type { TableColumn } from './Table.svelte';

type Row = { id: number; name: string; selected?: boolean; off?: boolean };
const Table = GenericTable<Row>;

const columns: TableColumn[] = [
  { key: 'name', label: 'name' },
  { key: 'state', label: 'state' },
  { key: 'updated', label: 'updated', sortable: true },
];

const data: Row[] = [
  { id: 1, name: 'usage-rollup' },
  { id: 2, name: 'nightly-backup', selected: true },
  { id: 3, name: 'model-cache-warm', off: true },
];

const cells = createRawSnippet((row: () => Row) => ({
  render: () => `<td>${row().name}</td>`,
}));

const rowProps = {
  rows: data,
  key: (r: Row) => r.id,
  cells,
  selected: (r: Row) => !!r.selected,
  off: (r: Row) => !!r.off,
};

test('renders a column header per column', async () => {
  await render(Table, { columns, ...rowProps });
  await expect.element(page.getByRole('columnheader', { name: 'name' })).toBeInTheDocument();
  await expect.element(page.getByRole('columnheader', { name: 'state' })).toBeInTheDocument();
  await expect.element(page.getByRole('columnheader', { name: 'updated' })).toBeInTheDocument();
});

test('only the sortable column gets aria-sort; others get none of it', async () => {
  await render(Table, { columns, ...rowProps, sort: { key: 'updated', dir: 'descending' } });
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
  await render(Table, { columns, ...rowProps, sort: null });
  await expect
    .element(page.getByRole('columnheader', { name: 'updated' }))
    .toHaveAttribute('aria-sort', 'none');
});

test('clicking a sortable header button reports the column key', async () => {
  const onSort = vi.fn();
  await render(Table, { columns, ...rowProps, onSort });
  await page.getByRole('button', { name: 'updated' }).click();
  expect(onSort).toHaveBeenCalledExactlyOnceWith('updated');
});

test('sortable header is keyboard-operable (native button semantics)', async () => {
  const onSort = vi.fn();
  await render(Table, { columns, ...rowProps, onSort });
  const btnEl = page.getByRole('button', { name: 'updated' }).element() as HTMLElement;
  btnEl.focus();
  await userEvent.keyboard('{Enter}');
  expect(onSort).toHaveBeenCalledExactlyOnceWith('updated');
});

test('non-sortable columns render as plain headers with no button', async () => {
  await render(Table, { columns, ...rowProps });
  await expect.element(page.getByRole('button', { name: 'name' })).not.toBeInTheDocument();
});

test('row state classes: selected and off rows carry their state marker classes', async () => {
  const { container } = await render(Table, { columns, ...rowProps });
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

test('renders the cells snippet per row and tags each row with its testid', async () => {
  const { container } = await render(Table, {
    columns,
    ...rowProps,
    rowTestid: (r: Row) => `row-${r.id}`,
  });
  await expect.element(page.getByRole('cell', { name: 'nightly-backup' })).toBeInTheDocument();
  expect(container.querySelector('[data-testid="row-3"]')?.textContent).toBe('model-cache-warm');
});

test('a hideLabel column keeps its header name for screen readers only', async () => {
  const { container } = await render(Table, {
    columns: [...columns, { key: 'actions', label: 'actions', hideLabel: true }],
    ...rowProps,
  });
  await expect.element(page.getByRole('columnheader', { name: 'actions' })).toBeInTheDocument();
  expect(container.querySelector('th .vh')?.textContent).toBe('actions');
});

test('a right-aligned column marks its header with the tr class', async () => {
  const { container } = await render(Table, {
    columns: [...columns, { key: 'cost', label: 'cost', align: 'right' }],
    ...rowProps,
  });
  const ths = Array.from(container.querySelectorAll('th'));
  expect(ths.filter((th) => th.classList.contains('tr')).map((th) => th.textContent)).toEqual([
    'cost',
  ]);
});
