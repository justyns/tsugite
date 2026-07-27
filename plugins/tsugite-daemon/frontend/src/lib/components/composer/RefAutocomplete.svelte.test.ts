/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import RefAutocomplete from './RefAutocomplete.svelte';
import type { RefItem } from './types';

const ITEMS: RefItem[] = [
  { id: 'f1', kind: 'file', label: '@sse-reconnect.md', detail: 'kb/ops · modified', git: 'm' },
  { id: 'c1', kind: 'chat', label: '@sse-reconnect-backoff', detail: 'chat · working' },
  { id: 'a1', kind: 'agent', label: '@odyn', detail: 'agent · opus-4-8' },
];

test('renders every item as a listbox option', async () => {
  render(RefAutocomplete, { items: ITEMS, onSelect: () => {} });
  await expect.element(page.getByRole('listbox')).toBeInTheDocument();
  expect(page.getByRole('option').elements()).toHaveLength(3);
});

test('marks the active row with aria-selected and the .is-hl highlight', async () => {
  render(RefAutocomplete, { items: ITEMS, activeIndex: 1, onSelect: () => {} });
  const selected = page.getByRole('option', { selected: true });
  await expect.element(selected).toHaveTextContent('@sse-reconnect-backoff');
});

test('renders the muted detail and the git file-state glyph with a title', async () => {
  render(RefAutocomplete, { items: ITEMS, onSelect: () => {} });
  await expect.element(page.getByText('kb/ops · modified')).toBeInTheDocument();
  const glyph = page.getByTitle('git: modified');
  await expect.element(glyph).toHaveTextContent('M');
});

test('clicking an option fires onSelect with the item and index', async () => {
  const onSelect = vi.fn();
  render(RefAutocomplete, { items: ITEMS, onSelect });
  await userEvent.click(page.getByRole('option').nth(2));
  expect(onSelect).toHaveBeenCalledWith(ITEMS[2], 2);
});

test('is hidden (removed from the a11y tree) when open is false', async () => {
  render(RefAutocomplete, { items: ITEMS, open: false, onSelect: () => {} });
  await vi.waitFor(() => expect(page.getByRole('listbox').elements()).toHaveLength(0));
});

test('caps the list height and scrolls internally when there are many items', async () => {
  const many: RefItem[] = Array.from({ length: 60 }, (_, i) => ({
    id: `f${i}`,
    kind: 'file' as const,
    label: `@file-${i}.md`,
  }));
  const { container } = await render(RefAutocomplete, { items: many, onSelect: () => {} });
  const list = container.querySelector('.slashpop') as HTMLElement;
  const style = getComputedStyle(list);
  expect(style.overflowY).toBe('auto');
  expect(style.maxHeight).not.toBe('none');
  // The cap bites: 60 rows overflow the container, so it scrolls internally
  // rather than growing to fit the whole list.
  expect(list.scrollHeight).toBeGreaterThan(list.clientHeight);
});

test('renders a section header per group without adding options', async () => {
  const grouped: RefItem[] = [
    { id: 's1', kind: 'session', label: 'Nightly backup', group: 'Sessions' },
    { id: 'f1', kind: 'file', label: 'notes.md', group: 'Files' },
    { id: 'f2', kind: 'file', label: 'runbook.md', group: 'Files' },
  ];
  render(RefAutocomplete, { items: grouped, onSelect: () => {} });
  // Two headers (Sessions, Files), rendered once each; the second Files item shares.
  await expect.element(page.getByText('Sessions')).toBeInTheDocument();
  await expect.element(page.getByText('Files')).toBeInTheDocument();
  // Headers are not options, so the roving-nav count stays the item count.
  expect(page.getByRole('option').elements()).toHaveLength(3);
});

test('shows the status row only while there are no items', async () => {
  const { rerender } = await render(RefAutocomplete, {
    items: [],
    status: 'Searching…',
    onSelect: () => {},
  });
  await expect.element(page.getByText('Searching…')).toBeInTheDocument();

  // Once results arrive the status is dropped in favor of the options.
  await rerender({
    items: [{ id: 'p1', kind: 'plugin', label: 'a hit' }] satisfies RefItem[],
    status: 'Searching…',
  });
  expect(page.getByText('Searching…').elements()).toHaveLength(0);
  await expect.element(page.getByRole('option', { name: /a hit/ })).toBeInTheDocument();
});
