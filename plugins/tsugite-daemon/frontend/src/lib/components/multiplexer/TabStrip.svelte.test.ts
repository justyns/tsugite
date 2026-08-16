/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import TabStrip, { type PaneTab } from './TabStrip.svelte';

const tabs: PaneTab[] = [
  { id: 'a', label: 'sse backoff', state: 'busy' },
  { id: 'b', label: 'backup prune', state: 'blocked' },
  { id: 'c', label: 'nightly term', state: 'idle' },
];

test('renders a tab per entry plus the new-tab button', async () => {
  render(TabStrip, { tabs, activeId: 'a', onNew: () => {} });
  await expect.element(page.getByRole('tab', { name: /sse backoff/ })).toBeInTheDocument();
  await expect.element(page.getByRole('tab', { name: /backup prune/ })).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: 'New tab' })).toBeInTheDocument();
  // active tab carries aria-selected
  await expect
    .element(page.getByRole('tab', { name: /sse backoff/ }))
    .toHaveAttribute('aria-selected', 'true');
});

test('clicking a tab selects it', async () => {
  const onSelect = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onSelect });
  await userEvent.click(page.getByRole('tab', { name: /backup prune/ }));
  expect(onSelect).toHaveBeenCalledWith('b');
});

test('ArrowRight moves roving focus to the next tab and selects it', async () => {
  const onSelect = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onSelect });
  await userEvent.click(page.getByRole('tab', { name: /sse backoff/ }));
  onSelect.mockClear();
  await userEvent.keyboard('{ArrowRight}');
  expect(onSelect).toHaveBeenCalledWith('b');
  await expect.element(page.getByRole('tab', { name: /backup prune/ })).toHaveFocus();
});

test('ArrowLeft from the first tab wraps to the last', async () => {
  const onSelect = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onSelect });
  await userEvent.click(page.getByRole('tab', { name: /sse backoff/ }));
  onSelect.mockClear();
  await userEvent.keyboard('{ArrowLeft}');
  expect(onSelect).toHaveBeenCalledWith('c');
  await expect.element(page.getByRole('tab', { name: /nightly term/ })).toHaveFocus();
});

test('End selects the last tab, Home the first', async () => {
  const onSelect = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onSelect });
  await userEvent.click(page.getByRole('tab', { name: /sse backoff/ }));
  onSelect.mockClear();
  await userEvent.keyboard('{End}');
  expect(onSelect).toHaveBeenLastCalledWith('c');
  await userEvent.keyboard('{Home}');
  expect(onSelect).toHaveBeenLastCalledWith('a');
});

test('Delete on a focused closable tab requests close', async () => {
  const onClose = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onClose });
  await userEvent.click(page.getByRole('tab', { name: /sse backoff/ }));
  await userEvent.keyboard('{Delete}');
  expect(onClose).toHaveBeenCalledWith('a');
});

test('a non-closable tab exposes no close control and ignores Delete', async () => {
  const onClose = vi.fn();
  const pinned: PaneTab[] = [{ id: 'a', label: 'pinned', state: 'busy', closable: false }];
  render(TabStrip, { tabs: pinned, activeId: 'a', onClose });
  await expect.element(page.getByRole('button', { name: /Close pinned/ })).not.toBeInTheDocument();
  await userEvent.click(page.getByRole('tab', { name: /pinned/ }));
  await userEvent.keyboard('{Delete}');
  expect(onClose).not.toHaveBeenCalled();
});

test('clicking the new-tab button fires onNew', async () => {
  const onNew = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onNew });
  await userEvent.click(page.getByRole('button', { name: 'New tab' }));
  expect(onNew).toHaveBeenCalledOnce();
});

test('right-clicking a tab opens the context menu with close-others/all wired', async () => {
  const onClose = vi.fn();
  const onCloseOthers = vi.fn();
  const onCloseAll = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onClose, onCloseOthers, onCloseAll });

  await page.getByRole('tab', { name: /sse backoff/ }).click({ button: 'right' });
  const menu = page.getByRole('menu', { name: 'Tab actions' });
  await expect.element(menu).toBeInTheDocument();

  await page.getByRole('menuitem', { name: 'Close others' }).click();
  expect(onCloseOthers).toHaveBeenCalledWith('a');
  await expect.element(menu).not.toBeInTheDocument();

  await page.getByRole('tab', { name: /sse backoff/ }).click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Close all' }).click();
  expect(onCloseAll).toHaveBeenCalledOnce();
});

test('the split affordance renders at the strip end when wired', async () => {
  const onSplit = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onSplit });
  await userEvent.click(page.getByRole('button', { name: 'Split pane' }));
  expect(onSplit).toHaveBeenCalledOnce();
});

test('an ephemeral tab renders as an italic preview; a pinned tab does not', async () => {
  const preview: PaneTab[] = [
    { id: 'a', label: 'preview.md', ephemeral: true },
    { id: 'b', label: 'pinned.md' },
  ];
  render(TabStrip, { tabs: preview, activeId: 'a' });
  await expect.element(page.getByRole('tab', { name: /preview\.md/ })).toHaveClass('is-preview');
  await expect.element(page.getByRole('tab', { name: /pinned\.md/ })).not.toHaveClass('is-preview');
});

test('double-clicking a tab pins it (onPin)', async () => {
  const onPin = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onPin });
  await userEvent.dblClick(page.getByRole('tab', { name: /sse backoff/ }));
  expect(onPin).toHaveBeenCalledWith('a');
});

test('middle-click closes a tab (but not an unclosable one)', async () => {
  const onClose = vi.fn();
  render(TabStrip, {
    tabs: [
      { id: 'a', label: 'sse backoff' },
      { id: 'b', label: 'pinned home', closable: false },
    ],
    activeId: 'a',
    onClose,
  });
  await page.getByRole('tab', { name: /sse backoff/ }).click({ button: 'middle' });
  expect(onClose).toHaveBeenCalledWith('a');

  await page.getByRole('tab', { name: /pinned home/ }).click({ button: 'middle' });
  expect(onClose).toHaveBeenCalledTimes(1);
});

function dragTabOnto(from: HTMLElement, to: HTMLElement, half: 'left' | 'right') {
  const dataTransfer = new DataTransfer();
  from.dispatchEvent(new DragEvent('dragstart', { dataTransfer, bubbles: true }));
  const r = to.getBoundingClientRect();
  const clientX = half === 'left' ? r.left + r.width * 0.25 : r.left + r.width * 0.75;
  for (const type of ['dragover', 'drop']) {
    to.dispatchEvent(new DragEvent(type, { dataTransfer, bubbles: true, clientX }));
  }
}

async function tabEl(name: RegExp): Promise<HTMLElement> {
  return (await page.getByRole('tab', { name }).element()) as HTMLElement;
}

test('dragging a tab onto the left half of an earlier tab reorders it there', async () => {
  const onReorder = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onReorder });

  dragTabOnto(await tabEl(/nightly term/), await tabEl(/sse backoff/), 'left');

  expect(onReorder).toHaveBeenCalledWith('c', 0);
});

test('dragging onto the right half of a tab lands after it', async () => {
  const onReorder = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onReorder });

  dragTabOnto(await tabEl(/sse backoff/), await tabEl(/backup prune/), 'right');

  expect(onReorder).toHaveBeenCalledWith('a', 2);
});

test('dropping a tab back onto itself reorders nothing', async () => {
  const onReorder = vi.fn();
  render(TabStrip, { tabs, activeId: 'a', onReorder });

  const el = await tabEl(/backup prune/);
  dragTabOnto(el, el, 'left');

  expect(onReorder).not.toHaveBeenCalled();
});
