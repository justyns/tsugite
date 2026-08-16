/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import SpaceBar from './SpaceBar.svelte';

const TWO = [
  { id: 's1', name: 'Main' },
  { id: 's2', name: 'Notes' },
];

function props(over: Partial<Record<string, unknown>> = {}) {
  return {
    spaces: TWO,
    activeId: 's1',
    onSelect: vi.fn(),
    onAdd: vi.fn(),
    onRename: vi.fn(),
    onClose: vi.fn(),
    ...over,
  };
}

test('lists every space and marks the active one', async () => {
  await render(SpaceBar, props());
  await expect
    .element(page.getByRole('button', { name: 'Main', exact: true }))
    .toHaveAttribute('aria-pressed', 'true');
  await expect
    .element(page.getByRole('button', { name: 'Notes', exact: true }))
    .toHaveAttribute('aria-pressed', 'false');
});

test('clicking a space selects it', async () => {
  const p = props();
  await render(SpaceBar, p);
  await page.getByRole('button', { name: 'Notes', exact: true }).click();
  expect(p.onSelect).toHaveBeenCalledWith('s2');
});

test('the add control creates a space', async () => {
  const p = props();
  await render(SpaceBar, p);
  await page.getByRole('button', { name: 'New space' }).click();
  expect(p.onAdd).toHaveBeenCalledOnce();
});

test('the close control removes that space', async () => {
  const p = props();
  await render(SpaceBar, p);
  await page.getByRole('button', { name: 'Close Notes' }).click();
  expect(p.onClose).toHaveBeenCalledWith('s2');
});

test('closing is withheld while one space is left, since the store refuses it', async () => {
  await render(SpaceBar, props({ spaces: [{ id: 's1', name: 'Main' }] }));
  await expect.element(page.getByRole('button', { name: 'Close Main' })).not.toBeInTheDocument();
});

test('double-clicking a space renames it in place', async () => {
  const p = props();
  await render(SpaceBar, p);
  await page.getByRole('button', { name: 'Main', exact: true }).dblClick();
  const input = page.getByRole('textbox', { name: 'Rename space' });
  await input.fill('Planning');
  await input
    .element()
    .dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  expect(p.onRename).toHaveBeenCalledWith('s1', 'Planning');
});

test('Escape abandons a rename', async () => {
  const p = props();
  await render(SpaceBar, p);
  await page.getByRole('button', { name: 'Main', exact: true }).dblClick();
  const input = page.getByRole('textbox', { name: 'Rename space' });
  await input.fill('Planning');
  await input
    .element()
    .dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  expect(p.onRename).not.toHaveBeenCalled();
  await expect.element(page.getByRole('button', { name: 'Main', exact: true })).toBeInTheDocument();
});

test('an empty rename leaves the name alone', async () => {
  const p = props();
  await render(SpaceBar, p);
  await page.getByRole('button', { name: 'Main', exact: true }).dblClick();
  const input = page.getByRole('textbox', { name: 'Rename space' });
  await input.fill('   ');
  await input
    .element()
    .dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  expect(p.onRename).not.toHaveBeenCalled();
});

test('Delete on a focused space closes it', async () => {
  const p = props();
  const { container } = await render(SpaceBar, p);
  const chip = [...container.querySelectorAll<HTMLButtonElement>('button')].find(
    (b) => b.textContent?.trim() === 'Notes',
  );
  chip?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete', bubbles: true }));
  expect(p.onClose).toHaveBeenCalledWith('s2');
});

test('right-click opens a menu offering rename and close', async () => {
  const p = props();
  const { container } = await render(SpaceBar, p);
  const chip = [...container.querySelectorAll<HTMLButtonElement>('button')].find(
    (b) => b.textContent?.trim() === 'Notes',
  );
  chip?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, clientX: 20, clientY: 20 }));
  await expect.element(page.getByRole('menu', { name: 'Space actions' })).toBeInTheDocument();
  await page.getByRole('menuitem', { name: 'Close' }).click();
  expect(p.onClose).toHaveBeenCalledWith('s2');
});

function dragChip(from: HTMLElement, to: HTMLElement, half: 'left' | 'right') {
  const dataTransfer = new DataTransfer();
  from.dispatchEvent(new DragEvent('dragstart', { dataTransfer, bubbles: true }));
  const r = to.getBoundingClientRect();
  const clientX = half === 'left' ? r.left + r.width * 0.25 : r.left + r.width * 0.75;
  for (const type of ['dragover', 'drop']) {
    to.dispatchEvent(new DragEvent(type, { dataTransfer, bubbles: true, clientX }));
  }
}

test('dragging a chip onto the left half of an earlier one reorders it there', async () => {
  const onReorder = vi.fn();
  const { container } = await render(SpaceBar, {
    spaces: [
      { id: 'a', name: 'Main' },
      { id: 'b', name: 'Ops' },
      { id: 'c', name: 'Notes' },
    ],
    activeId: 'a',
    onSelect: () => {},
    onAdd: () => {},
    onRename: () => {},
    onClose: () => {},
    onReorder,
  });
  const chips = [...container.querySelectorAll<HTMLElement>('.sp')];
  dragChip(chips[2]!, chips[0]!, 'left');
  expect(onReorder).toHaveBeenCalledWith('c', 0);
});

test('dropping a chip back on itself reorders nothing', async () => {
  const onReorder = vi.fn();
  const { container } = await render(SpaceBar, {
    spaces: [
      { id: 'a', name: 'Main' },
      { id: 'b', name: 'Ops' },
    ],
    activeId: 'a',
    onSelect: () => {},
    onAdd: () => {},
    onRename: () => {},
    onClose: () => {},
    onReorder,
  });
  const chips = [...container.querySelectorAll<HTMLElement>('.sp')];
  dragChip(chips[1]!, chips[1]!, 'left');
  expect(onReorder).not.toHaveBeenCalled();
});
