import { expect, test } from 'vitest';
import { type LeafNode, defaultLayout, dockAsTab, splitPane, collectLeaves } from './layout';
import { placeTree } from './place';

function nested() {
  const l = defaultLayout();
  const base = dockAsTab(l, l.root.id, { kind: 'chat' });
  const row = splitPane(base, base.root.id, 'row', { kind: 'artifact' });
  const [, b] = collectLeaves(row.root);
  return splitPane(row, b!.id, 'col', { kind: 'terminal' });
}

test('a lone leaf fills the pane area with no dividers', () => {
  const root: LeafNode = { type: 'leaf', id: 'p', tabs: [], activeTabId: null };
  expect(placeTree(root)).toEqual({
    panes: [{ leaf: root, x: 0, y: 0, w: 1, h: 1 }],
    dividers: [],
  });
});

test('a nested split places panes in tree order inside its parent cell', () => {
  const { panes, dividers } = placeTree(nested().root);
  expect(panes.map((p) => [p.x, p.y, p.w, p.h])).toEqual([
    [0, 0, 0.5, 1],
    [0.5, 0, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5],
  ]);
  expect(dividers.map((d) => [d.dir, d.at, d.span, d.pct])).toEqual([
    ['row', 0.5, 1, 50],
    ['col', 0.5, 1, 50],
  ]);
  expect(dividers[1]!.cross).toEqual({ x: 0.5, y: 0, w: 0.5, h: 1 });
});
