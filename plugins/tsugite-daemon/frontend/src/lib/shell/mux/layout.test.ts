import { describe, expect, test } from 'vitest';
import {
  LAYOUT_SCHEMA_VERSION,
  collectLeaves,
  closeTab,
  cycleTab,
  deserializeLayout,
  dockAsTab,
  defaultLayout,
  findLeaf,
  focusPane,
  moveTab,
  closeAllTabs,
  closeOtherTabs,
  openInPane,
  openPreview,
  pinTab,
  retargetOrOpen,
  retitleTab,
  resizeSplit,
  selectTab,
  serializeLayout,
  splitPane,
  surfaceKey,
  type LeafNode,
  type SplitNode,
} from './layout';

// A leaf is the only node with tabs; a split holds >=2 children. These narrow
// helpers keep the assertions readable and fail loudly on a wrong node type.
function asSplit(node: unknown): SplitNode {
  const n = node as SplitNode;
  expect(n.type).toBe('split');
  return n;
}
function asLeaf(node: unknown): LeafNode {
  const n = node as LeafNode;
  expect(n.type).toBe('leaf');
  return n;
}

/** A one-pane layout holding a single docked surface, for seeding split/close cases. */
function seeded(ref: Parameters<typeof dockAsTab>[2] = { kind: 'chat' }) {
  const l = defaultLayout();
  return dockAsTab(l, l.root.id, ref);
}

describe('defaultLayout', () => {
  test('is a single empty leaf, focused, at the current schema version', () => {
    const l = defaultLayout();
    expect(l.version).toBe(LAYOUT_SCHEMA_VERSION);
    const root = asLeaf(l.root);
    expect(root.tabs).toEqual([]);
    expect(root.activeTabId).toBeNull();
    expect(l.focusedPaneId).toBe(root.id);
  });

  test('two calls do not share node identity (no accidental aliasing)', () => {
    const a = defaultLayout();
    const b = defaultLayout();
    expect(a.root.id).not.toBe(b.root.id);
  });
});

describe('surfaceKey', () => {
  test('is stable regardless of param insertion order', () => {
    expect(surfaceKey({ kind: 'chat', params: { a: '1', b: '2' } })).toBe(
      surfaceKey({ kind: 'chat', params: { b: '2', a: '1' } }),
    );
  });
  test('distinguishes kind and params', () => {
    expect(surfaceKey({ kind: 'chat', params: { id: '1' } })).not.toBe(
      surfaceKey({ kind: 'terminal', params: { id: '1' } }),
    );
    expect(surfaceKey({ kind: 'chat', params: { id: '1' } })).not.toBe(
      surfaceKey({ kind: 'chat', params: { id: '2' } }),
    );
  });
});

describe('dockAsTab', () => {
  test('adds a tab to the pane and makes it active', () => {
    const l0 = defaultLayout();
    const pid = l0.root.id;
    const l1 = dockAsTab(l0, pid, { kind: 'chat', params: { id: 'sse' }, title: 'sse' });
    const leaf = asLeaf(l1.root);
    expect(leaf.tabs).toHaveLength(1);
    expect(leaf.tabs[0]!.kind).toBe('chat');
    expect(leaf.tabs[0]!.title).toBe('sse');
    expect(leaf.activeTabId).toBe(leaf.tabs[0]!.id);
  });

  test('does not mutate the input layout (pure)', () => {
    const l0 = defaultLayout();
    dockAsTab(l0, l0.root.id, { kind: 'chat' });
    expect(asLeaf(l0.root).tabs).toHaveLength(0);
  });

  test('docking the same surface twice in one pane activates the existing tab, no duplicate', () => {
    const l0 = defaultLayout();
    const pid = l0.root.id;
    const l1 = dockAsTab(l0, pid, { kind: 'chat', params: { id: 'sse' } });
    const firstId = asLeaf(l1.root).tabs[0]!.id;
    const l2 = dockAsTab(l1, pid, { kind: 'chat', params: { id: 'sse' } });
    const leaf = asLeaf(l2.root);
    expect(leaf.tabs).toHaveLength(1);
    expect(leaf.activeTabId).toBe(firstId);
  });

  test('a no-op on an unknown pane id returns an equivalent layout', () => {
    const l0 = dockAsTab(defaultLayout(), 'nope', { kind: 'chat' });
    expect(asLeaf(l0.root).tabs).toHaveLength(0);
  });
});

describe('openInPane', () => {
  test('opens into the focused pane when no target is given', () => {
    const l0 = defaultLayout();
    const l1 = openInPane(l0, { kind: 'chat', params: { id: 'a' } });
    expect(asLeaf(l1.root).tabs).toHaveLength(1);
  });

  test('honours an explicit target pane over the focused one', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = splitPane(l, l.root.id, 'row', { kind: 'terminal', params: { id: 't' } });
    const [left, right] = collectLeaves(l.root);
    // focus is on the new (right) leaf; explicitly target the left one
    const l2 = openInPane(l, { kind: 'file', params: { id: 'f' } }, left!.id);
    expect(findLeaf(l2, left!.id)!.tabs.some((t) => t.kind === 'file')).toBe(true);
    expect(findLeaf(l2, right!.id)!.tabs.some((t) => t.kind === 'file')).toBe(false);
  });
});

describe('closeOtherTabs / closeAllTabs', () => {
  test('closeOtherTabs keeps only the target tab, active and focused', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = dockAsTab(l, l.root.id, { kind: 'file', params: { id: 'f' } });
    l = dockAsTab(l, l.root.id, { kind: 'terminal', params: { id: 't' } });
    const keep = asLeaf(l.root).tabs[1]!;
    const l1 = closeOtherTabs(l, l.root.id, keep.id);
    const leaf = asLeaf(l1.root);
    expect(leaf.tabs.map((t) => t.id)).toEqual([keep.id]);
    expect(leaf.activeTabId).toBe(keep.id);
  });

  test('closeAllTabs empties a root pane but keeps it; a split member collapses', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = dockAsTab(l, l.root.id, { kind: 'file', params: { id: 'f' } });
    const l1 = closeAllTabs(l, l.root.id);
    expect(asLeaf(l1.root).tabs).toHaveLength(0);

    let split = seeded({ kind: 'chat', params: { id: 'a' } });
    split = splitPane(split, split.root.id, 'row', { kind: 'terminal', params: { id: 't' } });
    const [left, right] = collectLeaves(split.root);
    const l2 = closeAllTabs(split, right!.id);
    const leaves = collectLeaves(l2.root);
    expect(leaves).toHaveLength(1);
    expect(leaves[0]!.id).toBe(left!.id);
  });
});

describe('retargetOrOpen', () => {
  test("retargets the focused pane's existing tab of the kind instead of adding one", () => {
    const l0 = seeded({ kind: 'chat', params: { sessionId: 'a' }, title: 'A' });
    const l1 = retargetOrOpen(l0, { kind: 'chat', params: { sessionId: 'b' }, title: 'B' });
    const leaf = asLeaf(l1.root);
    expect(leaf.tabs).toHaveLength(1);
    expect(leaf.tabs[0]!.params).toEqual({ sessionId: 'b' });
    expect(leaf.tabs[0]!.title).toBe('B');
    expect(leaf.activeTabId).toBe(leaf.tabs[0]!.id);
  });

  test('keeps the tab id stable across a retarget (the surface instance survives)', () => {
    const l0 = seeded({ kind: 'chat', params: { sessionId: 'a' } });
    const before = asLeaf(l0.root).tabs[0]!.id;
    const l1 = retargetOrOpen(l0, { kind: 'chat', params: { sessionId: 'b' } });
    expect(asLeaf(l1.root).tabs[0]!.id).toBe(before);
  });

  test('prefers the focused pane, then falls back to any pane holding the kind', () => {
    let l = seeded({ kind: 'chat', params: { sessionId: 'a' } });
    l = splitPane(l, l.root.id, 'row', { kind: 'terminal', params: { id: 't' } });
    const [left, right] = collectLeaves(l.root);
    // Focus sits on the right (terminal-only) pane; the chat lives on the left.
    const l1 = retargetOrOpen(l, { kind: 'chat', params: { sessionId: 'b' } });
    const leftAfter = findLeaf(l1, left!.id)!;
    expect(leftAfter.tabs).toHaveLength(1);
    expect(leftAfter.tabs[0]!.params).toEqual({ sessionId: 'b' });
    expect(l1.focusedPaneId).toBe(left!.id);
    expect(findLeaf(l1, right!.id)!.tabs.some((t) => t.kind === 'chat')).toBe(false);
  });

  test('opens a new tab when no pane holds the kind', () => {
    const l0 = seeded({ kind: 'terminal', params: { id: 't' } });
    const l1 = retargetOrOpen(l0, { kind: 'chat', params: { sessionId: 'a' } });
    expect(asLeaf(l1.root).tabs.map((t) => t.kind)).toEqual(['terminal', 'chat']);
  });
});

describe('openPreview / pinTab', () => {
  test('opens a file into a single ephemeral tab, active and focused', () => {
    const l1 = openPreview(defaultLayout(), {
      kind: 'file',
      params: { path: 'a.md' },
      title: 'a.md',
    });
    const leaf = asLeaf(l1.root);
    expect(leaf.tabs).toHaveLength(1);
    expect(leaf.tabs[0]!.ephemeral).toBe(true);
    expect(leaf.tabs[0]!.params).toEqual({ path: 'a.md' });
    expect(leaf.activeTabId).toBe(leaf.tabs[0]!.id);
    expect(l1.focusedPaneId).toBe(leaf.id);
  });

  test('a second single-click reuses the same preview tab in place (B replaces A, id stable)', () => {
    let l = openPreview(defaultLayout(), { kind: 'file', params: { path: 'a.md' }, title: 'a.md' });
    const previewId = asLeaf(l.root).tabs[0]!.id;
    l = openPreview(l, { kind: 'file', params: { path: 'b.md' }, title: 'b.md' });
    const leaf = asLeaf(l.root);
    expect(leaf.tabs).toHaveLength(1);
    expect(leaf.tabs[0]!.id).toBe(previewId);
    expect(leaf.tabs[0]!.params).toEqual({ path: 'b.md' });
    expect(leaf.tabs[0]!.title).toBe('b.md');
    expect(leaf.tabs[0]!.ephemeral).toBe(true);
  });

  test('pinTab clears the ephemeral flag', () => {
    const l0 = openPreview(defaultLayout(), { kind: 'file', params: { path: 'a.md' } });
    const l1 = pinTab(l0, l0.root.id, asLeaf(l0.root).tabs[0]!.id);
    expect(asLeaf(l1.root).tabs[0]!.ephemeral).toBeUndefined();
  });

  test('after a pin, the next preview opens a SECOND tab and leaves the pinned one intact', () => {
    let l = openPreview(defaultLayout(), { kind: 'file', params: { path: 'a.md' }, title: 'a.md' });
    const pinnedId = asLeaf(l.root).tabs[0]!.id;
    l = pinTab(l, l.root.id, pinnedId);
    l = openPreview(l, { kind: 'file', params: { path: 'b.md' }, title: 'b.md' });
    const leaf = asLeaf(l.root);
    expect(leaf.tabs).toHaveLength(2);
    const pinned = leaf.tabs.find((t) => t.id === pinnedId)!;
    expect(pinned.params).toEqual({ path: 'a.md' });
    expect(pinned.ephemeral).toBeUndefined();
    const preview = leaf.tabs.find((t) => t.id !== pinnedId)!;
    expect(preview.ephemeral).toBe(true);
    expect(preview.params).toEqual({ path: 'b.md' });
    expect(leaf.activeTabId).toBe(preview.id);
  });

  test('clicking an already-pinned file just activates it, no duplicate tab', () => {
    let l = openPreview(defaultLayout(), { kind: 'file', params: { path: 'a.md' }, title: 'a.md' });
    const pinnedId = asLeaf(l.root).tabs[0]!.id;
    l = pinTab(l, l.root.id, pinnedId);
    // A preview of a different file so the pinned tab is not the active one.
    l = openPreview(l, { kind: 'file', params: { path: 'b.md' }, title: 'b.md' });
    expect(asLeaf(l.root).tabs).toHaveLength(2);
    // Re-clicking the pinned A activates it without spawning a preview.
    l = openPreview(l, { kind: 'file', params: { path: 'a.md' }, title: 'a.md' });
    const leaf = asLeaf(l.root);
    expect(leaf.tabs).toHaveLength(2);
    expect(leaf.activeTabId).toBe(pinnedId);
    // The ephemeral B is untouched (the preview slot was not reused for A).
    expect(leaf.tabs.find((t) => t.id !== pinnedId)!.ephemeral).toBe(true);
  });

  test('opens the preview into an explicit target pane, which becomes focused', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = splitPane(l, l.root.id, 'row', { kind: 'terminal', params: { id: 't' } });
    const [left, right] = collectLeaves(l.root);
    const l2 = openPreview(l, { kind: 'file', params: { path: 'a.md' } }, left!.id);
    expect(findLeaf(l2, left!.id)!.tabs.some((t) => t.kind === 'file' && t.ephemeral)).toBe(true);
    expect(findLeaf(l2, right!.id)!.tabs.some((t) => t.kind === 'file')).toBe(false);
    expect(l2.focusedPaneId).toBe(left!.id);
  });

  test('does not mutate the input layout (pure)', () => {
    const l0 = defaultLayout();
    openPreview(l0, { kind: 'file', params: { path: 'a.md' } });
    expect(asLeaf(l0.root).tabs).toHaveLength(0);
  });

  test('pinTab is a no-op on a missing pane or tab', () => {
    const l0 = openPreview(defaultLayout(), { kind: 'file', params: { path: 'a.md' } });
    const tabId = asLeaf(l0.root).tabs[0]!.id;
    expect(asLeaf(pinTab(l0, 'ghost', tabId).root).tabs[0]!.ephemeral).toBe(true);
    expect(asLeaf(pinTab(l0, l0.root.id, 'ghost').root).tabs[0]!.ephemeral).toBe(true);
  });

  test('an ephemeral tab survives a serialize/deserialize round-trip', () => {
    const l = openPreview(defaultLayout(), {
      kind: 'file',
      params: { path: 'a.md' },
      title: 'a.md',
    });
    const restored = deserializeLayout(serializeLayout(l));
    expect(asLeaf(restored.root).tabs[0]!.ephemeral).toBe(true);
  });
});

describe('splitPane', () => {
  test('splitting the root leaf row-wise yields a 2-child row split with even sizes', () => {
    const seed = seeded({ kind: 'chat' });
    const l1 = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' });
    const split = asSplit(l1.root);
    expect(split.dir).toBe('row');
    expect(split.children).toHaveLength(2);
    expect(split.sizes).toEqual([0.5, 0.5]);
    expect(split.sizes.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  test('the newly opened pane becomes focused', () => {
    const seed = seeded({ kind: 'chat' });
    const l1 = splitPane(seed, seed.root.id, 'row', { kind: 'terminal', params: { id: 't' } });
    const focused = findLeaf(l1, l1.focusedPaneId!)!;
    expect(focused.tabs[0]!.kind).toBe('terminal');
  });

  test("position 'before' places the new pane first", () => {
    const seed = seeded({ kind: 'chat' });
    const l1 = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' }, 'before');
    const split = asSplit(l1.root);
    expect(asLeaf(split.children[0]).tabs[0]!.kind).toBe('terminal');
    expect(asLeaf(split.children[1]).tabs[0]!.kind).toBe('chat');
  });

  test('splitting a pane in the SAME direction as its parent inserts a flat sibling', () => {
    const seed = seeded({ kind: 'chat' });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' }); // row[chat, term]
    const rightId = collectLeaves(l.root)[1]!.id;
    l = splitPane(l, rightId, 'row', { kind: 'file' }); // still one row split, 3 children
    const split = asSplit(l.root);
    expect(split.dir).toBe('row');
    expect(split.children).toHaveLength(3);
    expect(split.children.every((c) => c.type === 'leaf')).toBe(true);
    expect(split.sizes).toHaveLength(3);
    expect(split.sizes.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  test('splitting a pane in a DIFFERENT direction nests a new split (nested splits)', () => {
    const seed = seeded({ kind: 'chat' });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' }); // row[chat, term]
    const rightId = collectLeaves(l.root)[1]!.id;
    l = splitPane(l, rightId, 'col', { kind: 'file' }); // row[chat, col[term, file]]
    const rootSplit = asSplit(l.root);
    expect(rootSplit.dir).toBe('row');
    expect(rootSplit.children).toHaveLength(2);
    const nested = asSplit(rootSplit.children[1]);
    expect(nested.dir).toBe('col');
    expect(nested.children).toHaveLength(2);
    expect(collectLeaves(l.root)).toHaveLength(3);
  });
});

describe('closeTab', () => {
  test('removing a non-last tab keeps the pane and activates a neighbour', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id: 'b' } });
    const leaf0 = asLeaf(l.root);
    const firstId = leaf0.tabs[0]!.id;
    l = closeTab(l, l.root.id, leaf0.activeTabId!); // close active (b)
    const leaf = asLeaf(l.root);
    expect(leaf.tabs).toHaveLength(1);
    expect(leaf.activeTabId).toBe(firstId);
  });

  test('closing the last tab of the ONLY pane leaves an empty root leaf (root preserved)', () => {
    let l = seeded({ kind: 'chat' });
    l = closeTab(l, l.root.id, asLeaf(l.root).tabs[0]!.id);
    const leaf = asLeaf(l.root);
    expect(leaf.tabs).toEqual([]);
    expect(leaf.activeTabId).toBeNull();
    expect(l.focusedPaneId).toBe(leaf.id);
  });

  test('closing the last tab of a split pane collapses the split (orphan prevention)', () => {
    const seed = seeded({ kind: 'chat' });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' }); // row[chat, term]
    const [leftLeaf, rightLeaf] = collectLeaves(l.root);
    l = closeTab(l, rightLeaf!.id, findLeaf(l, rightLeaf!.id)!.tabs[0]!.id);
    // the split is gone; the surviving chat leaf is hoisted to the root
    const root = asLeaf(l.root);
    expect(root.tabs[0]!.kind).toBe('chat');
    expect(l.focusedPaneId).toBe(leftLeaf!.id);
  });

  test('deeply nested last-tab close collapses every now-single-child split', () => {
    const seed = seeded({ kind: 'chat' });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' }); // row[chat, term]
    const rightId = collectLeaves(l.root)[1]!.id;
    l = splitPane(l, rightId, 'col', { kind: 'file' }); // row[chat, col[term, file]]
    const fileLeaf = collectLeaves(l.root).find((x) => x.tabs[0]!.kind === 'file')!;
    l = closeTab(l, fileLeaf.id, fileLeaf.tabs[0]!.id);
    // col split had two children, now one -> collapses; row now has [chat, term]
    const rootSplit = asSplit(l.root);
    expect(rootSplit.children).toHaveLength(2);
    expect(rootSplit.children.every((c) => c.type === 'leaf')).toBe(true);
    expect(collectLeaves(l.root).map((x) => x.tabs[0]!.kind)).toEqual(['chat', 'terminal']);
  });
});

describe('moveTab', () => {
  test('moving the only tab out of a pane collapses the source and lands in the target', () => {
    const seed = seeded({ kind: 'chat', params: { id: 'a' } });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal', params: { id: 't' } }); // row[chat, term]
    const [left, right] = collectLeaves(l.root);
    const termTabId = findLeaf(l, right!.id)!.tabs[0]!.id;
    l = moveTab(l, right!.id, termTabId, left!.id);
    // source collapsed -> single leaf holding both tabs
    const root = asLeaf(l.root);
    expect(root.tabs.map((t) => t.kind).sort()).toEqual(['chat', 'terminal']);
    expect(root.id).toBe(left!.id);
  });

  test('moving within the same pane reorders the tab', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id: 'b' } });
    l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id: 'c' } });
    const leaf = asLeaf(l.root);
    const cId = leaf.tabs[2]!.id;
    l = moveTab(l, leaf.id, cId, leaf.id, 0); // move c to front
    expect(asLeaf(l.root).tabs.map((t) => t.params.id)).toEqual(['c', 'a', 'b']);
  });

  test('a forward move lands where the insertion index points', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    for (const id of ['b', 'c', 'd']) l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id } });
    const leaf = asLeaf(l.root);
    l = moveTab(l, leaf.id, leaf.tabs[0]!.id, leaf.id, 2);
    expect(asLeaf(l.root).tabs.map((t) => t.params.id)).toEqual(['b', 'a', 'c', 'd']);
  });
});

describe('resizeSplit', () => {
  test('shifts weight between the two panes around a divider, total preserved', () => {
    const seed = seeded({ kind: 'chat' });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' });
    const split = asSplit(l.root);
    l = resizeSplit(l, split.id, 0, 0.2);
    const after = asSplit(l.root);
    expect(after.sizes[0]).toBeCloseTo(0.7);
    expect(after.sizes[1]).toBeCloseTo(0.3);
    expect(after.sizes[0]! + after.sizes[1]!).toBeCloseTo(1);
  });

  test('clamps so neither pane collapses below the minimum', () => {
    const seed = seeded({ kind: 'chat' });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' });
    const split = asSplit(l.root);
    l = resizeSplit(l, split.id, 0, 5); // absurd delta
    const after = asSplit(l.root);
    expect(after.sizes[0]!).toBeLessThan(1);
    expect(after.sizes[1]!).toBeGreaterThan(0);
    expect(after.sizes[0]! + after.sizes[1]!).toBeCloseTo(1);
  });

  // Repeatedly splitting the same pane halves its slot with no floor, so an
  // adjacent pair can sum below 2*MIN_SIZE. A nudge there must still leave BOTH
  // weights strictly positive - the old clamp yielded a negative weight, which
  // the deserializer then rejected, wiping the whole layout on reload.
  function overSplit(times: number): ReturnType<typeof defaultLayout> {
    let l = seeded({ kind: 'chat', params: { id: 'keep' } });
    for (let i = 0; i < times; i++) {
      const first = collectLeaves(l.root)[0]!;
      l = splitPane(l, first.id, 'row', { kind: 'terminal', params: { id: `t${i}` } }, 'before');
    }
    return l;
  }

  test('never yields a zero/negative weight when the adjacent pair sums below 2*MIN_SIZE', () => {
    let l = overSplit(6);
    const split = asSplit(l.root);
    expect(split.sizes[0]! + split.sizes[1]!).toBeLessThan(0.05); // pair below MIN_SIZE
    l = resizeSplit(l, split.id, 0, 0.02);
    const after = asSplit(l.root);
    expect(after.sizes.every((s) => s > 0)).toBe(true);
    expect(after.sizes.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  test('a zero-delta nudge on a tiny pair is a lossless no-op (no drift into negatives)', () => {
    let l = overSplit(6);
    const split = asSplit(l.root);
    const before = [...split.sizes];
    l = resizeSplit(l, split.id, 0, 0);
    expect(asSplit(l.root).sizes.every((s) => s > 0)).toBe(true);
    expect(asSplit(l.root).sizes.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    // the pair may re-even, but the total and every sibling stay intact
    expect(asSplit(l.root).sizes.slice(2)).toEqual(before.slice(2));
  });

  test('an over-split layout survives its own persistence round-trip (no silent wipe)', () => {
    let l = overSplit(6);
    const split = asSplit(l.root);
    l = resizeSplit(l, split.id, 0, 0.02);
    expect(collectLeaves(l.root)).toHaveLength(7);
    const restored = deserializeLayout(serializeLayout(l));
    // The live layout must not fail its own deserializer back to a blank pane.
    expect(collectLeaves(restored.root)).toHaveLength(7);
  });
});

describe('focusPane / selectTab', () => {
  test('focusPane moves focus only to an existing pane', () => {
    const seed = seeded({ kind: 'chat' });
    const l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' });
    const leftId = collectLeaves(l.root)[0]!.id;
    expect(focusPane(l, leftId).focusedPaneId).toBe(leftId);
    expect(focusPane(l, 'ghost').focusedPaneId).toBe(l.focusedPaneId);
  });

  test('selectTab activates a tab within its pane', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id: 'b' } });
    const firstId = asLeaf(l.root).tabs[0]!.id;
    l = selectTab(l, l.root.id, firstId);
    expect(asLeaf(l.root).activeTabId).toBe(firstId);
  });
});

describe('cycleTab', () => {
  /** The `id` param of a pane's active tab, for terse assertions. */
  function activeParam(layout: ReturnType<typeof defaultLayout>, paneId: string): string {
    const leaf = findLeaf(layout, paneId)!;
    return leaf.tabs.find((t) => t.id === leaf.activeTabId)!.params.id!;
  }
  /** A single pane holding three chat tabs a,b,c (c active, being last docked). */
  function threeTabs() {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id: 'b' } });
    l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id: 'c' } });
    return l;
  }

  test('advances / retreats the focused pane active tab, clamped at both ends', () => {
    const l = threeTabs();
    const pid = l.root.id;
    // From the last tab, forward clamps.
    expect(activeParam(cycleTab(l, 1), pid)).toBe('c');
    // Walk back to the first, then clamp.
    let p = cycleTab(l, -1);
    expect(activeParam(p, pid)).toBe('b');
    p = cycleTab(p, -1);
    expect(activeParam(p, pid)).toBe('a');
    p = cycleTab(p, -1);
    expect(activeParam(p, pid)).toBe('a');
    // And forward again to the end.
    p = cycleTab(p, 1);
    expect(activeParam(p, pid)).toBe('b');
    p = cycleTab(p, 1);
    expect(activeParam(p, pid)).toBe('c');
    p = cycleTab(p, 1);
    expect(activeParam(p, pid)).toBe('c');
  });

  test('does not mutate the input layout (pure)', () => {
    const l = threeTabs();
    const before = asLeaf(l.root).activeTabId;
    cycleTab(l, -1);
    expect(asLeaf(l.root).activeTabId).toBe(before);
  });

  test('is a no-op on a single-tab pane', () => {
    const l = seeded({ kind: 'chat', params: { id: 'solo' } });
    const active = asLeaf(l.root).activeTabId;
    expect(asLeaf(cycleTab(l, 1).root).activeTabId).toBe(active);
    expect(asLeaf(cycleTab(l, -1).root).activeTabId).toBe(active);
  });

  test('cycles only the focused pane, leaving the others untouched', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id: 'b' } }); // left: a,b (b active)
    l = splitPane(l, l.root.id, 'row', { kind: 'terminal', params: { id: 't1' } });
    const [leftId, rightId] = collectLeaves(l.root).map((x) => x.id);
    l = dockAsTab(l, rightId!, { kind: 'terminal', params: { id: 't2' } }); // right: t1,t2 (t2 active, focused)

    const afterRight = cycleTab(l, -1);
    expect(activeParam(afterRight, rightId!)).toBe('t1');
    expect(activeParam(afterRight, leftId!)).toBe('b');

    const afterLeft = cycleTab(focusPane(l, leftId!), -1);
    expect(activeParam(afterLeft, leftId!)).toBe('a');
    expect(activeParam(afterLeft, rightId!)).toBe('t2');
  });

  test('falls back to the first leaf when no pane is focused', () => {
    let l = seeded({ kind: 'chat', params: { id: 'a' } });
    l = dockAsTab(l, l.root.id, { kind: 'chat', params: { id: 'b' } }); // a,b (b active)
    const noFocus = { ...l, focusedPaneId: null };
    expect(activeParam(cycleTab(noFocus, -1), l.root.id)).toBe('a');
  });
});

describe('serialize / deserialize', () => {
  test('round-trips a nested layout exactly', () => {
    const seed = seeded({ kind: 'chat' });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' });
    const rightId = collectLeaves(l.root)[1]!.id;
    l = splitPane(l, rightId, 'col', { kind: 'file' });
    const restored = deserializeLayout(serializeLayout(l));
    expect(restored).toEqual(l);
  });

  test.each([
    ['not json at all', 'not-json'],
    ['null input', null],
    [
      'wrong schema version',
      JSON.stringify({
        version: 999,
        root: { type: 'leaf', id: 'x', tabs: [], activeTabId: null },
        focusedPaneId: 'x',
      }),
    ],
    [
      'split with a single child (orphan)',
      JSON.stringify({
        version: LAYOUT_SCHEMA_VERSION,
        root: {
          type: 'split',
          id: 's',
          dir: 'row',
          children: [{ type: 'leaf', id: 'x', tabs: [], activeTabId: null }],
          sizes: [1],
        },
        focusedPaneId: 'x',
      }),
    ],
    [
      'sizes length mismatch',
      JSON.stringify({
        version: LAYOUT_SCHEMA_VERSION,
        root: {
          type: 'split',
          id: 's',
          dir: 'row',
          children: [
            { type: 'leaf', id: 'x', tabs: [], activeTabId: null },
            { type: 'leaf', id: 'y', tabs: [], activeTabId: null },
          ],
          sizes: [1],
        },
        focusedPaneId: 'x',
      }),
    ],
    [
      'unknown node type',
      JSON.stringify({
        version: LAYOUT_SCHEMA_VERSION,
        root: { type: 'weird' },
        focusedPaneId: null,
      }),
    ],
    [
      'activeTabId pointing at nothing',
      JSON.stringify({
        version: LAYOUT_SCHEMA_VERSION,
        root: { type: 'leaf', id: 'x', tabs: [], activeTabId: 'ghost' },
        focusedPaneId: 'x',
      }),
    ],
  ])('falls back to the default layout on corruption: %s', (_label, raw) => {
    const l = deserializeLayout(raw);
    expect(l.version).toBe(LAYOUT_SCHEMA_VERSION);
    expect(l.root.type).toBe('leaf');
    expect(asLeaf(l.root).tabs).toEqual([]);
  });

  test('a valid single-leaf payload survives deserialization', () => {
    const l = seeded({ kind: 'chat', params: { id: 'a' } });
    const restored = deserializeLayout(serializeLayout(l));
    expect(asLeaf(restored.root).tabs[0]!.kind).toBe('chat');
  });

  test('self-heals a pathologically deep tree to the default instead of throwing (boot guard)', () => {
    // A JSON-parseable but absurdly nested layout used to stack-overflow the
    // unguarded validation walk, white-screening the app at store construction.
    // Built as text on purpose: materializing the tree and JSON.stringify-ing it
    // recurses outside the guard under test, so the test itself overflowed at
    // this depth on some runs, and the allocation starved sibling workers into
    // hook timeouts.
    const DEPTH = 60000;
    const leaf = '{"type":"leaf","id":"leaf-x","tabs":[],"activeTabId":null}';
    const raw =
      `{"version":${LAYOUT_SCHEMA_VERSION},"root":` +
      '{"type":"split","id":"s","dir":"row","sizes":[0.5,0.5],"children":['.repeat(DEPTH) +
      leaf +
      `,${leaf}]}`.repeat(DEPTH) +
      ',"focusedPaneId":"leaf-x"}';
    let restored!: ReturnType<typeof deserializeLayout>;
    expect(() => (restored = deserializeLayout(raw))).not.toThrow();
    expect(restored.root.type).toBe('leaf');
    expect(asLeaf(restored.root).tabs).toEqual([]);
  });

  test('normalizes a persisted split whose sizes do not sum to 1 (documented load invariant)', () => {
    const raw = JSON.stringify({
      version: LAYOUT_SCHEMA_VERSION,
      root: {
        type: 'split',
        id: 's',
        dir: 'row',
        children: [
          { type: 'leaf', id: 'a', tabs: [], activeTabId: null },
          { type: 'leaf', id: 'b', tabs: [], activeTabId: null },
        ],
        sizes: [0.9, 0.9], // sums to 1.8
      },
      focusedPaneId: 'a',
    });
    const restored = deserializeLayout(raw);
    const split = asSplit(restored.root);
    expect(split.sizes.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(split.sizes).toEqual([0.5, 0.5]);
  });
});

describe('collectLeaves', () => {
  test('returns leaves left-to-right in tree order', () => {
    const seed = seeded({ kind: 'chat' });
    let l = splitPane(seed, seed.root.id, 'row', { kind: 'terminal' });
    const rightId = collectLeaves(l.root)[1]!.id;
    l = splitPane(l, rightId, 'col', { kind: 'file' }); // row[chat, col[term, file]]
    expect(collectLeaves(l.root).map((x) => x.tabs[0]!.kind)).toEqual(['chat', 'terminal', 'file']);
  });
});

describe('retitleTab', () => {
  test('renames a tab found by id alone, in whichever pane holds it', () => {
    const seed = seeded({ kind: 'chat' });
    const l = splitPane(seed, seed.root.id, 'row', { kind: 'plugin/demo/board' });
    const target = collectLeaves(l.root)[1]!.tabs[0]!;
    const next = retitleTab(l, target.id, 'report.docx');
    expect(collectLeaves(next.root)[1]!.tabs[0]!.title).toBe('report.docx');
    expect(collectLeaves(next.root)[0]!.tabs[0]!.title).toBeUndefined();
  });

  test('a tab id nothing holds leaves the layout alone', () => {
    const l = seeded({ kind: 'chat' });
    expect(retitleTab(l, 'no-such-tab', 'x')).toEqual(l);
  });
});

describe('plugin surfaces', () => {
  test('a plugin tab round-trips through persistence like any other kind', () => {
    const l = seeded({ kind: 'plugin/demo/board', params: { path: 'q4.docx' } });
    const restored = deserializeLayout(serializeLayout(l));
    expect(restored).toEqual(l);
  });

  test('a persisted tab whose plugin is gone survives the load', () => {
    // The registry is a render-time lookup, so deserialize must not police kinds:
    // dropping the tab would silently lose a pane the user arranged.
    const l = seeded({ kind: 'plugin/uninstalled/thing' });
    const restored = deserializeLayout(serializeLayout(l));
    expect(collectLeaves(restored.root)[0]!.tabs[0]!.kind).toBe('plugin/uninstalled/thing');
  });
});
