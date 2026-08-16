import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import {
  SPACES_SCHEMA_VERSION,
  SpacesStore,
  computeRollup,
  defaultSpace,
  parseSpaces,
  serializeSpaces,
  type Space,
} from './spaces.svelte';
import { collectLeaves, dockAsTab, type Layout } from '$lib/shell/mux/layout';
import { writeLocal } from '$lib/storage';

// persist() is a no-op in the node env (writeLocal guards on window); mock storage
// so the debounce / flush behaviour is observable through call counts.
vi.mock('$lib/storage', () => ({
  readLocal: vi.fn(() => null),
  writeLocal: vi.fn(),
  removeLocal: vi.fn(),
}));

function withTabs(space: Space, ...refs: Parameters<typeof dockAsTab>[2][]): Space {
  let layout: Layout = space.layout;
  for (const ref of refs) layout = dockAsTab(layout, collectLeaves(layout.root)[0]!.id, ref);
  return { ...space, layout };
}

describe('defaultSpace', () => {
  test('is a single named space whose one pane holds a single chat surface', () => {
    const s = defaultSpace();
    expect(s.name).toBeTruthy();
    const leaves = collectLeaves(s.layout.root);
    expect(leaves).toHaveLength(1);
    expect(leaves[0]!.tabs).toHaveLength(1);
    expect(leaves[0]!.tabs[0]!.kind).toBe('chat');
  });
});

describe('parseSpaces / serializeSpaces', () => {
  test('round-trips spaces and the active id', () => {
    const a = defaultSpace();
    const b = { ...defaultSpace(), name: 'Ops' };
    const raw = serializeSpaces([a, b], b.id);
    const parsed = parseSpaces(raw);
    expect(parsed.spaces.map((s) => s.name)).toEqual([a.name, 'Ops']);
    expect(parsed.activeSpaceId).toBe(b.id);
  });

  test('falls back to a single default space when the envelope is missing or corrupt', () => {
    for (const raw of [null, 'not-json', '{}', JSON.stringify({ version: 999, spaces: [] })]) {
      const parsed = parseSpaces(raw);
      expect(parsed.spaces).toHaveLength(1);
      expect(parsed.spaces[0]!.name).toBeTruthy();
      expect(parsed.activeSpaceId).toBe(parsed.spaces[0]!.id);
    }
  });

  test('self-heals one corrupt layout to a default without dropping the whole space list', () => {
    const good = defaultSpace();
    const raw = JSON.stringify({
      version: SPACES_SCHEMA_VERSION,
      activeSpaceId: good.id,
      spaces: [
        { id: good.id, name: good.name, layout: good.layout },
        {
          id: 'busted',
          name: 'Busted',
          layout: { version: 1, root: { type: 'split', children: [] } },
        },
      ],
    });
    const parsed = parseSpaces(raw);
    expect(parsed.spaces).toHaveLength(2);
    // the busted layout is replaced by a fresh single-leaf default
    const busted = parsed.spaces.find((s) => s.id === 'busted')!;
    expect(busted.layout.root.type).toBe('leaf');
  });

  test('an active id that names no surviving space resets to the first space', () => {
    const a = defaultSpace();
    const raw = serializeSpaces([a], 'ghost');
    expect(parseSpaces(raw).activeSpaceId).toBe(a.id);
  });

  test('one malformed space entry is dropped without discarding its valid siblings', () => {
    const good = defaultSpace('Keep me');
    const raw = JSON.stringify({
      version: SPACES_SCHEMA_VERSION,
      activeSpaceId: good.id,
      spaces: [
        { id: good.id, name: good.name, layout: good.layout },
        { id: 123, name: 'Bad id', layout: good.layout }, // non-string id
      ],
    });
    const parsed = parseSpaces(raw);
    // The bad entry is skipped; the sibling (and its layout) is preserved rather
    // than the whole workspace collapsing to a single fresh default.
    expect(parsed.spaces).toHaveLength(1);
    expect(parsed.spaces[0]!.id).toBe(good.id);
    expect(parsed.spaces[0]!.name).toBe('Keep me');
    expect(parsed.activeSpaceId).toBe(good.id);
  });

  test('an envelope with only malformed entries still falls back to a fresh default', () => {
    const raw = JSON.stringify({
      version: SPACES_SCHEMA_VERSION,
      activeSpaceId: 'x',
      spaces: [{ id: 1, name: 2 }],
    });
    const parsed = parseSpaces(raw);
    expect(parsed.spaces).toHaveLength(1);
    expect(parsed.spaces[0]!.name).toBeTruthy();
  });
});

describe('computeRollup', () => {
  test('buckets docked tab states into working / idle / blocked / done', () => {
    const s = withTabs(
      defaultSpace(),
      { kind: 'chat', params: { id: '1' }, state: 'busy' },
      { kind: 'chat', params: { id: '2' }, state: 'streaming' },
      { kind: 'chat', params: { id: '3' }, state: 'blocked' },
      { kind: 'chat', params: { id: '4' }, state: 'error' },
      { kind: 'chat', params: { id: '5' }, state: 'idle' },
      { kind: 'chat', params: { id: '6' }, state: 'done' },
    );
    // the default chat tab has no state and is not counted
    expect(computeRollup(s)).toEqual({ working: 2, blocked: 2, idle: 1, done: 1 });
  });

  test('accepts a caller-supplied resolver (the seam for live session data)', () => {
    const s = withTabs(defaultSpace(), { kind: 'chat', params: { id: 'x' } });
    const roll = computeRollup(s, () => 'working');
    expect(roll.working).toBeGreaterThan(0);
  });
});

describe('space management', () => {
  function firstPane(store: SpacesStore): string {
    return collectLeaves(store.active.layout.root)[0]!.id;
  }
  function surfaceKinds(store: SpacesStore, spaceId: string): string[] {
    const space = store.spaces.find((s) => s.id === spaceId)!;
    return collectLeaves(space.layout.root).flatMap((leaf) => leaf.tabs.map((t) => t.kind));
  }

  test('a new space starts on its own default layout and becomes active', () => {
    const store = new SpacesStore();
    const first = store.activeSpaceId;
    const second = store.addSpace('Notes');
    expect(store.spaces).toHaveLength(2);
    expect(store.activeSpaceId).toBe(second);
    expect(store.active.name).toBe('Notes');
    expect(collectLeaves(store.active.layout.root)).toHaveLength(1);
    expect(second).not.toBe(first);
  });

  // One space holds a chat+file split, another a single chat; switching restores each.
  test('each space keeps its own layout across a switch', () => {
    const store = new SpacesStore();
    const split = store.activeSpaceId;
    store.split(firstPane(store), 'row', { kind: 'file', params: { path: 'note.md' } });
    expect(collectLeaves(store.active.layout.root)).toHaveLength(2);
    expect(surfaceKinds(store, split)).toEqual(['chat', 'file']);

    const single = store.addSpace('Second');
    expect(collectLeaves(store.active.layout.root)).toHaveLength(1);
    expect(surfaceKinds(store, single)).toEqual(['chat']);
    // Splitting the second space leaves the first alone.
    expect(surfaceKinds(store, split)).toEqual(['chat', 'file']);

    store.setActive(split);
    expect(collectLeaves(store.active.layout.root)).toHaveLength(2);
    expect(surfaceKinds(store, split)).toEqual(['chat', 'file']);

    store.setActive(single);
    expect(collectLeaves(store.active.layout.root)).toHaveLength(1);
  });

  test('closing a tab in one space does not touch another', () => {
    const store = new SpacesStore();
    const a = store.activeSpaceId;
    store.split(firstPane(store), 'row', { kind: 'file', params: { path: 'note.md' } });
    const b = store.addSpace('Second');
    store.split(firstPane(store), 'row', { kind: 'terminal', params: {} });

    store.setActive(a);
    const pane = collectLeaves(store.active.layout.root)[0]!;
    store.closeTab(pane.id, pane.tabs[0]!.id);
    expect(surfaceKinds(store, a)).toEqual(['file']);
    expect(surfaceKinds(store, b)).toEqual(['chat', 'terminal']);
  });

  test('setActive ignores an id no space owns', () => {
    const store = new SpacesStore();
    const only = store.activeSpaceId;
    store.setActive('nope');
    expect(store.activeSpaceId).toBe(only);
  });

  test('renameSpace renames in place and leaves the layout alone', () => {
    const store = new SpacesStore();
    const id = store.activeSpaceId;
    const layout = store.active.layout;
    store.renameSpace(id, 'Planning');
    expect(store.active.name).toBe('Planning');
    expect(store.active.layout).toBe(layout);
  });

  test('removeSpace drops it and falls back to the neighbour', () => {
    const store = new SpacesStore();
    const first = store.activeSpaceId;
    const second = store.addSpace('Second');
    store.removeSpace(second);
    expect(store.spaces).toHaveLength(1);
    expect(store.activeSpaceId).toBe(first);
  });

  test('the last space is never removed - the shell always has somewhere to dock', () => {
    const store = new SpacesStore();
    store.removeSpace(store.activeSpaceId);
    expect(store.spaces).toHaveLength(1);
  });
});

describe('persist debounce', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(writeLocal).mockClear();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  function firstPane(store: SpacesStore): string {
    return collectLeaves(store.active.layout.root)[0]!.id;
  }

  test('coalesces a burst of mutations into a single trailing write', () => {
    const store = new SpacesStore();
    const pane = firstPane(store);
    store.focusPane(pane);
    store.focusPane(pane);
    store.focusPane(pane);
    // Nothing is written until the trailing timer elapses.
    expect(writeLocal).not.toHaveBeenCalled();
    vi.advanceTimersByTime(150);
    expect(writeLocal).toHaveBeenCalledTimes(1);
  });

  test('flush() writes the pending change immediately and cancels the debounce', () => {
    const store = new SpacesStore();
    store.focusPane(firstPane(store));
    expect(writeLocal).not.toHaveBeenCalled();
    store.flush();
    expect(writeLocal).toHaveBeenCalledTimes(1);
    // The debounce timer was cancelled, so it does not fire a second write.
    vi.advanceTimersByTime(150);
    expect(writeLocal).toHaveBeenCalledTimes(1);
  });

  test('flush() with nothing pending is a no-op', () => {
    const store = new SpacesStore();
    vi.mocked(writeLocal).mockClear();
    store.flush();
    expect(writeLocal).not.toHaveBeenCalled();
  });
});

describe('closing a space is undoable', () => {
  test('removeSpace hands back what it removed, and restoreSpace puts it back where it was', () => {
    const store = new SpacesStore();
    const a = store.activeSpaceId;
    const b = store.addSpace('B');
    const c = store.addSpace('C');
    store.setActive(b);
    const paneId = collectLeaves(store.active.layout.root)[0]!.id;
    store.dock(paneId, { kind: 'terminal', params: { id: 't1' } });

    const removed = store.removeSpace(b);
    expect(removed).not.toBeNull();
    expect(store.spaces.map((s) => s.id)).toEqual([a, c]);

    store.restoreSpace(removed!);
    expect(store.spaces.map((s) => s.id)).toEqual([a, b, c]);
    expect(store.activeSpaceId).toBe(b);
    // the layout comes back with it, not as a fresh empty space
    const kinds = collectLeaves(store.spaces[1]!.layout.root).flatMap((l) =>
      l.tabs.map((t) => t.kind),
    );
    expect(kinds).toContain('terminal');
  });

  test('removeSpace returns null when it refuses to drop the last space', () => {
    const store = new SpacesStore();
    expect(store.removeSpace(store.activeSpaceId)).toBeNull();
  });
});

describe('spaces reorder', () => {
  test('moveSpace takes an insertion index in the current order', () => {
    const store = new SpacesStore();
    const a = store.activeSpaceId;
    const b = store.addSpace('B');
    const c = store.addSpace('C');

    store.moveSpace(c, 0);
    expect(store.spaces.map((s) => s.id)).toEqual([c, a, b]);

    store.moveSpace(c, 3);
    expect(store.spaces.map((s) => s.id)).toEqual([a, b, c]);
  });

  test('reordering leaves the active space and every layout alone', () => {
    const store = new SpacesStore();
    const b = store.addSpace('B');
    const paneId = collectLeaves(store.active.layout.root)[0]!.id;
    store.dock(paneId, { kind: 'terminal', params: { id: 't1' } });

    store.moveSpace(b, 0);
    expect(store.activeSpaceId).toBe(b);
    const kinds = collectLeaves(store.active.layout.root).flatMap((l) => l.tabs.map((t) => t.kind));
    expect(kinds).toContain('terminal');
  });
});
