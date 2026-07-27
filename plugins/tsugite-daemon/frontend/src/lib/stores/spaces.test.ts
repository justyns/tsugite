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
