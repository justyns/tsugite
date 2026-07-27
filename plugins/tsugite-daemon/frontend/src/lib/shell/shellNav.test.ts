import { describe, expect, test } from 'vitest';
import { defaultLayout, dockAsTab, splitPane } from './mux/layout';
import { focusedViewId, surfaceViewId } from './shellNav';

describe('surfaceViewId', () => {
  test('aliases the singular surface kinds to their nav view id', () => {
    expect(surfaceViewId('chat')).toBe('chats');
    expect(surfaceViewId('terminal')).toBe('terminals');
    expect(surfaceViewId('file')).toBe('files');
  });

  test('passes through kinds that are already view ids', () => {
    expect(surfaceViewId('jobs')).toBe('jobs');
    expect(surfaceViewId('schedules')).toBe('schedules');
    expect(surfaceViewId('usage')).toBe('usage');
  });
});

describe('focusedViewId', () => {
  test('an empty root leaf resolves to no view', () => {
    expect(focusedViewId(defaultLayout())).toBe('');
  });

  test('reads the surface active in the focused pane, mapped to its view id', () => {
    const base = defaultLayout();
    const withChat = dockAsTab(base, base.root.id, { kind: 'chat', title: 'Chat' });
    expect(focusedViewId(withChat)).toBe('chats');
  });

  test('a split follows focus to the freshly-opened pane', () => {
    const base = defaultLayout();
    const paneId = base.root.id;
    const withChat = dockAsTab(base, paneId, { kind: 'chat' });
    const split = splitPane(withChat, paneId, 'row', { kind: 'jobs' });
    expect(focusedViewId(split)).toBe('jobs');
  });
});
