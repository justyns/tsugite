import { describe, expect, test } from 'vitest';
import { defaultLayout, dockAsTab, focusPane, splitPane } from './mux/layout';
import { dockedChatSessionId, focusedViewId, surfaceViewId } from './shellNav';

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

describe('dockedChatSessionId', () => {
  test('no chat docked resolves to null', () => {
    const base = defaultLayout();
    const withJobs = dockAsTab(base, base.root.id, { kind: 'jobs' });
    expect(dockedChatSessionId(withJobs)).toBeNull();
  });

  test('reads the sessionId off the one docked chat tab', () => {
    const base = defaultLayout();
    const withChat = dockAsTab(base, base.root.id, {
      kind: 'chat',
      params: { sessionId: 'sess-1' },
    });
    expect(dockedChatSessionId(withChat)).toBe('sess-1');
  });

  test('an artifact pane taking focus does not lose the docked chat', () => {
    const base = defaultLayout();
    const chatPaneId = base.root.id;
    const withChat = dockAsTab(base, chatPaneId, {
      kind: 'chat',
      params: { sessionId: 'sess-1' },
    });
    const split = splitPane(withChat, chatPaneId, 'row', {
      kind: 'artifact',
      params: { id: 'agent' },
    });
    // splitPane focuses the new pane, giving the artifact pane focus, not the chat.
    expect(dockedChatSessionId(split)).toBe('sess-1');
  });

  test('a focused artifact stamped with a session names that session', () => {
    const base = defaultLayout();
    const chatPaneId = base.root.id;
    const withChat = dockAsTab(base, chatPaneId, { kind: 'chat', params: { sessionId: 'sess-1' } });
    const split = splitPane(withChat, chatPaneId, 'row', {
      kind: 'artifact',
      params: { id: 'agent', sessionId: 'sess-1' },
    });
    expect(dockedChatSessionId(split)).toBe('sess-1');
  });

  test('a chat tab hidden behind the active one does not name the docked session', () => {
    const base = defaultLayout();
    const paneId = base.root.id;
    const hiddenFirst = dockAsTab(base, paneId, { kind: 'chat', params: { sessionId: 'hidden' } });
    const withActive = dockAsTab(hiddenFirst, paneId, {
      kind: 'chat',
      params: { sessionId: 'shown' },
    });
    const split = splitPane(withActive, paneId, 'row', {
      kind: 'artifact',
      params: { id: 'agent' },
    });
    expect(dockedChatSessionId(split)).toBe('shown');
  });

  test('prefers the focused chat tab when two chats are split side by side', () => {
    const base = defaultLayout();
    const paneAId = base.root.id;
    const withA = dockAsTab(base, paneAId, { kind: 'chat', params: { sessionId: 'sess-a' } });
    const split = splitPane(withA, paneAId, 'row', {
      kind: 'chat',
      params: { sessionId: 'sess-b' },
    });
    expect(dockedChatSessionId(split)).toBe('sess-b');
    const backToA = focusPane(split, paneAId);
    expect(dockedChatSessionId(backToA)).toBe('sess-a');
  });
});
