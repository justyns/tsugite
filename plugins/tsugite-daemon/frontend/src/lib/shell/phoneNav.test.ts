import { describe, expect, test } from 'vitest';
import { workspacePhoneScreen, workspaceBackAction } from './phoneNav';
import type { Route } from '$lib/router.svelte';

describe('workspacePhoneScreen', () => {
  test('desktop/tablet never drills down (both rail + content share the grid)', () => {
    for (const view of ['chats', 'terminals', 'files'] as const) {
      expect(
        workspacePhoneScreen({ narrow: false, view, workspaceView: view, params: { any: 'x' } }),
      ).toBeNull();
    }
  });

  test('phone with the view content param shows the content screen', () => {
    expect(
      workspacePhoneScreen({
        narrow: true,
        view: 'chats',
        workspaceView: 'chats',
        params: { sessionId: 's1' },
      }),
    ).toBe('content');
    expect(
      workspacePhoneScreen({
        narrow: true,
        view: 'terminals',
        workspaceView: 'terminals',
        params: { terminalId: 't1' },
      }),
    ).toBe('content');
    expect(
      workspacePhoneScreen({
        narrow: true,
        view: 'files',
        workspaceView: 'files',
        params: { path: 'a/b.md' },
      }),
    ).toBe('content');
  });

  test('phone with no content param shows the list screen', () => {
    for (const view of ['chats', 'terminals', 'files'] as const) {
      expect(workspacePhoneScreen({ narrow: true, view, workspaceView: view, params: {} })).toBe(
        'list',
      );
    }
  });

  test("another view's param does not trigger content (only this view's key counts)", () => {
    // A files hash carrying a sessionId (stale/foreign param) is still the files list.
    expect(
      workspacePhoneScreen({
        narrow: true,
        view: 'files',
        workspaceView: 'files',
        params: { sessionId: 's1' },
      }),
    ).toBe('list');
  });

  test('a deep link carrying the content param lands directly on content', () => {
    expect(
      workspacePhoneScreen({
        narrow: true,
        view: 'terminals',
        workspaceView: 'terminals',
        params: { terminalId: 'deep-1' },
      }),
    ).toBe('content');
  });

  test('an empty boot hash or a full-view hash over the restored workspace is the list', () => {
    expect(
      workspacePhoneScreen({ narrow: true, view: '', workspaceView: 'files', params: {} }),
    ).toBe('list');
    expect(
      workspacePhoneScreen({
        narrow: true,
        view: 'jobs',
        workspaceView: 'chats',
        params: { sessionId: 'x' },
      }),
    ).toBe('list');
  });
});

describe('workspaceBackAction', () => {
  const listPrev = (view: string): Route => ({ view, params: {} });

  test('when the list is the previous entry, pop it (clean stack, matches browser back)', () => {
    for (const view of ['chats', 'terminals', 'files'] as const) {
      expect(workspaceBackAction(listPrev(view), view)).toEqual({ kind: 'pop' });
    }
  });

  test('a cold entry (no history behind us) pushes the list instead of escaping the app', () => {
    expect(workspaceBackAction(null, 'terminals')).toEqual({ kind: 'list' });
  });

  test('arriving from another view (deep link) pushes the list', () => {
    expect(workspaceBackAction({ view: 'jobs', params: {} }, 'files')).toEqual({ kind: 'list' });
  });

  test('arriving from another item pushes the list (never pop into a sibling)', () => {
    expect(workspaceBackAction({ view: 'chats', params: { sessionId: 'other' } }, 'chats')).toEqual(
      { kind: 'list' },
    );
  });
});
