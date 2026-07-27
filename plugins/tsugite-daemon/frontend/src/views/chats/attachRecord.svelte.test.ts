/// <reference types="@vitest/browser/context" />
import { afterEach, expect, test, vi } from 'vitest';
import { attachRecordToChat, refMarkerHtml, parseRefMarker } from './attachRecord';
import { sessions } from '$lib/stores/sessions.svelte';
import { router } from '$lib/router.svelte';
import { api } from '$lib/api/client';
import { toasts } from '$lib/components/feedback/toast-store.svelte';

afterEach(() => {
  vi.restoreAllMocks();
  sessions.rows = [];
  router.view = '';
  router.params = {};
});

test('parseRefMarker round-trips a copied marker', () => {
  expect(parseRefMarker(refMarkerHtml('session', '20260722_042329_odyn_85fc3c'))).toEqual({
    kind: 'session',
    id: '20260722_042329_odyn_85fc3c',
  });
  expect(parseRefMarker(refMarkerHtml('job', 'job-1a2b3c4d'))).toEqual({
    kind: 'job',
    id: 'job-1a2b3c4d',
  });
});

test('parseRefMarker ignores html that carries no marker', () => {
  expect(parseRefMarker('<b>just some pasted html</b>')).toBeNull();
  expect(parseRefMarker('')).toBeNull();
});

test('attachRecordToChat with no chat open warns and captures nothing', async () => {
  sessions.rows = [];
  router.view = '';
  router.params = {};
  const post = vi.spyOn(api, 'post');
  const push = vi.spyOn(toasts, 'push');

  await attachRecordToChat('job', 'job-1a2b3c4d');

  expect(push).toHaveBeenCalledWith('warn', 'Open a chat first');
  // No target -> the daemon capture endpoint is never hit.
  expect(post).not.toHaveBeenCalled();
});
