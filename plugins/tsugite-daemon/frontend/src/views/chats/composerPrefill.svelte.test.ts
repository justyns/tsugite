/// <reference types="@vitest/browser/context" />
import { expect, test } from 'vitest';
import { ComposerPrefill } from './composerPrefill.svelte';

test('a request is delivered to the composer for its own session', () => {
  const store = new ComposerPrefill();
  store.request('s1', '/status', true);
  expect(store.consume('s1')).toEqual({ sessionId: 's1', text: '/status', run: true });
});

test('a request is invisible to composers for other sessions, and stays pending', () => {
  const store = new ComposerPrefill();
  store.request('s1', '/model ', false);
  expect(store.consume('s2')).toBeNull();
  // The intended composer still gets it once it mounts / re-runs its effect.
  expect(store.consume('s1')).toEqual({ sessionId: 's1', text: '/model ', run: false });
});

test('a request fires at most once', () => {
  const store = new ComposerPrefill();
  store.request('s1', '/status', true);
  expect(store.consume('s1')).not.toBeNull();
  expect(store.consume('s1')).toBeNull();
});

test('a newer request supersedes an unconsumed one', () => {
  const store = new ComposerPrefill();
  store.request('s1', '/model ', false);
  store.request('s2', '/status', true);
  expect(store.consume('s1')).toBeNull();
  expect(store.consume('s2')).toEqual({ sessionId: 's2', text: '/status', run: true });
});
