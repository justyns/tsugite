/// <reference types="@vitest/browser/context" />
import { expect, test } from 'vitest';
import { ContextAttach } from './contextAttach.svelte';

const item = { key: 'session:s', label: 'a session', value: 'status: active' };

test('a request is delivered to the composer for its own session', () => {
  const store = new ContextAttach();
  store.request('s1', [item]);
  expect(store.consume('s1')).toEqual({ sessionId: 's1', items: [item] });
});

test('a request is invisible to composers for other sessions, and stays pending', () => {
  const store = new ContextAttach();
  store.request('s1', [item]);
  expect(store.consume('s2')).toBeNull();
  // The intended composer still gets it once it mounts / re-runs its effect.
  expect(store.consume('s1')).toEqual({ sessionId: 's1', items: [item] });
});

test('a request fires at most once', () => {
  const store = new ContextAttach();
  store.request('s1', [item]);
  expect(store.consume('s1')).not.toBeNull();
  expect(store.consume('s1')).toBeNull();
});

test('a newer request supersedes an unconsumed one', () => {
  const store = new ContextAttach();
  store.request('s1', [item]);
  store.request('s2', [item]);
  expect(store.consume('s1')).toBeNull();
  expect(store.consume('s2')).toEqual({ sessionId: 's2', items: [item] });
});
