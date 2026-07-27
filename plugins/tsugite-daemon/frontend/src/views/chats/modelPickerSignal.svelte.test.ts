/// <reference types="@vitest/browser/context" />
import { expect, test } from 'vitest';
import { ModelPickerRequest } from './modelPickerSignal.svelte';

test('a request is delivered to the picker for its own session', () => {
  const store = new ModelPickerRequest();
  store.request('s1');
  expect(store.consume('s1')).toBe(true);
});

test('a request is invisible to pickers for other sessions, and stays pending', () => {
  const store = new ModelPickerRequest();
  store.request('s1');
  expect(store.consume('s2')).toBe(false);
  // The intended picker still gets it once its effect re-runs.
  expect(store.consume('s1')).toBe(true);
});

test('a request fires at most once', () => {
  const store = new ModelPickerRequest();
  store.request('s1');
  expect(store.consume('s1')).toBe(true);
  expect(store.consume('s1')).toBe(false);
});

test('a newer request supersedes an unconsumed one', () => {
  const store = new ModelPickerRequest();
  store.request('s1');
  store.request('s2');
  expect(store.consume('s1')).toBe(false);
  expect(store.consume('s2')).toBe(true);
});
