import { expect, test } from 'vitest';
import { neighborSession } from './chatNav';

const ids = ['a', 'b', 'c'];

test('steps to the next / previous id from the middle', () => {
  expect(neighborSession(ids, 'b', 1)).toBe('c');
  expect(neighborSession(ids, 'b', -1)).toBe('a');
});

test('clamps at each end (no wrap)', () => {
  expect(neighborSession(ids, 'c', 1)).toBeNull();
  expect(neighborSession(ids, 'a', -1)).toBeNull();
});

test('a current id not in the list is a no-op', () => {
  expect(neighborSession(ids, 'zzz', 1)).toBeNull();
  expect(neighborSession(ids, null, 1)).toBeNull();
});

test('a single-entry or empty list can never move', () => {
  expect(neighborSession(['only'], 'only', 1)).toBeNull();
  expect(neighborSession(['only'], 'only', -1)).toBeNull();
  expect(neighborSession([], 'a', 1)).toBeNull();
});
