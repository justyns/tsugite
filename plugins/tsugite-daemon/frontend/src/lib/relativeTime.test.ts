import { expect, test } from 'vitest';
import { relativeTime } from './relativeTime';

const ago = (ms: number) => new Date(Date.now() - ms).toISOString();

test('relativeTime buckets by magnitude', () => {
  expect(relativeTime(ago(5_000))).toBe('just now'); // under 45s
  expect(relativeTime(ago(3 * 60_000))).toBe('3m ago');
  expect(relativeTime(ago(2 * 3_600_000))).toBe('2h ago');
  expect(relativeTime(ago(3 * 86_400_000))).toBe('3d ago');
});

test('relativeTime returns empty for an unparseable timestamp', () => {
  expect(relativeTime('not a date')).toBe('');
});
