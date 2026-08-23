import { expect, test } from 'vitest';
import { formatAgo } from './relativeTime';

// Local noon, so the local-time date fallback renders the same day on any runner TZ.
const NOW = new Date(2026, 6, 14, 12, 0).getTime();

function before(ms: number): string {
  return new Date(NOW - ms).toISOString();
}

test('formatAgo buckets by magnitude, falling back to a date past a month', () => {
  expect(formatAgo(before(30_000), NOW)).toBe('just now');
  expect(formatAgo(before(5 * 60_000), NOW)).toBe('5m ago');
  expect(formatAgo(before(3 * 3_600_000), NOW)).toBe('3h ago');
  expect(formatAgo(before(3 * 24 * 3_600_000), NOW)).toBe('3d ago');
  expect(formatAgo(before(40 * 24 * 3_600_000), NOW)).toBe('jun 4');
  expect(formatAgo(before(400 * 24 * 3_600_000), NOW)).toBe('jun 9, 2025');
});

test('the bare style drops the ago suffix, keeping the date fallback', () => {
  expect(formatAgo(before(30_000), NOW, 'bare')).toBe('now');
  expect(formatAgo(before(12 * 60_000), NOW, 'bare')).toBe('12m');
  expect(formatAgo(before(3_600_000), NOW, 'bare')).toBe('1h');
  expect(formatAgo(before(3 * 24 * 3_600_000), NOW, 'bare')).toBe('3d');
  expect(formatAgo(before(40 * 24 * 3_600_000), NOW, 'bare')).toBe('jun 4');
});

test('formatAgo returns empty for a missing or unparseable timestamp', () => {
  expect(formatAgo(null, NOW)).toBe('');
  expect(formatAgo(undefined, NOW)).toBe('');
  expect(formatAgo('not a date', NOW)).toBe('');
  expect(formatAgo('not a date', NOW, 'bare')).toBe('');
});

test('formatAgo defaults to the current wall clock', () => {
  expect(formatAgo(new Date(Date.now() - 3 * 60_000).toISOString())).toBe('3m ago');
});

test('formatAgo reads a future timestamp as just now', () => {
  expect(formatAgo(before(-5_000), NOW)).toBe('just now');
  expect(formatAgo(before(-5_000), NOW, 'bare')).toBe('now');
});
