import { describe, expect, it } from 'vitest';
import { relativeAgo, relativeTime } from './format';

const NOW = Date.parse('2026-07-14T12:00:00Z');
const ago = (ms: number) => new Date(NOW - ms).toISOString();
const S = 1000;
const M = 60 * S;
const H = 60 * M;
const D = 24 * H;

describe('relativeTime', () => {
  it('reads as now under 45 seconds', () => {
    expect(relativeTime(ago(0), NOW)).toBe('now');
    expect(relativeTime(ago(30 * S), NOW)).toBe('now');
  });
  it('scales through minutes, hours, days', () => {
    expect(relativeTime(ago(12 * M), NOW)).toBe('12m');
    expect(relativeTime(ago(1 * H), NOW)).toBe('1h');
    expect(relativeTime(ago(3 * D), NOW)).toBe('3d');
  });
  it('never goes negative for a slightly-future stamp', () => {
    expect(relativeTime(ago(-5 * S), NOW)).toBe('now');
  });
  it('is empty for a missing or unparseable stamp', () => {
    expect(relativeTime(null, NOW)).toBe('');
    expect(relativeTime('not-a-date', NOW)).toBe('');
  });
});

describe('relativeAgo', () => {
  it('appends "ago" except for now / empty', () => {
    expect(relativeAgo(ago(12 * M), NOW)).toBe('12m ago');
    expect(relativeAgo(ago(0), NOW)).toBe('now');
    expect(relativeAgo(null, NOW)).toBe('');
  });
});
