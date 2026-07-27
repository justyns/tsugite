import { describe, expect, test } from 'vitest';
import { formatStale, paneSkeletonWidths } from './format';

describe('formatStale', () => {
  test('renders the bare label when there is no stale-since timestamp', () => {
    expect(formatStale(null, 1_000)).toBe('stale');
  });

  test('renders elapsed whole seconds once a timestamp is known', () => {
    expect(formatStale(0, 6_000)).toBe('stale · 6s');
  });

  test('rounds to the nearest second rather than truncating', () => {
    expect(formatStale(0, 2_600)).toBe('stale · 3s');
  });

  test('never goes negative for a clock that ticked before the effect ran', () => {
    expect(formatStale(1_000, 900)).toBe('stale · 0s');
  });
});

describe('paneSkeletonWidths', () => {
  test('returns one width per requested line', () => {
    expect(paneSkeletonWidths(4)).toEqual([72, 88, 55, 80]);
  });

  test('cycles the width palette for longer skeletons', () => {
    const widths = paneSkeletonWidths(8);
    expect(widths).toHaveLength(8);
    expect(widths[6]).toBe(widths[0]);
    expect(widths[7]).toBe(widths[1]);
  });

  test('handles zero lines', () => {
    expect(paneSkeletonWidths(0)).toEqual([]);
  });
});
