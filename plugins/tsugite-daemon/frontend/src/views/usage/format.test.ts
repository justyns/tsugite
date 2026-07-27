import { describe, expect, test } from 'vitest';
import {
  formatDayLabel,
  formatLastRun,
  formatRuns,
  formatTokensCompact,
  formatUsd,
} from './format';

describe('formatUsd', () => {
  test('pads whole numbers to two decimals', () => {
    expect(formatUsd(92)).toBe('$92.00');
  });

  test('groups thousands', () => {
    expect(formatUsd(1332.38)).toBe('$1,332.38');
  });

  test('rounds sub-cent totals to two decimals', () => {
    expect(formatUsd(1.845)).toBe('$1.85');
  });

  test('zero renders as $0.00, not $0', () => {
    expect(formatUsd(0)).toBe('$0.00');
  });

  test('null/undefined (unset cost_usd rows) render as $0.00 rather than throwing', () => {
    expect(formatUsd(null)).toBe('$0.00');
    expect(formatUsd(undefined)).toBe('$0.00');
  });
});

describe('formatTokensCompact', () => {
  test('renders sub-1000 counts as plain integers', () => {
    expect(formatTokensCompact(0)).toBe('0');
    expect(formatTokensCompact(999)).toBe('999');
  });

  test('renders thousands with a k suffix, dropping a trailing .0', () => {
    expect(formatTokensCompact(412000)).toBe('412k');
    expect(formatTokensCompact(31900)).toBe('31.9k');
  });

  test('renders millions with an M suffix', () => {
    expect(formatTokensCompact(40700000)).toBe('40.7M');
    expect(formatTokensCompact(490500000)).toBe('490.5M');
  });

  test('renders billions with a B suffix', () => {
    expect(formatTokensCompact(1_250_000_000)).toBe('1.3B');
  });

  test('null/undefined render as 0', () => {
    expect(formatTokensCompact(null)).toBe('0');
    expect(formatTokensCompact(undefined)).toBe('0');
  });
});

describe('formatRuns', () => {
  test('groups thousands with no decimals', () => {
    expect(formatRuns(2396)).toBe('2,396');
  });

  test('small counts pass through', () => {
    expect(formatRuns(4)).toBe('4');
  });
});

describe('formatDayLabel', () => {
  test('renders an ISO day as "mon dd"', () => {
    expect(formatDayLabel('2026-07-12')).toBe('jul 12');
  });

  test('zero-pads single-digit days', () => {
    expect(formatDayLabel('2026-01-05')).toBe('jan 05');
  });

  test('an unparseable period falls back to the raw string', () => {
    expect(formatDayLabel('2026-W28')).toBe('2026-W28');
  });
});

describe('formatLastRun', () => {
  test('renders an ISO timestamp as "mon dd hh:mm" (UTC, tz-safe string slice)', () => {
    expect(formatLastRun('2026-07-15T08:04:00+00:00')).toBe('jul 15 08:04');
  });

  test('accepts a trailing-Z timestamp', () => {
    expect(formatLastRun('2026-01-05T23:59:12Z')).toBe('jan 05 23:59');
  });

  test('null/undefined (never run) render as a dash', () => {
    expect(formatLastRun(null)).toBe('-');
    expect(formatLastRun(undefined)).toBe('-');
  });

  test('an unparseable value falls back to the raw string', () => {
    expect(formatLastRun('whenever')).toBe('whenever');
  });
});
