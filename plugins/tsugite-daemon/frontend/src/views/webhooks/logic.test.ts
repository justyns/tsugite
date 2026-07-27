import { describe, expect, test } from 'vitest';
import { buildTestPayload, deliveryPath, deliveryUrl, isValidSource, relativeTime } from './logic';

describe('isValidSource', () => {
  test('accepts letters, digits, dot, underscore, dash', () => {
    expect(isValidSource('inbox-forward')).toBe(true);
    expect(isValidSource('github.events_v2')).toBe(true);
    expect(isValidSource('a')).toBe(true);
  });

  test('rejects empty, spaces, slashes, and other punctuation', () => {
    expect(isValidSource('')).toBe(false);
    expect(isValidSource('bad source')).toBe(false);
    expect(isValidSource('a/b')).toBe(false);
    expect(isValidSource('a..%b')).toBe(false);
  });

  test('rejects longer than 64 chars, accepts exactly 64', () => {
    expect(isValidSource('a'.repeat(64))).toBe(true);
    expect(isValidSource('a'.repeat(65))).toBe(false);
  });
});

describe('relativeTime', () => {
  const now = new Date('2026-07-14T12:00:00Z').getTime();

  test('under a minute reads "just now"', () => {
    expect(relativeTime('2026-07-14T11:59:31Z', now)).toBe('just now');
  });

  test('minutes', () => {
    expect(relativeTime('2026-07-14T11:55:00Z', now)).toBe('5m');
  });

  test('hours', () => {
    expect(relativeTime('2026-07-14T09:00:00Z', now)).toBe('3h');
  });

  test('days', () => {
    expect(relativeTime('2026-07-11T12:00:00Z', now)).toBe('3d');
  });

  test('weeks', () => {
    expect(relativeTime('2026-06-20T12:00:00Z', now)).toBe('3w');
  });

  test('months once past the ~5 week rollover', () => {
    // 74 days back (May has 31): 74/30 rounds to 2, not the week-bucket's 3w+.
    expect(relativeTime('2026-05-01T12:00:00Z', now)).toBe('2mo');
  });

  test('an unparseable timestamp returns empty rather than "NaNm"', () => {
    expect(relativeTime('not-a-date', now)).toBe('');
  });

  test('defaults `now` to the current time when omitted', () => {
    expect(relativeTime(new Date().toISOString())).toBe('just now');
  });
});

describe('deliveryPath / deliveryUrl', () => {
  test('path is the real top-level route, not under /api', () => {
    expect(deliveryPath('abc123')).toBe('/webhook/abc123');
  });

  test('path encodes tokens that need it', () => {
    expect(deliveryPath('a/b c')).toBe('/webhook/a%2Fb%20c');
  });

  test('url joins the given origin with the delivery path', () => {
    expect(deliveryUrl('abc123', 'https://tsugite.example.com')).toBe(
      'https://tsugite.example.com/webhook/abc123',
    );
  });
});

describe('buildTestPayload', () => {
  test('carries the source and a real ISO timestamp so the daemon log line means something', () => {
    const at = new Date('2026-07-14T12:00:00Z').getTime();
    const payload = buildTestPayload('inbox-forward', at);
    expect(payload.event).toBe('test');
    expect(payload.source).toBe('inbox-forward');
    expect(payload.sent_at).toBe('2026-07-14T12:00:00.000Z');
    expect(payload.message.length).toBeGreaterThan(0);
  });
});
