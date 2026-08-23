import { describe, expect, test } from 'vitest';
import { buildTestPayload, deliveryPath, deliveryUrl, isValidSource } from './logic';

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

describe('relaveryPath / deliveryUrl', () => {
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
