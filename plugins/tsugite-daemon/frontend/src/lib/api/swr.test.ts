import { describe, expect, it } from 'vitest';
import { parseCache, serializeCache } from './swr';

describe('serializeCache / parseCache', () => {
  it('round-trips a payload under a matching version', () => {
    const raw = serializeCache([{ id: 'a' }], 1, 1000);
    expect(parseCache(raw, 1)).toEqual([{ id: 'a' }]);
  });

  it('drops a payload written under a different schema version', () => {
    const raw = serializeCache({ x: 1 }, 1);
    expect(parseCache(raw, 2)).toBeNull();
  });

  it('returns null for missing or corrupt entries', () => {
    expect(parseCache(null, 1)).toBeNull();
    expect(parseCache('not-json', 1)).toBeNull();
    expect(parseCache('42', 1)).toBeNull();
  });

  it('embeds the schema version and a timestamp in the envelope', () => {
    const env = JSON.parse(serializeCache('hi', 3, 12345));
    expect(env).toMatchObject({ v: 3, t: 12345, data: 'hi' });
  });
});
