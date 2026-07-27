import { describe, expect, test } from 'vitest';
import { MUX_SURFACE_MIME, hasSurfaceDrag, readSurfaceDrag, writeSurfaceDrag } from './drag';

// Minimal stand-in for the parts of DataTransfer these helpers touch.
function fakeDT(initial: Record<string, string> = {}): DataTransfer {
  const store = new Map(Object.entries(initial));
  return {
    setData: (t: string, v: string) => void store.set(t, v),
    getData: (t: string) => store.get(t) ?? '',
    get types() {
      return [...store.keys()];
    },
    effectAllowed: 'none',
  } as unknown as DataTransfer;
}

describe('surface drag transport', () => {
  test('round-trips a surface ref through the custom MIME type', () => {
    const dt = fakeDT();
    writeSurfaceDrag(dt, { kind: 'chat', params: { id: 'sse' }, title: 'sse' });
    expect(readSurfaceDrag(dt)).toEqual({ kind: 'chat', params: { id: 'sse' }, title: 'sse' });
  });

  test('hasSurfaceDrag reports the MIME without reading the payload (dragover-safe)', () => {
    expect(hasSurfaceDrag(fakeDT({ [MUX_SURFACE_MIME]: '{}' }))).toBe(true);
    expect(hasSurfaceDrag(fakeDT({ 'text/plain': 'hi' }))).toBe(false);
    expect(hasSurfaceDrag(null)).toBe(false);
  });

  test('readSurfaceDrag returns null on absent or malformed payloads', () => {
    expect(readSurfaceDrag(null)).toBeNull();
    expect(readSurfaceDrag(fakeDT())).toBeNull();
    expect(readSurfaceDrag(fakeDT({ [MUX_SURFACE_MIME]: 'not-json' }))).toBeNull();
    expect(readSurfaceDrag(fakeDT({ [MUX_SURFACE_MIME]: '{"nope":1}' }))).toBeNull();
  });
});
