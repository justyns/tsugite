// Node project: the registry's location provider wraps getCurrentLocation, which
// reads navigator.geolocation - stub it the same way geolocation.test.ts does.
import { afterEach, expect, test, vi } from 'vitest';
import { contextProviders, contextProvider } from './contextProviders';

function stubGeolocation(impl: Geolocation['getCurrentPosition']) {
  vi.stubGlobal('navigator', { geolocation: { getCurrentPosition: impl } });
}
afterEach(() => vi.unstubAllGlobals());

test('ships exactly one provider - location - with the metadata a menu entry needs', () => {
  expect(contextProviders.map((p) => p.key)).toEqual(['location']);
  const loc = contextProvider('location')!;
  expect(loc.label).toBe('Location');
  expect(loc.icon).toBe('pin');
  expect(loc.autoAttachStoreKey).toBe('tsugite_geo_autoattach');
});

test('contextProvider(key) resolves a provider, or undefined for an unknown key', () => {
  expect(contextProvider('location')?.key).toBe('location');
  expect(contextProvider('nope')).toBeUndefined();
});

test('location capture() formats the fix as the value on success', async () => {
  stubGeolocation((ok) =>
    ok({
      coords: { latitude: 37.7749, longitude: -122.4194, accuracy: 20 },
    } as GeolocationPosition),
  );
  const res = await contextProvider('location')!.capture();
  expect(res).toEqual({ value: '37.77490, -122.41940 (±20m)' });
});

test('location capture() returns a typed error (with code) on failure', async () => {
  stubGeolocation((_ok, err) =>
    err?.({ code: 1, message: '', PERMISSION_DENIED: 1, POSITION_UNAVAILABLE: 2, TIMEOUT: 3 }),
  );
  const res = await contextProvider('location')!.capture();
  expect('error' in res).toBe(true);
  if ('error' in res) expect(res.error.code).toBe('permission');
});
