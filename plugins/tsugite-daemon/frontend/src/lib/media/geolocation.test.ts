import { afterEach, describe, expect, test, vi } from 'vitest';
import { formatGeoFix, getCurrentLocation, type GeoFix } from './geolocation';

const FIX: GeoFix = { latitude: 37.7749, longitude: -122.4194, accuracy: 20 };

/** A fake GeolocationPositionError carrying the numeric constants the mapper
 *  compares against, so `err.code === err.PERMISSION_DENIED` resolves. */
function geoError(code: 1 | 2 | 3): GeolocationPositionError {
  return {
    code,
    message: '',
    PERMISSION_DENIED: 1,
    POSITION_UNAVAILABLE: 2,
    TIMEOUT: 3,
  };
}

/** Install a `navigator.geolocation.getCurrentPosition` that drives the success
 *  or error callback, standing in for the browser API in the node project. */
function stubGeolocation(impl: Geolocation['getCurrentPosition']) {
  vi.stubGlobal('navigator', { geolocation: { getCurrentPosition: impl } });
}

afterEach(() => vi.unstubAllGlobals());

describe('formatGeoFix', () => {
  test('renders lat/lon to five decimals and accuracy rounded in meters', () => {
    expect(formatGeoFix(FIX)).toBe('37.77490, -122.41940 (±20m)');
    expect(formatGeoFix({ latitude: 1.234567, longitude: -2.7, accuracy: 12.6 })).toBe(
      '1.23457, -2.70000 (±13m)',
    );
  });
});

describe('getCurrentLocation', () => {
  test('resolves a typed fix on success', async () => {
    stubGeolocation((ok) =>
      ok({
        coords: { latitude: 37.7749, longitude: -122.4194, accuracy: 20 },
      } as GeolocationPosition),
    );
    const res = await getCurrentLocation();
    expect(res).toEqual({ ok: true, fix: FIX });
  });

  test('maps PERMISSION_DENIED to a typed permission error', async () => {
    stubGeolocation((_ok, err) => err?.(geoError(1)));
    const res = await getCurrentLocation();
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error.code).toBe('permission');
  });

  test('maps POSITION_UNAVAILABLE to a typed unavailable error', async () => {
    stubGeolocation((_ok, err) => err?.(geoError(2)));
    const res = await getCurrentLocation();
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error.code).toBe('unavailable');
  });

  test('maps TIMEOUT to a typed timeout error', async () => {
    stubGeolocation((_ok, err) => err?.(geoError(3)));
    const res = await getCurrentLocation();
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error.code).toBe('timeout');
  });

  test('reports unsupported when the geolocation API is absent', async () => {
    vi.stubGlobal('navigator', {});
    const res = await getCurrentLocation();
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error.code).toBe('unsupported');
  });

  test('reports unsupported in an insecure context even if the API exists', async () => {
    vi.stubGlobal('window', { isSecureContext: false });
    vi.stubGlobal('navigator', { geolocation: { getCurrentPosition: vi.fn() } });
    const res = await getCurrentLocation();
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error.code).toBe('unsupported');
  });
});
