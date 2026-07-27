/**
 * Browser geolocation capture wrapped as a typed result the UI can render without
 * touching the callback-style `navigator.geolocation` API or its numeric error
 * codes. `getCurrentLocation` never rejects: a missing API, an insecure (non-https)
 * context, or any getCurrentPosition failure resolves to a typed `{ ok: false }`
 * the caller turns into a toast. Coordinates come from the browser only - the
 * daemon has no location of its own, so the client captures them and sends them
 * as structured context metadata (via the location context provider).
 */

export interface GeoFix {
  /** Decimal degrees. */
  latitude: number;
  longitude: number;
  /** Radius of 95% confidence, in meters. */
  accuracy: number;
}

export type GeoErrorCode = 'unsupported' | 'permission' | 'unavailable' | 'timeout';

export interface GeoError {
  code: GeoErrorCode;
  /** Human-readable, ready to drop into a toast body. */
  message: string;
}

export type GeoResult = { ok: true; fix: GeoFix } | { ok: false; error: GeoError };

/** enableHighAccuracy trades battery for GPS-grade precision; a 10s ceiling stops
 *  a hung fix from stalling a send, and a short maximumAge lets a warm fix return
 *  instantly on repeat sends while still being "where you are now". */
export const DEFAULT_GEO_OPTIONS: PositionOptions = {
  enableHighAccuracy: true,
  timeout: 10_000,
  maximumAge: 60_000,
};

const MESSAGES: Record<GeoErrorCode, string> = {
  unsupported: 'Location needs a secure (https) connection and a browser that supports it.',
  permission: 'Location permission was denied.',
  unavailable: 'Your location could not be determined.',
  timeout: 'Timed out while getting your location.',
};

function mapError(err: GeolocationPositionError): GeoError {
  const code: GeoErrorCode =
    err.code === err.PERMISSION_DENIED
      ? 'permission'
      : err.code === err.TIMEOUT
        ? 'timeout'
        : 'unavailable';
  return { code, message: MESSAGES[code] };
}

function unsupported(): GeoResult {
  return { ok: false, error: { code: 'unsupported', message: MESSAGES.unsupported } };
}

export async function getCurrentLocation(
  options: PositionOptions = DEFAULT_GEO_OPTIONS,
): Promise<GeoResult> {
  // getCurrentPosition throws/errors in an insecure context anyway; catching it
  // up front gives a clearer message than a generic permission failure.
  if (typeof window !== 'undefined' && window.isSecureContext === false) return unsupported();
  const geo = typeof navigator !== 'undefined' ? navigator.geolocation : undefined;
  if (!geo) return unsupported();
  return new Promise((resolve) => {
    geo.getCurrentPosition(
      (pos) =>
        resolve({
          ok: true,
          fix: {
            latitude: pos.coords.latitude,
            longitude: pos.coords.longitude,
            accuracy: pos.coords.accuracy,
          },
        }),
      (err) => resolve({ ok: false, error: mapError(err) }),
      options,
    );
  });
}

/** `37.77490, -122.41940 (±20m)` - the location provider's captured value, shared
 *  by the composer chip and the rendered context gutter so they never disagree on
 *  the numbers. Five decimals is ~1m. */
export function formatGeoFix(fix: GeoFix): string {
  return `${fix.latitude.toFixed(5)}, ${fix.longitude.toFixed(5)} (±${Math.round(fix.accuracy)}m)`;
}
