/**
 * localStorage access guarded for non-browser contexts. We key off `window` (not
 * `localStorage`) so this stays a no-op under SSR, vitest's node env, and Node's
 * flag-gated built-in `localStorage` - none of which we want to touch.
 */
function available(): boolean {
  return typeof window !== 'undefined';
}

export function readLocal(key: string): string | null {
  return available() ? window.localStorage.getItem(key) : null;
}

export function writeLocal(key: string, value: string): void {
  if (available()) window.localStorage.setItem(key, value);
}

export function removeLocal(key: string): void {
  if (available()) window.localStorage.removeItem(key);
}
