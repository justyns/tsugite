// Test support for the localStorage-backed stores. The node project has no
// `window`, so `storage.ts` no-ops unless a test stubs one in first.
import { beforeAll } from 'vitest';

export function fakeLocalStorage() {
  const map = new Map<string, string>();
  return {
    getItem: (k: string) => (map.has(k) ? (map.get(k) as string) : null),
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
  };
}

/** Transform the `.svelte.ts` module once, outside the 5s test budget: under
 *  full-suite load that first compile otherwise times the first test out. */
export function preloadStore(load: () => Promise<unknown>): void {
  beforeAll(load);
}
