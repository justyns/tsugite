/**
 * A per-device on/off preference, persisted to localStorage. Exported as a class
 * instance - never a reassigned $state binding.
 */
import { readLocal, writeLocal } from '$lib/storage';

class BooleanPref {
  enabled = $state(false);

  #key: string;

  constructor(key: string, fallback: boolean) {
    this.#key = key;
    // Only the opposite of the fallback counts as stored, so an absent key and a
    // value from some older spelling of it both read as the default.
    const stored = readLocal(key);
    this.enabled = fallback ? stored !== 'false' : stored === 'true';
  }

  set(next: boolean): void {
    this.enabled = next;
    writeLocal(this.#key, String(next));
  }
}

/** @param fallback What an unset key means, so a default-on toggle stays on. */
export function booleanPref(key: string, fallback: boolean): BooleanPref {
  return new BooleanPref(key, fallback);
}
