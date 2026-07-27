/**
 * Per-provider "auto-attach this context to every send" preference, persisted to
 * localStorage. Device-specific and privacy-sensitive (only the browser can read
 * GPS, timezone, ...), so it can't live in daemon config. Keyed by the provider's
 * `autoAttachStoreKey`, memoized so the Settings toggle and the composer's send
 * gather share one reactive instance. Exported as a lookup returning a mutated
 * class instance - never a reassigned $state binding.
 */
import { readLocal, writeLocal } from '$lib/storage';

class AutoAttachStore {
  enabled = $state<boolean>(false);
  private key: string;

  constructor(key: string) {
    this.key = key;
    this.enabled = readLocal(key) === 'true';
  }

  set(next: boolean): void {
    this.enabled = next;
    writeLocal(this.key, String(next));
  }
}

const stores = new Map<string, AutoAttachStore>();

export function autoAttachStore(key: string): AutoAttachStore {
  let store = stores.get(key);
  if (!store) {
    store = new AutoAttachStore(key);
    stores.set(key, store);
  }
  return store;
}
