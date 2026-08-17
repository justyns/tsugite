/**
 * Whether the conversation keeps itself pinned to the newest output as it
 * streams, persisted per-device to localStorage (`tsugite_auto_follow`,
 * default on).
 */
import { readLocal, writeLocal } from '$lib/storage';

const KEY = 'tsugite_auto_follow';

class AutoFollowStore {
  enabled = $state<boolean>(readLocal(KEY) !== 'false');

  set(next: boolean): void {
    this.enabled = next;
    writeLocal(KEY, String(next));
  }
}

export const autoFollow = new AutoFollowStore();
