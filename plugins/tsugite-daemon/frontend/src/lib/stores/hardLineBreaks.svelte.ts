/**
 * Whether a soft line break in the person's own message renders as a hard line
 * break, persisted to localStorage (`tsugite_hard_line_breaks`, default on).
 * Per-device like the other rendering preferences.
 */
import { readLocal, writeLocal } from '$lib/storage';

const KEY = 'tsugite_hard_line_breaks';

class HardLineBreaksStore {
  enabled = $state<boolean>(readLocal(KEY) !== 'false');

  set(next: boolean): void {
    this.enabled = next;
    writeLocal(KEY, String(next));
  }
}

export const hardLineBreaks = new HardLineBreaksStore();
