import type { ConnState } from '$lib/stores/conn.svelte';

/** The three visual states the Conn indicator ships (on/re/off). */
export type ConnDisplay = 'on' | 're' | 'off';

/**
 * Collapse the store's four wire states onto the indicator's three visuals.
 * `connecting` is the pre-first-connect state and must never warn, so it reads
 * optimistically as `on` until a real drop moves the store to `reconnecting`;
 * `lost` is the terminal give-up state and is the only one that shows offline.
 */
export function toConnDisplay(status: ConnState): ConnDisplay {
  switch (status) {
    case 'reconnecting':
      return 're';
    case 'lost':
      return 'off';
    default:
      return 'on';
  }
}
