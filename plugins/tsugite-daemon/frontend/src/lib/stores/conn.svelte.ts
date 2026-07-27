/**
 * SSE connection state, fed by sse.ts. `connecting` is the pre-first-connect
 * state (never warns); after a successful connect a drop shows `reconnecting`,
 * and `lost` is the terminal give-up state. `lastSeq` mirrors the replay cursor.
 * Exported as a class instance - never a reassigned $state binding.
 */
export type ConnState = 'connecting' | 'live' | 'reconnecting' | 'lost';

class ConnStore {
  status = $state<ConnState>('connecting');
  lastSeq = $state(0);

  setConnected(connected: boolean): void {
    if (connected) this.status = 'live';
    else if (this.status !== 'connecting') this.status = 'reconnecting';
  }

  markLost(): void {
    this.status = 'lost';
  }
}

export const conn = new ConnStore();
