// Session-pill states. The broader job/connection state-language table also
// rides `.t-pill[data-st]`, but this type is scoped to session states only.
export type PillState = 'idle' | 'busy' | 'streaming' | 'compacting' | 'interrupted';

export const PILL_STATES: readonly PillState[] = [
  'idle',
  'busy',
  'streaming',
  'compacting',
  'interrupted',
];
