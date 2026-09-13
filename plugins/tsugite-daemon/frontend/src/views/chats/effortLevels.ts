import { api } from '$lib/api/client';

export interface EffortLevels {
  model: string;
  supported_effort_levels: string[] | null;
}

// Cleared once the request settles.
const inFlight = new Map<string, Promise<EffortLevels>>();

/** The model chip and the effort seg ask at the same time. One request answers both. */
export function fetchEffortLevels(sessionId: string): Promise<EffortLevels> {
  const path = `/api/chat/effort-levels?session_id=${encodeURIComponent(sessionId)}`;
  const pending = inFlight.get(path);
  if (pending) return pending;
  const request = api
    .get<EffortLevels>(path)
    .finally(() => inFlight.delete(path)) as Promise<EffortLevels>;
  inFlight.set(path, request);
  return request;
}
