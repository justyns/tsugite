import type { RouteParams } from '$lib/router.svelte';

/** Route params for a chat deep link, shared by in-app navigation and window.open. */
export function chatRouteParams(sessionId: string, agent?: string): RouteParams {
  return { sessionId, ...(agent ? { agent } : {}) };
}
