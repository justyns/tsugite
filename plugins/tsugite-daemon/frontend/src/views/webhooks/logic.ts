/** Pure helpers for the Webhooks view (kept out of View.svelte so they're
 * directly unit-testable). */

// Mirrors adapters/http/webhooks.py's server-side source-slug pattern exactly
// - the source lands in inbox filenames, so the server restricts it to a safe
// slug. Validating the same pattern client-side fails the form fast instead
// of always round-tripping a 400.
const SOURCE_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;

export function isValidSource(source: string): boolean {
  return SOURCE_PATTERN.test(source);
}

/** The real, working delivery path - adapters/http/webhooks.py mounts this
 * top-level (token-in-path auth), not under /api. */
export function deliveryPath(token: string): string {
  return `/webhook/${encodeURIComponent(token)}`;
}

/** Absolute URL for copy-to-clipboard - an external service needs a full URL
 * in its webhook-URL field, not a path relative to this app. */
export function deliveryUrl(token: string, origin: string): string {
  return `${origin}${deliveryPath(token)}`;
}

export interface TestFirePayload {
  event: string;
  source: string;
  message: string;
  sent_at: string;
}

/** Synthetic body for the "test fire" action. This is a real POST to the real
 * public delivery endpoint - a 202 back means the whole path (token lookup,
 * inbox write) actually works, not just that the create form validated. */
export function buildTestPayload(source: string, nowMs: number = Date.now()): TestFirePayload {
  return {
    event: 'test',
    source,
    message: 'Test delivery sent from the Tsugite web UI',
    sent_at: new Date(nowMs).toISOString(),
  };
}
