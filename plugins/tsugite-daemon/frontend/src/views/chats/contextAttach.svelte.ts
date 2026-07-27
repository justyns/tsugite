/**
 * A one-shot channel that pushes captured context items into a specific chat's
 * composer: the "add to chat" action on a session / job / terminal, and the
 * reference-paste path, both resolve a target chat and hand its composer the
 * items to chip. Reactive `$state` so the request survives the navigation that
 * focuses the target chat - the composer's effect consumes it once its own
 * `sessionId` matches, whether it was already mounted or mounts afterwards.
 * Mirrors composerPrefill. Exported as a class instance - never a reassigned
 * $state binding.
 */
import type { ContextItem } from '$lib/context/contextProviders';

export interface AttachRequest {
  sessionId: string;
  items: ContextItem[];
}

export class ContextAttach {
  #pending = $state<AttachRequest | null>(null);

  request(sessionId: string, items: ContextItem[]): void {
    this.#pending = { sessionId, items };
  }

  /** Reactive peek at the pending request without clearing it, so a consumer's
   *  effect subscribes and a later request re-runs it. */
  get pending(): AttachRequest | null {
    return this.#pending;
  }

  /** Take the pending request iff it targets `sessionId`, clearing it so it fires
   *  once; a mismatch leaves it pending for the composer it was meant for. */
  consume(sessionId: string): AttachRequest | null {
    const req = this.#pending;
    if (!req || req.sessionId !== sessionId) return null;
    this.#pending = null;
    return req;
  }
}

export const contextAttach = new ContextAttach();
