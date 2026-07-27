/**
 * A one-shot channel from the app shell to a chat's composer: the command palette
 * requests that a session's composer either run a slash command outright or prefill
 * its input with `/name `. Reactive `$state` so the request survives the navigation
 * that opens/switches to the target chat - the composer's effect consumes it once
 * its own `sessionId` matches, whether it was already mounted or mounts afterwards.
 * Exported as a class instance - never a reassigned $state binding.
 */
export interface PrefillRequest {
  sessionId: string;
  text: string;
  /** Run the command immediately (no user input needed) vs. prefill + focus. */
  run: boolean;
}

export class ComposerPrefill {
  #pending = $state<PrefillRequest | null>(null);

  request(sessionId: string, text: string, run: boolean): void {
    this.#pending = { sessionId, text, run };
  }

  /** Reactive peek at the pending request without clearing it, so a consumer can
   *  hold off (e.g. wait for its command list to load) before calling consume. */
  get pending(): PrefillRequest | null {
    return this.#pending;
  }

  /** Take the pending request iff it targets `sessionId`, clearing it so it fires
   *  once; a mismatch leaves it pending for the composer it was meant for. Reading
   *  it inside a `$effect` subscribes that effect, so a later request re-runs it. */
  consume(sessionId: string): PrefillRequest | null {
    const req = this.#pending;
    if (!req || req.sessionId !== sessionId) return null;
    this.#pending = null;
    return req;
  }
}

export const composerPrefill = new ComposerPrefill();
