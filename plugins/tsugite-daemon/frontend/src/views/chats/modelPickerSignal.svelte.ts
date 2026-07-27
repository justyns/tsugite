/**
 * A one-shot channel from the app shell / composer to a chat's header model
 * picker: a `/model` slash command (from the palette or the inline `/` menu) asks
 * that session's picker to open its popover instead of dropping the user into a
 * text field. Reactive `$state` so the request survives the navigation that
 * switches to the target chat - ModelPicker's effect consumes it once its own
 * `sessionId` matches. Mirrors composerPrefill; exported as a class instance,
 * never a reassigned $state binding.
 */
export class ModelPickerRequest {
  #pending = $state<{ sessionId: string } | null>(null);

  request(sessionId: string): void {
    this.#pending = { sessionId };
  }

  /** Reactive peek at the pending request without clearing it, so a consumer's
   *  `$effect` subscribes and re-runs when a later request lands. */
  get pending(): { sessionId: string } | null {
    return this.#pending;
  }

  /** Take the pending request iff it targets `sessionId`, clearing it so it fires
   *  once; a mismatch leaves it pending for the picker it was meant for. */
  consume(sessionId: string): boolean {
    const req = this.#pending;
    if (!req || req.sessionId !== sessionId) return false;
    this.#pending = null;
    return true;
  }
}

export const modelPickerRequest = new ModelPickerRequest();
