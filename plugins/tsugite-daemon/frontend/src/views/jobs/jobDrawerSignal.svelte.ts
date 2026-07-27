/**
 * A one-shot channel from an in-chat job tile to the Jobs view: clicking a
 * tile's "open" navigates to the Jobs view AND asks it to open that job's detail
 * drawer, instead of just landing on the board. Reactive `$state` so the request
 * survives the navigation; the Jobs view's effect consumes it once the job is
 * present in the store. Mirrors modelPickerRequest; exported as a class instance,
 * never a reassigned $state binding.
 */
export class JobDrawerRequest {
  #jobId = $state<string | null>(null);

  request(jobId: string): void {
    this.#jobId = jobId;
  }

  /** Reactive peek at the pending job id without clearing it, so a consumer's
   *  `$effect` subscribes and re-runs when a later request (or the store) lands. */
  get pending(): string | null {
    return this.#jobId;
  }

  /** Clear the pending request so it fires once. */
  consume(): void {
    this.#jobId = null;
  }
}

export const jobDrawerRequest = new JobDrawerRequest();
