/**
 * Auth store: bearer token + user id, mirrored to localStorage. `gated` drives
 * the token-entry pane; the API client flips it via `requireAuth()` on a 401 so
 * an expired token re-prompts without wiping what's stored. Exported as a class
 * instance - never a reassigned $state binding. localStorage keys are a wire
 * contract; do not rename.
 */
import { readLocal, removeLocal, writeLocal } from '$lib/storage';

const TOKEN_KEY = 'tsugite_token';
const USER_KEY = 'tsugite_user_id';
const DEFAULT_USER = 'web-user-1';

const initialToken = readLocal(TOKEN_KEY) ?? '';

class AuthStore {
  token = $state(initialToken);
  userId = $state(readLocal(USER_KEY) ?? DEFAULT_USER);
  gated = $state(!initialToken);
  /** Why the gate re-appeared (shown inline on the token pane); empty on first run. */
  gateReason = $state('');

  get authed(): boolean {
    return !this.gated && !!this.token;
  }

  save(next: string): void {
    this.token = next;
    this.gated = false;
    this.gateReason = '';
    writeLocal(TOKEN_KEY, next);
  }

  setUserId(next: string): void {
    this.userId = next;
    writeLocal(USER_KEY, next);
  }

  requireAuth(reason = ''): void {
    this.gated = true;
    this.gateReason = reason;
  }

  clear(): void {
    this.token = '';
    this.gated = true;
    removeLocal(TOKEN_KEY);
  }
}

export const auth = new AuthStore();
