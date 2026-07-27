/**
 * Attribute bag that tells password-manager EXTENSIONS to leave a field alone.
 * `autocomplete="off"` is not enough: 1Password, LastPass, Bitwarden, and
 * Dashlane each use their own opt-out attribute and otherwise heuristically
 * decorate any focusable text input. Spread onto filter/search inputs
 * (`<input {...pwmIgnore} />`) - never onto a field where autofill is wanted
 * (e.g. the auth token gate).
 *
 * This does NOT cover Chromium's BUILT-IN password manager, which ignores all
 * of the above: that also needs `type="search"` on the input (Chromium excludes
 * search fields from its credential heuristics). Set the type per input - not
 * here - so the bag can't override an input's own type via spread order.
 */
export const pwmIgnore = {
  autocomplete: 'off',
  'data-1p-ignore': '',
  'data-lpignore': 'true',
  'data-bwignore': '',
  'data-form-type': 'other',
} as const;
