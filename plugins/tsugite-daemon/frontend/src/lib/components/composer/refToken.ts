export type RefTrigger = '@' | '#';

export interface RefToken {
  /** The trigger character that opened the token. */
  trigger: RefTrigger;
  /** Text typed between the trigger and the caret (may be empty). */
  query: string;
  /** Index of the trigger character in the source string. */
  start: number;
  /** Caret index - the exclusive end of the token. */
  end: number;
}

const TRIGGERS = new Set(['@', '#']);

/**
 * Detect an active @/# reference token at the caret.
 *
 * Scanning left from the caret, a plain token is active when we reach a trigger
 * character that sits at a word boundary (start of input or preceded by
 * whitespace) with no intervening whitespace or second trigger. Returns null
 * otherwise, so `user@host` (trigger mid-word) and `@a b|` (whitespace before
 * the caret) never open the popover.
 *
 * A plain token ends at the first whitespace. A prefix-scoped query may contain
 * spaces: when `prefixes` is given and the token's text begins with one of them
 * followed by a space (`@jira auth`), the whitespace is kept and the whole run is
 * the query. Anything else after a space still closes the popover, so ordinary
 * prose typed after a mention behaves as before.
 */
export function parseRefToken(
  value: string,
  caret: number,
  prefixes: string[] = [],
): RefToken | null {
  if (caret < 0) return null;
  const end = Math.min(caret, value.length);
  // Fast path: a plain token, bounded by the first whitespace left of the caret.
  for (let i = end - 1; i >= 0; i--) {
    const ch = value.charAt(i);
    if (TRIGGERS.has(ch)) {
      // charAt(-1) yields '' at the start, which is a valid word boundary.
      const boundary = i === 0 || /\s/.test(value.charAt(i - 1));
      if (!boundary) return null;
      return { trigger: ch as RefTrigger, query: value.slice(i + 1, end), start: i, end };
    }
    if (/\s/.test(ch)) break; // maybe a prefix-scoped token; fall through to below
  }
  // Prefix path: find the nearest boundary trigger past the spaces and accept it
  // only when its query opens with a known prefix + space.
  if (prefixes.length === 0) return null;
  for (let i = end - 1; i >= 0; i--) {
    const ch = value.charAt(i);
    if (TRIGGERS.has(ch)) {
      const boundary = i === 0 || /\s/.test(value.charAt(i - 1));
      if (!boundary) return null;
      const query = value.slice(i + 1, end);
      const scoped = prefixes.some((p) => query.startsWith(`${p} `));
      return scoped ? { trigger: ch as RefTrigger, query, start: i, end } : null;
    }
  }
  return null;
}
