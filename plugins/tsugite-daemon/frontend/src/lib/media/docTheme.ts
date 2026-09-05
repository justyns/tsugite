/**
 * The app's theme, resolved into literals a sandboxed preview frame can use.
 *
 * The frame is an opaque origin and cannot see the app's stylesheet, but
 * `readDocTheme` runs in the host, so it resolves the tokens off `[data-theme]`
 * and writes them into a sheet the caller injects ahead of the document. The
 * document's own styles come after and still win.
 *
 * The one place theme tokens become literals; the sheets downstream read them
 * back through `var()`.
 */

/** What the injected sheets read. Named rather than enumerated the way
 *  `$lib/plugins/bridge` does for plugin surfaces: the frame reaches no network,
 *  so the app's webfont tokens would be a trap there. A `var()` added to a sheet
 *  needs its token added here. */
const TOKENS = [
  '--bg1',
  '--bg2',
  '--bg3',
  '--tx0',
  '--tx1',
  '--tx2',
  '--bd0',
  '--bd1',
  '--acc',
  '--brand',
  '--st-err',
];

export interface DocTheme {
  /** `color-scheme` for the frame element, so its `Canvas` ground matches. */
  scheme: string;
  /** Injected ahead of the document. */
  sheet: string;
}

export function readDocTheme(el: Element): DocTheme {
  const style = getComputedStyle(el);
  const read = (name: string) => style.getPropertyValue(name).trim();
  const scheme = read('--scheme');
  const decls = TOKENS.map((name) => `${name}: ${read(name)};`).join(' ');
  return {
    scheme,
    sheet:
      `:root { color-scheme: ${scheme}; ${decls} }` +
      'body { background: var(--bg1); color: var(--tx1); }' +
      'a { color: var(--acc); }',
  };
}
