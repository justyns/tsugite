/* xterm.js lazy-loader.

   Pulls xterm + its fit / search / web-links addons from jsDelivr on first
   use, exposes them as globals (matches the upstream UMD pattern), and gives
   callers a `createTerminalRenderer(el, opts)` factory that wires the
   Catppuccin-mapped ANSI theme + the standard scrollback/cursor settings.

   The brief picked Catppuccin-mapped as the only palette - the canonical
   ANSI toggle from the design is deferred (truecolor + 256-color still pass
   through untouched). The palette below is sourced from
   terminal-sample.js's THEME_CATPPUCCIN. */

const XTERM_VERSION = '5.3.0';
const ADDON_FIT_VERSION = '0.8.0';
const ADDON_SEARCH_VERSION = '0.13.0';
const ADDON_WEBLINKS_VERSION = '0.9.0';

const SCRIPTS = [
  `https://cdn.jsdelivr.net/npm/xterm@${XTERM_VERSION}/lib/xterm.min.js`,
  `https://cdn.jsdelivr.net/npm/xterm-addon-fit@${ADDON_FIT_VERSION}/lib/xterm-addon-fit.min.js`,
  `https://cdn.jsdelivr.net/npm/xterm-addon-search@${ADDON_SEARCH_VERSION}/lib/xterm-addon-search.min.js`,
  `https://cdn.jsdelivr.net/npm/xterm-addon-web-links@${ADDON_WEBLINKS_VERSION}/lib/xterm-addon-web-links.min.js`,
];
const CSS_HREF = `https://cdn.jsdelivr.net/npm/xterm@${XTERM_VERSION}/css/xterm.min.css`;

/* Catppuccin Frappé-mapped 16-color palette. The truecolor/256-color slots
   that ANSI sequences directly request (e.g. \x1b[38;5;245m) bypass this
   table and render untouched, so exact-color tools stay accurate. */
export const THEME_CATPPUCCIN = {
  foreground: '#c6d0f5',
  background: '#232634',
  cursor: '#f2d5cf',
  cursorAccent: '#232634',
  selectionBackground: '#51576d',
  black: '#51576d',
  red: '#e78284',
  green: '#a6d189',
  yellow: '#e5c890',
  blue: '#8caaee',
  magenta: '#f4b8e4',
  cyan: '#81c8be',
  white: '#b5bfe2',
  brightBlack: '#626880',
  brightRed: '#e78284',
  brightGreen: '#a6d189',
  brightYellow: '#e5c890',
  brightBlue: '#8caaee',
  brightMagenta: '#f4b8e4',
  brightCyan: '#81c8be',
  brightWhite: '#a5adce',
};

let _loadPromise = null;

function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === '1') return resolve();
      existing.addEventListener('load', () => resolve());
      existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)));
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.addEventListener('load', () => { s.dataset.loaded = '1'; resolve(); });
    s.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)));
    document.head.appendChild(s);
  });
}

function _loadStylesheet(href) {
  if (document.querySelector(`link[href="${href}"]`)) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = href;
  document.head.appendChild(link);
}

export function loadXterm() {
  if (_loadPromise) return _loadPromise;
  _loadStylesheet(CSS_HREF);
  // Scripts must load sequentially: addons reference window.Terminal at register time.
  _loadPromise = SCRIPTS.reduce(
    (p, src) => p.then(() => _loadScript(src)),
    Promise.resolve(),
  );
  return _loadPromise;
}

/* createTerminalRenderer(el, opts)
     opts:
       theme        - palette dict (defaults to Catppuccin-mapped)
       fontSize     - px (default 12.5)
       cursorBlink  - bool
       disableStdin - bool (default true - the backend handles input later)
       scrollback   - int (default 5000)

   Returns { term, fit, search, write, dispose } where:
     write(chunk) - writes a raw ANSI string to the terminal and auto-tails
                    if the user hasn't scrolled away.
     dispose()   - tears down xterm, addons, and our scroll listener. */
export async function createTerminalRenderer(el, opts = {}) {
  await loadXterm();
  if (!el) throw new Error('createTerminalRenderer: target element is required');

  const term = new window.Terminal({
    fontFamily: '"JetBrains Mono", ui-monospace, Menlo, Consolas, monospace',
    fontSize: opts.fontSize ?? 12.5,
    lineHeight: 1.3,
    letterSpacing: 0,
    fontWeight: 400,
    fontWeightBold: 700,
    cursorBlink: !!opts.cursorBlink,
    cursorStyle: 'block',
    cursorInactiveStyle: 'none',
    scrollback: opts.scrollback ?? 5000,
    disableStdin: opts.disableStdin !== false,
    convertEol: false,
    theme: opts.theme || THEME_CATPPUCCIN,
    allowProposedApi: true,
  });

  let fit = null;
  if (window.FitAddon?.FitAddon) {
    fit = new window.FitAddon.FitAddon();
    term.loadAddon(fit);
  }
  let search = null;
  if (window.SearchAddon?.SearchAddon) {
    search = new window.SearchAddon.SearchAddon();
    term.loadAddon(search);
  }
  if (window.WebLinksAddon?.WebLinksAddon) {
    try { term.loadAddon(new window.WebLinksAddon.WebLinksAddon()); } catch { /* non-fatal */ }
  }

  term.open(el);
  try { fit?.fit(); } catch { /* non-fatal */ }

  let followTail = true;
  let onScrollState = opts.onScrollState || null;
  const viewport = el.querySelector('.xterm-viewport');
  const checkScroll = () => {
    if (!viewport) return;
    const atBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 6;
    followTail = atBottom;
    if (onScrollState) onScrollState(atBottom);
  };
  if (viewport) viewport.addEventListener('scroll', checkScroll, { passive: true });
  setTimeout(checkScroll, 60);

  return {
    term,
    fit,
    search,
    write(chunk) {
      if (chunk == null) return;
      term.write(chunk);
      if (followTail) term.scrollToBottom();
    },
    setTheme(nextTheme) {
      try { term.options.theme = nextTheme; } catch { /* older xterm builds - non-fatal */ }
    },
    setOnScrollState(fn) {
      onScrollState = fn;
    },
    isFollowingTail() { return followTail; },
    jumpToBottom() {
      term.scrollToBottom();
      followTail = true;
      if (onScrollState) onScrollState(true);
    },
    fitNow() {
      try { fit?.fit(); } catch { /* non-fatal */ }
    },
    dispose() {
      try { viewport?.removeEventListener('scroll', checkScroll); } catch { /* non-fatal */ }
      try { term.dispose(); } catch { /* non-fatal */ }
    },
  };
}
