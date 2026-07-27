/**
 * Theme store, persisted to localStorage (`tsugite_theme`, default mocha). Sets
 * `data-theme` on <html> so tokens.css re-skins the app, and syncs the
 * `<meta name="theme-color">` chrome to the active theme's --bg0. Exported as a
 * class instance - never a reassigned $state binding.
 */
import { readLocal, writeLocal } from '$lib/storage';

const THEMES = ['mocha', 'macchiato', 'frappe', 'latte', 'gruvbox'] as const;
export type Theme = (typeof THEMES)[number];

const KEY = 'tsugite_theme';

function isTheme(value: string | null): value is Theme {
  return value !== null && (THEMES as readonly string[]).includes(value);
}

function load(): Theme {
  const stored = readLocal(KEY);
  return isTheme(stored) ? stored : 'mocha';
}

function apply(next: Theme): void {
  if (typeof document === 'undefined') return;
  document.documentElement.dataset.theme = next;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (!meta) return;
  // Defer a frame so --bg0 resolves against the freshly-set data-theme.
  requestAnimationFrame(() => {
    const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg0').trim();
    if (bg) meta.setAttribute('content', bg);
  });
}

class ThemeStore {
  readonly list = THEMES;
  current = $state<Theme>(load());

  constructor() {
    apply(this.current);
  }

  set(next: Theme): void {
    this.current = next;
    writeLocal(KEY, next);
    apply(next);
  }
}

export const theme = new ThemeStore();
