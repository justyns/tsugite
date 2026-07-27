/**
 * The single source of truth for the app's user-facing keyboard shortcuts: the
 * `?` help overlay renders this list, so a new shell chord becomes discoverable
 * the moment it's added here. `keys` are canonical display tokens ('Mod'/'Alt'
 * render platform-aware via `keyLabel`, everything else literal) - NOT the raw
 * event values `resolveShellShortcut` matches, which the two keep in sync by hand.
 */
export type ShortcutGroup = 'Global' | 'Navigation' | 'Chat';

interface Shortcut {
  keys: string[];
  label: string;
  group: ShortcutGroup;
}

export const SHORTCUTS: Shortcut[] = [
  { keys: ['Mod', 'K'], label: 'Command palette', group: 'Global' },
  { keys: ['Mod', ','], label: 'Settings', group: 'Global' },
  { keys: ['Mod', 'Shift', 'O'], label: 'New chat', group: 'Global' },
  { keys: ['?'], label: 'Keyboard shortcuts', group: 'Global' },
  { keys: ['Alt', '↑'], label: 'Previous chat', group: 'Navigation' },
  { keys: ['Alt', '↓'], label: 'Next chat', group: 'Navigation' },
  { keys: ['Mod', 'Shift', '['], label: 'Previous tab', group: 'Navigation' },
  { keys: ['Mod', 'Shift', ']'], label: 'Next tab', group: 'Navigation' },
  { keys: ['Enter'], label: 'Send', group: 'Chat' },
  { keys: ['Shift', 'Enter'], label: 'New line', group: 'Chat' },
  { keys: ['Esc'], label: 'Stop generating', group: 'Chat' },
];

/** True on macOS, where 'Mod' reads as ⌘ (Ctrl elsewhere). Guarded for SSR /
 *  no-window / no-navigator like phoneNav's isPhoneWidth. */
function isMac(): boolean {
  if (typeof navigator === 'undefined') return false;
  const nav = navigator as Navigator & { userAgentData?: { platform?: string } };
  return /mac/i.test(nav.userAgentData?.platform ?? nav.platform ?? '');
}

/** Render one canonical key token for display: 'Mod' and 'Alt' resolve to the
 *  platform's glyph/word, every other token is literal. */
export function keyLabel(token: string): string {
  if (token === 'Mod') return isMac() ? '⌘' : 'Ctrl';
  if (token === 'Alt') return isMac() ? '⌥' : 'Alt';
  return token;
}
