/**
 * Global keyboard shortcuts the app shell owns. Views own their own internal
 * keys (list nav, composer, etc.); this resolves only the shell-level chords so
 * the mapping stays a pure, testable function of the event's shape.
 */
export type ShellShortcut =
  | 'toggle-palette'
  | 'open-palette'
  | 'open-settings'
  | 'new-chat'
  | 'show-help'
  | 'next-chat'
  | 'prev-chat'
  | 'next-tab'
  | 'prev-tab'
  | null;

export interface KeyContext {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
  /** Focus is in a text field, so bare-key chords must yield to typing. */
  typing: boolean;
}

export function resolveShellShortcut(e: KeyContext): ShellShortcut {
  const mod = e.metaKey || e.ctrlKey;
  const key = e.key.toLowerCase();
  // Mod/Alt chords fire regardless of `typing`; only the bare `/` and `?` yield
  // to a field. Every chord below dodges a browser-reserved one (⌘/Ctrl+N,
  // ⌘/Ctrl+Tab, ⌘/Ctrl+1..9), which the page never sees.
  // ⌘/Ctrl+N is reserved, so new-chat rides ⌘/Ctrl+Shift+O like ChatGPT.
  if (mod && e.shiftKey && key === 'o') return 'new-chat';
  // ⌘/Ctrl+Shift+] / [ cycle mux tabs. Holding Shift makes the browser report the
  // SHIFTED glyph (] -> }, [ -> {), so match both forms - the bare bracket never
  // arrives on a US layout, but other layouts / key reports may still send it.
  if (mod && e.shiftKey && (e.key === ']' || e.key === '}')) return 'next-tab';
  if (mod && e.shiftKey && (e.key === '[' || e.key === '{')) return 'prev-tab';
  if (mod && key === 'k') return 'toggle-palette';
  if (mod && e.key === ',') return 'open-settings';
  // Alt+Arrow steps the chat list, even mid-compose (Alt+arrow is inert in a
  // textarea on Linux/Firefox; macOS Option+arrow is paragraph-nav, an accepted
  // minor caveat).
  if (e.altKey && !mod && e.key === 'ArrowDown') return 'next-chat';
  if (e.altKey && !mod && e.key === 'ArrowUp') return 'prev-chat';
  if (!mod && !e.typing && e.key === '/') return 'open-palette';
  // `?` is Shift+/; typing-gated so it never fires mid-message.
  if (!mod && !e.altKey && !e.typing && e.key === '?') return 'show-help';
  return null;
}
