// One focus trap for every modal overlay (Modal, Palette, SetSecretModal). The
// query is the strictest of the three it replaced (Modal's): it skips disabled
// controls (via :not([disabled])) and unrendered ones (offsetParent === null),
// so Tab never lands on something the user can't see or use. The currently
// focused element always counts - a dialog container with tabindex="-1" that
// holds focus on open must still anchor the wrap.
const FOCUSABLE =
  'a[href],button:not([disabled]),input:not([disabled]),textarea:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])';

/** Visible, enabled, tab-reachable descendants of `root`, in DOM order. */
export function focusables(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (el) => el.offsetParent !== null || el === document.activeElement,
  );
}

/**
 * Svelte action that keeps Tab / Shift+Tab focus cycling within `node`. It owns
 * only the Tab wrap; each consumer keeps its own Escape and restore-focus
 * handling.
 */
export function trapFocus(node: HTMLElement) {
  function onKeydown(event: KeyboardEvent) {
    if (event.key !== 'Tab') return;
    const f = focusables(node);
    const first = f[0];
    const last = f[f.length - 1];
    if (!first || !last) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
  node.addEventListener('keydown', onKeydown);
  return { destroy: () => node.removeEventListener('keydown', onKeydown) };
}
