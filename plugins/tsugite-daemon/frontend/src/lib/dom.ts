/**
 * True when the event target is a field that owns typed input - a text control
 * or a contenteditable. Global keyboard handlers (the command-palette shortcut,
 * the search hotkey) check this so a bare key never fires while the user is
 * typing into a field.
 */
export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT' ||
    target.isContentEditable
  );
}

/**
 * The left edge a right-anchored popover must clear: the rightmost left-edge
 * among ancestors that clip or scroll their overflow (0 = viewport). Callers
 * flip to left-anchored when the popover would spill past it.
 */
export function clipBoundaryLeft(el: HTMLElement): number {
  let bound = 0;
  for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
    const s = getComputedStyle(node);
    if (/(auto|scroll|hidden|clip)/.test(s.overflowX + s.overflowY)) {
      bound = Math.max(bound, node.getBoundingClientRect().left);
    }
  }
  return bound;
}
