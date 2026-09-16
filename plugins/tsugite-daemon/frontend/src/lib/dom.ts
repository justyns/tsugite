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
