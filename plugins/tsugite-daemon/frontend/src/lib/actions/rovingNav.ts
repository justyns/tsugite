// Pure roving-focus arithmetic shared by every arrow-key widget (segmented
// control, tab strip, generated-UI radiogroup). Framework- and DOM-free so the
// keyboard contract (arrow keys move, Home/End jump, wrap at the ends) is
// unit-tested without mounting a component; callers own focus + selection.

/** Index the given key moves to from `current`, or null if the key is unhandled. */
export function nextRovingIndex(current: number, key: string, count: number): number | null {
  if (count <= 0) return null;
  switch (key) {
    case 'ArrowRight':
    case 'ArrowDown':
      return (current + 1 + count) % count;
    case 'ArrowLeft':
    case 'ArrowUp':
      return (current - 1 + count) % count;
    case 'Home':
      return 0;
    case 'End':
      return count - 1;
    default:
      return null;
  }
}
