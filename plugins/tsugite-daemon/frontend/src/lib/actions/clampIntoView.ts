/** Gap kept between a clamped popover and the edge it was pushed off. */
const GUTTER = 8;

/**
 * Horizontal px to move a box spanning [left, right] so it sits within
 * [min, max]. A box too wide for that span is pinned to `min`.
 */
export function clampShiftX(left: number, right: number, min: number, max: number): number {
  return Math.max(Math.min(0, max - right), min - left);
}

/**
 * Translates a chip-anchored popover horizontally so it clears both edges of the
 * viewport and of every ancestor that clips or scrolls its overflow.
 */
export function clampIntoView(el: HTMLElement): void {
  let min = 0;
  let max = document.documentElement.clientWidth;
  for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
    const s = getComputedStyle(node);
    if (/(auto|scroll|hidden|clip)/.test(s.overflowX + s.overflowY)) {
      const box = node.getBoundingClientRect();
      min = Math.max(min, box.left);
      max = Math.min(max, box.right);
    }
  }
  const r = el.getBoundingClientRect();
  const shift = Math.round(clampShiftX(r.left, r.right, min + GUTTER, max - GUTTER));
  if (shift) el.style.transform = `translateX(${shift}px)`;
}
