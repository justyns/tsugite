/** Pane heights below this get the compact layout: a tighter chat header, a
 *  folded composer, and no tab strip on a lone unsplit tab. A narrow viewport
 *  compacts at any height (see PaneView). The pane is the window less the top bar,
 *  measured at 38px. A 693px-tall half-height tile leaves 655 and compacts, a
 *  720px-tall laptop window leaves 682 and keeps the full layout.
 *  Playwright's default 1280x720 viewport sits on the full side. */
export const SHORT_PANE_PX = 670;

export function isShortPane(height: number): boolean {
  return height > 0 && height < SHORT_PANE_PX;
}
