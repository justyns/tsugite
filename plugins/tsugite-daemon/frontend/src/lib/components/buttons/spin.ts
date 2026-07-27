// Braille-frame spinner glyph driver, shared by Button (loading state) and
// Pill (busy state) - the two `.t-spin` consumers in this group. Each caller
// owns its own interval (started/stopped by its own $effect), so a spinner
// that isn't currently visible costs nothing.
const FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const REDUCED_GLYPH = '∙';
const FRAME_MS = 96;

function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

/**
 * Starts driving a `.t-spin` glyph via `setFrame`, respecting
 * prefers-reduced-motion (a single static glyph, no timer). Returns a
 * cleanup function that stops the animation; call it from an `$effect`'s
 * teardown.
 */
export function startSpin(setFrame: (glyph: string) => void): () => void {
  if (prefersReducedMotion()) {
    setFrame(REDUCED_GLYPH);
    return () => {};
  }
  let i = 0;
  setFrame(FRAMES[i] ?? REDUCED_GLYPH);
  const id = setInterval(() => {
    i = (i + 1) % FRAMES.length;
    setFrame(FRAMES[i] ?? REDUCED_GLYPH);
  }, FRAME_MS);
  return () => clearInterval(id);
}
