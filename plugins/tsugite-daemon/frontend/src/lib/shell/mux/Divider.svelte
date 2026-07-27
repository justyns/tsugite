<script lang="ts">
  import type { SplitDir } from './layout';

  let {
    dir,
    splitId,
    index,
    valueNow,
    onResize,
    label = 'Resize panes',
  }: {
    /** Orientation of the split this divider sits in. */
    dir: SplitDir;
    splitId: string;
    /** Divider between child `index` and `index + 1`. */
    index: number;
    /** First pane's share of the pair, 0-100, for aria-valuenow. */
    valueNow: number;
    onResize?: (splitId: string, dividerIndex: number, deltaFraction: number) => void;
    label?: string;
  } = $props();

  // Keyboard nudges the boundary; the layout reducer clamps at the extremes.
  const STEP = 0.02;
  const BIG = 1;

  // WAI-ARIA requires aria-valuenow to sit within [valuemin, valuemax]. The share
  // handed in is a pane's percent of its pair, which can exceed this band in an
  // over-split layout, so clamp it into the divider's declared range.
  const VALUE_MIN = 5;
  const VALUE_MAX = 95;
  const ariaValueNow = $derived(Math.max(VALUE_MIN, Math.min(VALUE_MAX, valueNow)));

  let handle = $state<HTMLDivElement>();
  let last = 0;
  // Pointer moves can fire several times per frame; accumulate their deltas and
  // apply at most one resize per animation frame (flushing the remainder on
  // pointer-up), so the reducer + rerender run once a frame, not once an event.
  let pendingDelta = 0;
  let rafId = 0;

  function flushResize() {
    rafId = 0;
    const delta = pendingDelta;
    pendingDelta = 0;
    if (delta !== 0) onResize?.(splitId, index, delta);
  }

  $effect(() => () => {
    if (rafId !== 0) cancelAnimationFrame(rafId);
  });

  // Axis size of the parent split, to convert a pixel drag into a fraction.
  function axisSize(): number {
    const parent = handle?.parentElement;
    if (!parent) return 0;
    return dir === 'row' ? parent.clientWidth : parent.clientHeight;
  }

  function onpointerdown(e: PointerEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    last = dir === 'row' ? e.clientX : e.clientY;
    handle?.setPointerCapture(e.pointerId);
    handle?.classList.add('is-drag');
  }

  function onpointermove(e: PointerEvent) {
    if (!handle?.hasPointerCapture(e.pointerId)) return;
    const size = axisSize();
    if (size <= 0) return;
    const pos = dir === 'row' ? e.clientX : e.clientY;
    pendingDelta += (pos - last) / size;
    last = pos;
    if (rafId === 0) rafId = requestAnimationFrame(flushResize);
  }

  function endDrag(e: PointerEvent) {
    if (handle?.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId);
    handle?.classList.remove('is-drag');
    // Apply whatever accrued since the last frame so the final position isn't lost.
    if (rafId !== 0) {
      cancelAnimationFrame(rafId);
      flushResize();
    }
  }

  function onkeydown(e: KeyboardEvent) {
    const dec = dir === 'row' ? 'ArrowLeft' : 'ArrowUp';
    const inc = dir === 'row' ? 'ArrowRight' : 'ArrowDown';
    let delta = 0;
    if (e.key === dec) delta = -STEP;
    else if (e.key === inc) delta = STEP;
    else if (e.key === 'Home') delta = -BIG;
    else if (e.key === 'End') delta = BIG;
    else return;
    e.preventDefault();
    onResize?.(splitId, index, delta);
  }
</script>

<!-- WAI-ARIA window-splitter pattern: a role="separator" that is focusable and
     resizable via the keyboard (aria-valuenow/min/max). The a11y linter treats
     separators as non-interactive, but this is the sanctioned resize-handle role. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div
  bind:this={handle}
  class="mux-rz"
  class:is-col={dir === 'col'}
  role="separator"
  tabindex="0"
  aria-orientation={dir === 'row' ? 'vertical' : 'horizontal'}
  aria-label={label}
  aria-valuemin={VALUE_MIN}
  aria-valuemax={VALUE_MAX}
  aria-valuenow={ariaValueNow}
  {onpointerdown}
  {onpointermove}
  onpointerup={endDrag}
  onpointercancel={endDrag}
  {onkeydown}
></div>

<style>
  /* Grabber that overlays the 1px grid line between two panes. Row splits get a
     vertical bar; col splits a horizontal one. */
  .mux-rz {
    position: relative;
    flex: none;
    align-self: stretch;
    width: 1px;
    background: var(--bd1);
    cursor: col-resize;
    touch-action: none;
  }
  .mux-rz.is-col {
    width: auto;
    height: 1px;
    cursor: row-resize;
  }
  /* Widen the hit area past the 1px line without shifting layout. */
  .mux-rz::before {
    content: '';
    position: absolute;
    inset: 0 -3px;
    z-index: 30;
  }
  .mux-rz.is-col::before {
    inset: -3px 0;
  }
  /* Visible grabber pill, highlighted on hover / focus / drag. */
  .mux-rz::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    translate: -50% -50%;
    width: 3px;
    height: 34px;
    border-radius: 2px;
    background: var(--bd1);
    z-index: 31;
  }
  .mux-rz.is-col::after {
    width: 34px;
    height: 3px;
  }
  .mux-rz:hover::after,
  .mux-rz:focus-visible::after,
  .mux-rz:global(.is-drag)::after {
    background: var(--acc);
  }
  .mux-rz:focus-visible {
    outline: none;
  }
</style>
