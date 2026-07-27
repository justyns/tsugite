<script lang="ts">
  // Badge.
  // Informational counts are ambient (square, outlined, muted mono).
  // Action-required counts are warm filled pills with a halo - a different
  // shape, weight and temperature, so the distinction survives color-blindness.
  import type { Snippet } from 'svelte';

  let {
    /** `dot` renders no content - an 8px unread marker; pass `label` for its name. */
    variant = 'info',
    /** Accessible name, e.g. "3 jobs running" or "unread". Optional when the
     * visible count is already self-explanatory. */
    label,
    children,
  }: {
    variant?: 'info' | 'action' | 'err' | 'dot';
    label?: string;
    children?: Snippet;
  } = $props();
</script>

<span
  class="t-badge"
  class:t-badge--act={variant === 'action'}
  class:t-badge--err={variant === 'err'}
  class:t-badge--dot={variant === 'dot'}
  aria-label={label}
>
  {#if variant !== 'dot' && children}{@render children()}{/if}
</span>

<style>
  /* ---- badge ---- */
  .t-badge {
    display: inline-grid;
    place-items: center;
    min-width: 17px;
    height: 16px;
    padding: 0 5px;
    border-radius: var(--r-sm);
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    background: var(--bg3);
    border: 1px solid var(--bd1);
    color: var(--tx2);
  }
  .t-badge--act {
    border-radius: var(--r-full);
    background: var(--st-warn);
    border-color: transparent;
    color: var(--bg0);
    font-weight: 700;
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--st-warn) 18%, transparent);
  }
  .t-badge--err {
    border-radius: var(--r-full);
    background: var(--st-err);
    border-color: transparent;
    color: var(--bg0);
    font-weight: 700;
  }
  .t-badge--dot {
    min-width: 8px;
    width: 8px;
    height: 8px;
    padding: 0;
    border-radius: 50%;
    background: var(--acc);
    border: none;
  }
</style>
