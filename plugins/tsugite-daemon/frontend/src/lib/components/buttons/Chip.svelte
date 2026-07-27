<script lang="ts">
  // Chip - a metadata tag for env source, agent name, file/session references.
  import type { Snippet } from 'svelte';

  let {
    /** Dashed border - a not-yet-attached reference (composer file/chat/etc). */
    variant = 'default',
    /** Shows a trailing remove (×) button. */
    removable = false,
    removeLabel = 'Remove',
    onRemove,
    icon,
    children,
  }: {
    variant?: 'default' | 'ref';
    removable?: boolean;
    removeLabel?: string;
    onRemove?: () => void;
    /** Leading icon, e.g. `{#snippet icon()}<svg class="ic">…</svg>{/snippet}`. */
    icon?: Snippet;
    children?: Snippet;
  } = $props();

  function handleRemove(event: MouseEvent) {
    event.stopPropagation();
    onRemove?.();
  }
</script>

<span class="t-chip" class:t-chip--ref={variant === 'ref'}>
  {#if icon}{@render icon()}{/if}
  {#if children}{@render children()}{/if}
  {#if removable}
    <button type="button" class="x" aria-label={removeLabel} onclick={handleRemove}>
      <svg class="ic" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />
      </svg>
    </button>
  {/if}
</span>

<style>
  /* ---- kbd + chip ---- */
  .t-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
    cursor: default;
  }
  /* `:global(.ic)` on the descendant half: `icon` is a consumer-supplied
     snippet, so that <svg> is authored by (and scope-hashed to) the caller,
     not this component — without `:global` this rule could never match it. */
  .t-chip :global(.ic) {
    width: 10px;
    height: 10px;
    color: var(--tx3);
    flex: none;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .t-chip .x .ic {
    width: 9px;
    height: 9px;
  }
  .t-chip .x {
    cursor: pointer;
    color: var(--tx3);
    display: inline-flex;
    /* `.x` is a real <button> here (for keyboard/AT access); reset the UA button
       chrome so it reads as a bare glyph. */
    background: none;
    border: 0;
    padding: 0;
    margin: 0;
    font: inherit;
    line-height: 1;
  }
  .t-chip .x:hover {
    color: var(--st-err);
  }
  .t-chip--ref {
    border-style: dashed;
  }
  .t-chip--ref :global(.ic) {
    color: var(--acc);
  }
</style>
