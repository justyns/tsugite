<script lang="ts">
  // Button.
  import type { Snippet } from 'svelte';
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import { startSpin } from './spin';

  let {
    /** Visual intent. `pri` unblocks something; `danger` is tinted, not solid. */
    variant = 'default',
    /** `sm` (23px) is the density default inside rows; default is 28px. */
    size = 'default',
    /** Square, no padding, for a lone icon - pair with an `aria-label`. */
    iconOnly = false,
    /** Shows the braille spinner in place of `icon` and ignores clicks. */
    loading = false,
    disabled = false,
    icon,
    children,
    onclick,
    ...rest
  }: {
    variant?: 'default' | 'pri' | 'danger' | 'ghost';
    size?: 'default' | 'sm';
    iconOnly?: boolean;
    loading?: boolean;
    /** Leading icon, e.g. `{#snippet icon()}<svg class="ic">…</svg>{/snippet}`. Hidden while loading. */
    icon?: Snippet;
    children?: Snippet;
  } & HTMLButtonAttributes = $props();

  let frame = $state('⠋');

  $effect(() => {
    if (!loading) return;
    return startSpin((glyph) => (frame = glyph));
  });

  function handleClick(event: MouseEvent & { currentTarget: EventTarget & HTMLButtonElement }) {
    if (loading || disabled) return;
    onclick?.(event);
  }
</script>

<button
  type="button"
  class="t-btn"
  class:t-btn--pri={variant === 'pri'}
  class:t-btn--danger={variant === 'danger'}
  class:t-btn--ghost={variant === 'ghost'}
  class:t-btn--sm={size === 'sm'}
  class:t-btn--icon={iconOnly}
  class:is-loading={loading}
  {disabled}
  aria-busy={loading || undefined}
  {...rest}
  onclick={handleClick}
>
  <span class="t-spin" aria-hidden="true">{frame}</span>
  {#if icon && !loading}{@render icon()}{/if}
  {#if children}{@render children()}{/if}
</button>

<style>
  /* ---- button ---- */
  .t-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    height: 28px;
    padding: 0 11px;
    border-radius: var(--r-md);
    border: 1px solid var(--bd1);
    background: var(--bg3);
    color: var(--tx0);
    font: 500 var(--fs-md) / 1 var(--font-ui);
    cursor: pointer;
    white-space: nowrap;
    transition:
      background var(--t-1) var(--ease),
      border-color var(--t-1) var(--ease),
      filter var(--t-1) var(--ease);
  }
  /* `[data-state="x"]` hooks let the gallery force-preview a state statically,
     alongside the real pseudo-classes — both drive the exact same look.
     `:global(...)` on the ancestor half: that div is rendered by the gallery,
     not this component, so it never carries this component's scope hash —
     without `:global` the compiled selector could never match it. */
  :is(.t-btn:hover, :global([data-state='hover']) .t-btn) {
    background: var(--bg4);
    border-color: color-mix(in oklab, var(--bd1) 60%, var(--tx3));
  }
  .t-btn:active {
    filter: brightness(0.94);
  }
  :is(.t-btn:focus-visible, :global([data-state='focus']) .t-btn) {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .t-btn--pri {
    background: var(--acc);
    border-color: transparent;
    color: var(--on-acc);
    font-weight: 600;
  }
  :is(.t-btn--pri:hover, :global([data-state='hover']) .t-btn--pri) {
    background: color-mix(in oklab, var(--acc) 88%, var(--tx0));
    border-color: transparent;
    filter: none;
  }
  .t-btn--danger {
    background: color-mix(in oklab, var(--st-err) 13%, transparent);
    border-color: color-mix(in oklab, var(--st-err) 38%, transparent);
    color: var(--st-err);
  }
  :is(.t-btn--danger:hover, :global([data-state='hover']) .t-btn--danger) {
    background: color-mix(in oklab, var(--st-err) 22%, transparent);
    border-color: color-mix(in oklab, var(--st-err) 55%, transparent);
  }
  .t-btn--ghost {
    background: transparent;
    border-color: transparent;
    color: var(--tx1);
  }
  :is(.t-btn--ghost:hover, :global([data-state='hover']) .t-btn--ghost) {
    background: var(--bg3);
    color: var(--tx0);
    border-color: transparent;
  }
  .t-btn--sm {
    height: 23px;
    padding: 0 8px;
    font-size: var(--fs-sm);
    gap: 5px;
  }
  .t-btn--icon {
    width: 26px;
    padding: 0;
  }
  .t-btn .t-spin {
    display: none;
  }
  :is(.t-btn.is-loading, :global([data-state='loading']) .t-btn) .t-spin {
    display: inline-block;
  }
  :is(.t-btn.is-loading, :global([data-state='loading']) .t-btn) {
    pointer-events: none;
    color: color-mix(in oklab, currentColor 65%, transparent);
  }
  :is(.t-btn[disabled], :global([data-state='disabled']) .t-btn) {
    opacity: 0.45;
    pointer-events: none;
  }

  /* spinner */
  .t-spin {
    font-family: var(--font-mono);
    font-weight: 600;
    display: inline-block;
    width: 1.1ch;
    line-height: 1;
    color: var(--spin-c, currentColor);
    flex: none;
  }
</style>
