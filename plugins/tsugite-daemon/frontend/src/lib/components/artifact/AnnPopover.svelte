<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';

  // Selection popover raised when text is highlighted in an artifact/doc:
  // comment on the span, ask the agent about it, or copy a stable reference.
  // Two skins: `menu` (standalone, UI-font buttons) and `art` (docked in the
  // artifact pane, compact t-iconbtn buttons). Same three actions either way.
  // .t-iconbtn is a bare icon button with no shared component (toast/code-block
  // keep their own too); its 11px icon sizing lives globally in tokens.css.
  let {
    open = true,
    variant = 'menu',
    isStatic = false,
    x,
    y,
    onComment,
    onAsk,
    onCopyRef,
  }: {
    open?: boolean;
    variant?: 'menu' | 'art';
    isStatic?: boolean;
    x?: number;
    y?: number;
    onComment?: () => void;
    onAsk?: () => void;
    onCopyRef?: () => void;
  } = $props();

  const pos = $derived(!isStatic && x != null && y != null ? `--x:${x}px;--y:${y}px` : undefined);
  const btnClass = $derived(variant === 'art' ? 't-iconbtn' : '');
</script>

<div
  class="ann-pop"
  class:art-pop={variant === 'art'}
  class:is-open={open}
  class:is-static={isStatic}
  role="menu"
  aria-label="Selection actions"
  style={pos}
>
  <button type="button" class={btnClass} role="menuitem" onclick={() => onComment?.()}>
    <Icon name="edit" />Comment
  </button>
  <button type="button" class={btnClass} role="menuitem" onclick={() => onAsk?.()}>
    {#if variant === 'art'}
      <Icon name="sparkle" />
    {:else}
      <Icon name="chat" />
    {/if}Ask agent
  </button>
  <button type="button" class={btnClass} role="menuitem" onclick={() => onCopyRef?.()}>
    {#if variant === 'art'}
      <Icon name="link" />
    {:else}
      <Icon name="copy" />
    {/if}Copy ref
  </button>
</div>

<style>
  .ann-pop {
    position: absolute;
    left: var(--x, auto);
    top: var(--y, auto);
    z-index: 40;
    display: none;
    gap: 2px;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    padding: 3px;
  }
  /* Specimen/static placement (galleries) - opts out of the absolute anchor. */
  .ann-pop.is-static {
    position: static;
  }
  .ann-pop.is-open {
    display: inline-flex;
  }
  .ann-pop button {
    display: inline-flex;
    gap: 5px;
    align-items: center;
    background: none;
    border: 0;
    color: var(--tx1);
    font: 500 var(--fs-xs) / 1 var(--font-ui);
    padding: 5px 8px;
    border-radius: var(--r-sm);
    cursor: pointer;
    white-space: nowrap;
  }
  .ann-pop button:hover {
    background: var(--bg4);
    color: var(--tx0);
  }
  .ann-pop :global(.ic) {
    width: 11px;
    height: 11px;
    color: var(--tx3);
  }
  /* art skin uses the compact mono icon-button */
  .ann-pop.art-pop button.t-iconbtn {
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    padding: 2px 4px;
    gap: 4px;
  }
  .ann-pop.art-pop button.t-iconbtn:hover {
    color: var(--tx0);
    background: var(--bg3);
  }
</style>
