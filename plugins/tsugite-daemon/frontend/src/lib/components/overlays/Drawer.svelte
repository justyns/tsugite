<script lang="ts">
  import { untrack, type Snippet } from 'svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  // Right-side inspection panel. Slides in over a positioned pane, non-modal
  // (role="complementary" - you can tab back to the page, so no focus trap),
  // Esc closes it, and it's `inert` while off-screen so its content stays out of
  // the tab order + a11y tree during the slide. Mount inside a
  // position:relative frame; it shares the modal's scrim + motion tokens.
  let {
    open = false,
    onclose,
    title,
    label,
    status,
    children,
    footer,
  }: {
    open?: boolean;
    onclose?: () => void;
    title: string;
    label?: string;
    status?: Snippet;
    children: Snippet;
    footer?: Snippet;
  } = $props();

  const titleId = $props.id();

  let rootEl = $state<HTMLElement | null>(null);
  let restoreTo: HTMLElement | null = null;
  // Snapshot the initial open state once; the $effect below tracks transitions.
  let wasOpen = untrack(() => open);

  // Move focus to the close affordance only on a genuine closed→open transition
  // (a specimen that mounts already-open never grabs focus), and restore it to
  // the opener on close.
  $effect(() => {
    if (open === wasOpen) return;
    if (open) {
      restoreTo = document.activeElement as HTMLElement | null;
      // The close control is a shared <Button>, which owns its own <button> and
      // doesn't forward a DOM ref - reach it by its label to focus it on open.
      rootEl?.querySelector<HTMLButtonElement>('[aria-label="Close detail"]')?.focus();
    } else {
      restoreTo?.focus();
      restoreTo = null;
    }
    wasOpen = open;
  });

  function onKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      event.stopPropagation();
      onclose?.();
    }
  }
</script>

<!-- Explicit role="complementary": an <aside> nested in sectioning content maps
     to `generic`, so the role is kept to guarantee the semantic in any pane.
     Esc-to-close is scoped here so it only fires while focus is in the drawer. -->
<!-- svelte-ignore a11y_no_redundant_roles -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<aside
  bind:this={rootEl}
  class="t-drawer"
  class:is-open={open}
  role="complementary"
  aria-label={label}
  aria-labelledby={label ? undefined : titleId}
  inert={!open}
  onkeydown={onKeydown}
>
  <div class="t-drawer-hd">
    {#if status}{@render status()}{/if}
    <h3 id={titleId}>{title}</h3>
    <Button
      variant="ghost"
      size="sm"
      iconOnly
      aria-label="Close detail"
      onclick={() => onclose?.()}
    >
      {#snippet icon()}<Icon name="x" />{/snippet}
    </Button>
  </div>
  <div class="t-drawer-bd">{@render children()}</div>
  {#if footer}
    <div class="t-drawer-ft">{@render footer()}</div>
  {/if}
</aside>

<style>
  .t-drawer {
    position: absolute;
    inset: 0 0 0 auto;
    width: min(480px, 94%);
    background: var(--bg1);
    border-left: 1px solid var(--bd1);
    box-shadow: var(--sh-3);
    translate: 102% 0;
    transition: translate var(--t-3) var(--ease);
    z-index: 30;
    display: flex;
    flex-direction: column;
  }
  .t-drawer.is-open {
    translate: 0 0;
  }
  .t-drawer-hd {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 11px 14px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
  }
  .t-drawer-hd h3 {
    margin: 0;
    font: 600 var(--fs-lg) / 1.3 var(--font-ui);
    flex: 1;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .t-drawer-bd {
    overflow-y: auto;
    padding: 13px 14px 18px;
    display: grid;
    gap: 16px;
    align-content: start;
  }
  .t-drawer-ft {
    flex: none;
    display: flex;
    gap: 8px;
    padding: 10px 14px;
    border-top: 1px solid var(--bd0);
    background: var(--bg2);
  }
  /* Narrow: the drawer becomes a full-width sheet. */
  @media (max-width: 640px) {
    .t-drawer {
      width: 100%;
      border-left: 0;
    }
  }
</style>
