<script lang="ts">
  // Shell for the chat's raw-inspection overlays. Rides the shared Scrim (backdrop
  // + click-away); Esc, the focus trap and focus restore live here. The caller sizes
  // it, fills the header's action slot and the scrolling body, and dresses its own
  // buttons in the shared `.raw-copy` face.
  import { onMount, type Snippet } from 'svelte';
  import Scrim from '$lib/components/overlays/Scrim.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import { trapFocus } from '$lib/actions/trapFocus';

  let {
    title,
    width = '920px',
    testid,
    onClose,
    actions,
    children,
  }: {
    title: string;
    width?: string;
    testid?: string;
    onClose: () => void;
    actions?: Snippet;
    children: Snippet;
  } = $props();

  let dialogEl = $state<HTMLElement | null>(null);
  let restoreTo: HTMLElement | null = null;

  onMount(() => {
    restoreTo = document.activeElement as HTMLElement | null;
    dialogEl?.focus();
    return () => restoreTo?.focus();
  });

  function onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.stopPropagation();
      onClose();
    }
  }
</script>

<Scrim open onclose={onClose}>
  <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
  <div
    class="raw"
    style="width: min({width}, 100%)"
    role="dialog"
    aria-modal="true"
    aria-label={title}
    tabindex="-1"
    bind:this={dialogEl}
    onkeydown={onKeydown}
    use:trapFocus
    data-testid={testid}
  >
    <div class="raw-hd">
      <h3>{title}</h3>
      <div class="raw-hd-r">
        {#if actions}{@render actions()}{/if}
        <Button variant="ghost" size="sm" iconOnly aria-label="Close" onclick={onClose}>
          {#snippet icon()}<Icon name="x" />{/snippet}
        </Button>
      </div>
    </div>
    <div class="raw-bd">{@render children()}</div>
  </div>
</Scrim>

<style>
  .raw {
    max-height: min(86vh, 900px);
    display: flex;
    flex-direction: column;
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-3);
    overflow: hidden;
  }
  .raw-hd {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 9px;
    padding: 11px 14px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
  }
  .raw-hd h3 {
    margin: 0;
    font: 600 var(--fs-lg) / 1.3 var(--font-ui);
  }
  .raw-hd-r {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .raw-bd {
    overflow-y: auto;
    padding: 13px 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  /* Header and body content are caller-provided (snippets), so the shared button
     face is :global, scoped under this dialog. */
  .raw :global(.raw-copy) {
    flex: none;
    background: none;
    border: 1px solid var(--bd1);
    border-radius: var(--r-sm);
    color: var(--tx2);
    font: 500 var(--fs-2xs) var(--font-mono);
    padding: 2px 7px;
    cursor: pointer;
  }
  .raw :global(.raw-copy:hover) {
    background: var(--bg3);
    color: var(--tx0);
  }
  @media (max-width: 640px) {
    .raw {
      max-height: 92vh;
    }
  }
</style>
