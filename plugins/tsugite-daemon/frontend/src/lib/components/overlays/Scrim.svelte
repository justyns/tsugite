<script lang="ts">
  import type { Snippet } from 'svelte';

  // Shared full-viewport backdrop for centered overlays (Modal composes it).
  // Presentation-only: the dialog it wraps owns focus + the Esc key path. Kept
  // mounted and toggled via `.is-open` (display none↔flex) so the modal's enter
  // animation replays on every open.
  let {
    open = false,
    onclose,
    children,
  }: {
    open?: boolean;
    onclose?: () => void;
    children: Snippet;
  } = $props();

  function onBackdropClick(event: MouseEvent) {
    // Only a click on the scrim itself (not one bubbled up from the modal)
    // dismisses; the modal is a child, so its clicks have a different target.
    if (event.target === event.currentTarget) onclose?.();
  }
</script>

<!-- Backdrop dismissal is a pointer convenience; the keyboard equivalent (Esc)
     lives on the dialog inside. The scrim is role="presentation", so it
     intentionally carries no key handler of its own. -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="t-scrim" class:is-open={open} role="presentation" onclick={onBackdropClick}>
  {@render children()}
</div>

<style>
  .t-scrim {
    position: fixed;
    inset: 0;
    z-index: 250;
    background: color-mix(in oklab, var(--bg0) 55%, transparent);
    backdrop-filter: blur(2px);
    display: none;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  .t-scrim.is-open {
    display: flex;
  }
</style>
