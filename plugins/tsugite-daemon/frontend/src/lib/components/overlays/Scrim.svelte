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

<!-- Suppression is correct here: click-to-dismiss is a pointer-only convenience;
     the keyboard path is Esc, owned by the dialog this wraps (Modal/Picker/Palette
     all implement it). The handler must stay on .t-scrim itself (a unit test
     clicks it directly), and .t-scrim can't take an interactive role/become a
     <button> - it wraps a dialog, so that would nest interactive controls (axe
     nested-interactive). A key handler on this presentation backdrop would be
     dead code (it never holds focus). -->
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
