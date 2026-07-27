<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    bordered = false,
    terminal = false,
    label,
    tabs,
    header,
    children,
    panelId,
    activeTabId,
  }: {
    /** Standalone frame (border + radius). Off in the split grid, where the
     *  1px grid gap draws the divider between panes. */
    bordered?: boolean;
    /** Body is a terminal (dark background, mono type). */
    terminal?: boolean;
    /** Accessible name for the pane region. */
    label?: string;
    /** Tab strip, stacked above the header. */
    tabs?: Snippet;
    /** Pane header (title · state · model · context · split/close). */
    header?: Snippet;
    /** Scrollable pane body. */
    children?: Snippet;
    /** When the pane hosts a TabStrip, pass the same panelId to both: the body
     *  becomes the role="tabpanel" the tabs control (WAI-ARIA tabs pattern). */
    panelId?: string;
    /** Active tab id; labels the tabpanel via `{panelId}-tab-{activeTabId}`. */
    activeTabId?: string;
  } = $props();
</script>

<section class="mux-pane" class:is-framed={bordered} aria-label={label}>
  {@render tabs?.()}
  {@render header?.()}
  {#if children}
    <div
      class="mux-bd"
      class:is-term={terminal}
      id={panelId}
      role={panelId ? 'tabpanel' : undefined}
      aria-labelledby={panelId && activeTabId ? `${panelId}-tab-${activeTabId}` : undefined}
    >
      {@render children()}
    </div>
  {/if}
</section>

<style>
  .mux-pane {
    display: flex;
    flex-direction: column;
    background: var(--bg0);
    min-width: 0;
    min-height: 0;
  }
  /* Card presents the pane in a standalone frame; the live split grid omits it. */
  .mux-pane.is-framed {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow: hidden;
  }
  /* A flex column, so surfaces that declare `flex: 1; min-height: 0` fill the
     pane and own their internal scrolling (the chat timeline must scroll under
     a sticky composer, never the pane body as one rigid block). Block-level
     content still overflows here and scrolls as before. */
  .mux-bd {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    min-height: 0;
  }
  /* #14161f is the literal terminal background (not a surface token). */
  .mux-bd.is-term {
    background: #14161f;
    padding: 8px 12px;
    font: 400 var(--fs-xs) / 1.6 var(--font-mono);
  }
</style>
