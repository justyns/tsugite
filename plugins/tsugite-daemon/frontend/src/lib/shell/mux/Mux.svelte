<script lang="ts">
  import MuxNode from './MuxNode.svelte';
  import PaneView from './PaneView.svelte';
  import { type LeafNode, type Layout, collectLeaves } from './layout';
  import type { MuxContent, MuxHandlers } from './types';
  import { TESTID } from '$lib/testids';

  let {
    layout,
    content,
    narrow: narrowProp,
    ...handlers
  }: {
    layout: Layout;
    content?: MuxContent;
    /** Force single-pane mode. Omit to follow the viewport (<=700px). */
    narrow?: boolean;
  } & MuxHandlers = $props();

  const leaves = $derived(collectLeaves(layout.root));
  const multiPane = $derived(leaves.length > 1);
  const focusedLeaf = $derived<LeafNode>(
    leaves.find((l) => l.id === layout.focusedPaneId) ?? leaves[0]!,
  );

  // Phones get one pane + a switcher; the split tree is desktop-only.
  let mqNarrow = $state(false);
  $effect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const m = window.matchMedia('(max-width: 700px)');
    const sync = () => (mqNarrow = m.matches);
    sync();
    m.addEventListener('change', sync);
    return () => m.removeEventListener('change', sync);
  });
  const narrow = $derived(narrowProp ?? mqNarrow);

  function leafLabel(leaf: LeafNode): string {
    const active = leaf.tabs.find((t) => t.id === leaf.activeTabId);
    return active ? (active.title ?? active.kind) : 'empty';
  }

  // Focus recovery: closing a pane or a tab unmounts the element that had keyboard
  // focus, so the browser drops focus to <body>. When that happens, move focus to
  // the surviving focused pane so a keyboard user isn't dumped to the top of the
  // document (WCAG 2.4.3, Focus Order). Every other op leaves focus on a live
  // element, so the <body> check keeps this from hijacking ordinary focus changes;
  // the priming skip keeps it from grabbing focus on first mount.
  let rootEl = $state<HTMLElement>();
  let focusPrimed = false;
  $effect(() => {
    void layout.root; // re-run after every reducer-driven rerender
    if (!focusPrimed) {
      focusPrimed = true;
      return;
    }
    if (document.activeElement !== document.body) return;
    // Built from the shared muxPane testid: the selector doubles as the
    // focus-recovery mechanism, so it must track the same constant PaneView tags.
    rootEl
      ?.querySelector<HTMLElement>(`[data-testid="${TESTID.muxPane}"][data-focused="true"]`)
      ?.focus();
  });
</script>

<div class="mux-root" data-testid={TESTID.mux} bind:this={rootEl}>
  {#if narrow}
    {#if leaves.length > 1}
      <div class="mux-switch" role="group" aria-label="Switch pane" data-testid={TESTID.muxSwitch}>
        {#each leaves as leaf (leaf.id)}
          <button
            type="button"
            class="mux-sw"
            class:is-active={leaf.id === focusedLeaf.id}
            aria-pressed={leaf.id === focusedLeaf.id}
            onclick={() => handlers.onFocusPane?.(leaf.id)}
          >
            {leafLabel(leaf)}
          </button>
        {/each}
      </div>
    {/if}
    <div class="mux-panes">
      <PaneView pane={focusedLeaf} focused showRing={false} narrow {content} {...handlers} />
    </div>
  {:else}
    <div class="mux-panes">
      <MuxNode
        node={layout.root}
        focusedPaneId={layout.focusedPaneId}
        {multiPane}
        {content}
        {...handlers}
      />
    </div>
  {/if}
</div>

<style>
  .mux-root {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
    min-height: 0;
    height: 100%;
    background: var(--bg0);
  }
  /* The pane area; split dividers draw their own 1px lines, so this just needs a
     line-coloured backdrop for the seams. */
  .mux-panes {
    flex: 1;
    display: flex;
    min-width: 0;
    min-height: 0;
    background: var(--bd1);
    overflow-x: auto;
  }

  /* Mobile pane switcher (.mux-switch). */
  .mux-switch {
    display: flex;
    gap: 4px;
    padding: 7px 10px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg1);
    overflow-x: auto;
    scrollbar-width: none;
  }
  .mux-switch::-webkit-scrollbar {
    display: none;
  }
  .mux-sw {
    flex: none;
    max-width: 40vw;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    border: 1px solid var(--bd1);
    background: var(--bg2);
    color: var(--tx2);
    font: 500 var(--fs-xs) / 1 var(--font-ui);
    padding: 6px 10px;
    border-radius: var(--r-md);
    cursor: pointer;
  }
  .mux-sw:hover {
    color: var(--tx0);
  }
  .mux-sw.is-active {
    background: var(--bg0);
    color: var(--tx0);
    border-color: var(--acc);
  }
</style>
