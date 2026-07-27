<script lang="ts">
  import Divider from './Divider.svelte';
  import PaneView from './PaneView.svelte';
  import MuxNode from './MuxNode.svelte';
  import type { LayoutNode } from './layout';
  import type { MuxContent, MuxHandlers } from './types';

  let {
    node,
    focusedPaneId,
    multiPane,
    content,
    ...handlers
  }: {
    node: LayoutNode;
    focusedPaneId: string | null;
    /** True when the whole layout has more than one pane (drives focus rings). */
    multiPane: boolean;
    content?: MuxContent;
  } & MuxHandlers = $props();

  /** First pane's share of a divider's pair, as a whole percent for aria-valuenow. */
  function pairPct(sizes: number[], i: number): number {
    const a = sizes[i] ?? 0;
    const b = sizes[i + 1] ?? 0;
    const total = a + b;
    return total > 0 ? Math.round((a / total) * 100) : 50;
  }
</script>

{#if node.type === 'leaf'}
  <PaneView
    pane={node}
    focused={node.id === focusedPaneId}
    showRing={multiPane}
    {content}
    {...handlers}
  />
{:else}
  <div class="mux-split" class:is-col={node.dir === 'col'} role="group" aria-label="Split panes">
    {#each node.children as child, i (child.id)}
      <div class="mux-cell" style="--fr:{node.sizes[i]}">
        <MuxNode node={child} {focusedPaneId} {multiPane} {content} {...handlers} />
      </div>
      {#if i < node.children.length - 1}
        <Divider
          dir={node.dir}
          splitId={node.id}
          index={i}
          valueNow={pairPct(node.sizes, i)}
          onResize={handlers.onResize}
        />
      {/if}
    {/each}
  </div>
{/if}

<style>
  .mux-split {
    display: flex;
    flex-direction: row;
    flex: 1;
    min-width: 0;
    min-height: 0;
  }
  .mux-split.is-col {
    flex-direction: column;
  }
  .mux-cell {
    display: flex;
    flex: var(--fr, 1) 1 0;
    /* A pane never shrinks below readability; when splits outgrow the frame,
       the pane area scrolls (.mux-panes has overflow-x auto). */
    min-width: 260px;
    min-height: 0;
    overflow: hidden;
  }
</style>
