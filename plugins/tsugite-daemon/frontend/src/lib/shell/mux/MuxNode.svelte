<script lang="ts">
  import Divider from './Divider.svelte';
  import PaneView from './PaneView.svelte';
  import type { LayoutNode } from './layout';
  import { MIN_PANE_PX, placeTree } from './place';
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

  const placement = $derived(placeTree(node));

  const minWidth = $derived(Math.max(...placement.panes.map((p) => MIN_PANE_PX / p.w)));

  const pct = (f: number) => `${f * 100}%`;

  function dividerStyle(d: (typeof placement.dividers)[number]): string {
    return d.dir === 'row'
      ? `position:absolute;left:${pct(d.at)};top:${pct(d.cross.y)};height:${pct(d.cross.h)};width:1px;margin-left:-0.5px`
      : `position:absolute;top:${pct(d.at)};left:${pct(d.cross.x)};width:${pct(d.cross.w)};height:1px;margin-top:-0.5px`;
  }
</script>

<!-- Every pane renders once in this flat list, keyed by leaf id, and is placed from the
     layout tree. A component instance cannot move between nested blocks, so nesting
     the panes by tree depth would remount them when a split forms or collapses. -->
<div
  class="mux-split"
  style="min-width:{minWidth}px"
  role={node.type === 'split' ? 'group' : undefined}
  aria-label={node.type === 'split' ? 'Split panes' : undefined}
>
  {#each placement.panes as r (r.leaf.id)}
    <div class="mux-cell" style="left:{pct(r.x)};top:{pct(r.y)};width:{pct(r.w)};height:{pct(r.h)}">
      <PaneView
        pane={r.leaf}
        focused={r.leaf.id === focusedPaneId}
        {multiPane}
        {content}
        {...handlers}
      />
    </div>
  {/each}
  {#each placement.dividers as d (`${d.splitId}:${d.index}`)}
    <Divider
      dir={d.dir}
      splitId={d.splitId}
      index={d.index}
      valueNow={d.pct}
      span={d.span}
      style={dividerStyle(d)}
      onResize={handlers.onResize}
    />
  {/each}
</div>

<style>
  .mux-split {
    position: relative;
    flex: 1;
    min-height: 0;
  }
  .mux-cell {
    position: absolute;
    display: flex;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
  }
</style>
