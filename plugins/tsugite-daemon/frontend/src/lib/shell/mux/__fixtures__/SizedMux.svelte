<script lang="ts">
  // A Mux inside a host of fixed height, the way the app's pane area bounds it.
  // narrow={false} forces the desktop split tree. The test viewport would
  // otherwise trip the <=700px single-pane mode.
  import Mux from '../Mux.svelte';
  import type { Layout } from '../layout';
  import type { MuxContent, MuxHandlers } from '../types';

  let {
    layout,
    height = 800,
    content,
    ...handlers
  }: {
    layout: Layout;
    height?: number;
    content?: MuxContent;
  } & MuxHandlers = $props();
</script>

<div class="host" style="--h:{height}px">
  <Mux {layout} narrow={false} {content} {...handlers} />
</div>

<style>
  .host {
    display: flex;
    flex-direction: column;
    height: var(--h);
  }
</style>
