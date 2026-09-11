<script lang="ts">
  // Mirrors the spaces store: the layout lives in `$state`, every reducer gets
  // the proxy itself, and the result is reassigned.
  import Mux from '../Mux.svelte';
  import { type Layout, focusPane, selectTab } from '../layout';
  import MountCounter from './MountCounter.svelte';
  import ParamsProbe from './ParamsProbe.svelte';

  let { initial }: { initial: Layout } = $props();

  // svelte-ignore state_referenced_locally -- seeds once; the reducers own it after.
  let layout = $state<Layout>(initial);

  function apply(fn: (l: Layout) => Layout) {
    layout = fn(layout);
  }
</script>

<Mux
  {layout}
  narrow={false}
  onFocusPane={(paneId) => apply((l) => focusPane(l, paneId))}
  onSelectTab={(paneId, tabId) => apply((l) => selectTab(l, paneId, tabId))}
>
  {#snippet content(tab)}
    {#key tab.id}
      <MountCounter id={tab.id} />
      <ParamsProbe id={tab.id} params={tab.params} />
    {/key}
  {/snippet}
</Mux>
