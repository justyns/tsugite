<script lang="ts">
  // Mirrors the chrome's mux wiring (App.svelte): every docked surface is keyed
  // by tab id, so a tab the mux unmounts rebuilds its surface on the way back.
  import SizedMux from './SizedMux.svelte';
  import type { Layout } from '../layout';
  import type { MuxHandlers } from '../types';
  import MountCounter from './MountCounter.svelte';

  let { layout, ...handlers }: { layout: Layout } & MuxHandlers = $props();
</script>

<SizedMux {layout} {...handlers}>
  {#snippet content(tab)}
    {#key tab.id}
      <MountCounter id={tab.id} />
    {/key}
  {/snippet}
</SizedMux>
