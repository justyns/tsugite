<script lang="ts">
  // Mirrors the chrome's mux wiring (App.svelte): every docked surface is keyed
  // by tab id, so a tab the mux unmounts rebuilds its surface on the way back.
  import Mux from '../Mux.svelte';
  import type { Layout } from '../layout';
  import type { MuxHandlers } from '../types';
  import MountCounter from './MountCounter.svelte';

  let { layout, ...handlers }: { layout: Layout } & MuxHandlers = $props();
</script>

<Mux {layout} narrow={false} {...handlers}>
  {#snippet content(tab)}
    {#key tab.id}
      <MountCounter id={tab.id} />
    {/key}
  {/snippet}
</Mux>
