<script lang="ts">
  import SizedMux from './SizedMux.svelte';
  import MountCounter from './MountCounter.svelte';
  import type { Layout } from '../layout';

  let { layout }: { layout: Layout } = $props();
</script>

<SizedMux {layout}>
  {#snippet content(tab)}
    <MountCounter id={tab.id} />
  {/snippet}
</SizedMux>
