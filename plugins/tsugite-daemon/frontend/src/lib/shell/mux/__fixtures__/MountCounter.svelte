<script module lang="ts">
  /** Mounts per tab id, so a test can tell a kept-alive surface from a rebuilt one. */
  export const mountCounts = new Map<string, number>();
</script>

<script lang="ts">
  import { onMount } from 'svelte';

  let { id }: { id: string } = $props();

  onMount(() => mountCounts.set(id, (mountCounts.get(id) ?? 0) + 1));
</script>

<div data-mounted={id}></div>
