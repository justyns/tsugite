<script lang="ts">
  // Responsive wrap of Stat tiles - grid-template-columns:repeat(auto-fit,...);
  // reuses Stat rather than re-declaring the .t-stat rules.
  import Stat from './Stat.svelte';
  import type { StatItem } from './Stat.svelte';

  let { stats }: { stats: StatItem[] } = $props();
</script>

<div class="t-statgrid">
  {#each stats as stat (stat.label)}
    <Stat value={stat.value} label={stat.label} tone={stat.tone} delta={stat.delta} />
  {/each}
</div>

<style>
  .t-statgrid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
    gap: 8px;
  }
</style>
