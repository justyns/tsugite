<script lang="ts">
  // One provider's quota report: its name, then a meter per window or the error.
  // Shared by the Usage view cards and the chat header quota popover.
  import Meter from '$lib/components/datadisplay/Meter.svelte';
  import type { UsageProviderRow } from '$lib/stores/usage.svelte';
  import { QUOTA_WARN_PCT, formatPercent, formatResetAt } from './format';

  let { row }: { row: UsageProviderRow } = $props();
</script>

<div class="u-prov-body">
  <span class="u-prov-name">{row.label}</span>
  {#if row.error}
    <span class="u-prov-err">{row.error}</span>
  {:else}
    {#each row.windows as win}
      {@const reset = formatResetAt(win.resets_at)}
      <div class="u-win">
        <span class="u-win-name">{win.label}</span>
        <Meter
          value={win.used_pct}
          max={100}
          label="{row.label} {win.label}"
          displayText={formatPercent(win.used_pct)}
          warn={win.used_pct >= QUOTA_WARN_PCT}
        />
        {#if reset}<span class="u-win-reset">{reset}</span>{/if}
      </div>
    {/each}
  {/if}
</div>

<style>
  .u-prov-body {
    display: grid;
    gap: 7px;
    min-width: 0;
  }
  .u-prov-name {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
  }
  .u-prov-err {
    font-size: var(--fs-xs);
    color: var(--tx2);
  }
  .u-win {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    min-width: 0;
  }
  .u-win-name {
    flex: 1;
    min-width: 0;
    font-size: var(--fs-xs);
    color: var(--tx2);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .u-win-reset {
    font: var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
  }
</style>
