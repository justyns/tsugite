<script lang="ts">
  // Single stat tile: big value + small mono label, optional delta note and
  // semantic tone on the value (the tone enum maps to the state tokens in the
  // stylesheet below).
  export type StatTone = 'ok' | 'warn' | 'err' | 'info';

  export type StatItem = {
    value: string;
    label: string;
    tone?: StatTone;
    delta?: string;
  };

  let { value, label, tone, delta }: StatItem = $props();
</script>

<div class="t-stat">
  <span class="v" data-tone={tone}>{value}</span>
  <span class="l">{label}</span>
  {#if delta}
    <span class="d">{delta}</span>
  {/if}
</div>

<style>
  .t-stat {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 8px 10px;
    background: var(--bg2);
    display: grid;
    gap: 3px;
  }
  .t-stat .v {
    font: 600 var(--fs-2xl) / 1 var(--font-ui);
    color: var(--tx0);
  }
  .t-stat .v[data-tone='ok'] {
    color: var(--st-ok);
  }
  .t-stat .v[data-tone='warn'] {
    color: var(--st-warn);
  }
  .t-stat .v[data-tone='err'] {
    color: var(--st-err);
  }
  .t-stat .v[data-tone='info'] {
    color: var(--st-info);
  }
  .t-stat .l {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .t-stat .d {
    font: 500 var(--fs-2xs) var(--font-mono);
  }
</style>
