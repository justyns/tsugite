<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import { clampPct, spaceStateMeta, type SpaceState } from './rowState';

  let {
    title,
    who,
    state,
    contextPct,
    contextTokens,
    contextWarn = false,
    isActive = false,
    onSelect,
  }: {
    title: string;
    who: string;
    state: SpaceState;
    /** 0-100. Out-of-range values are clamped for display. */
    contextPct: number;
    /** Pre-formatted token count, e.g. "34k" - the caller owns rounding. */
    contextTokens: string;
    /** Caller-decided "context getting tight" warning; this only renders it. */
    contextWarn?: boolean;
    isActive?: boolean;
    onSelect?: () => void;
  } = $props();

  const meta = $derived(spaceStateMeta(state));
  const barPct = $derived(clampPct(contextPct));

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect?.();
    }
  }
</script>

<div
  class="sp-row"
  data-st={state}
  class:is-active={isActive}
  role="button"
  tabindex="0"
  onclick={() => onSelect?.()}
  onkeydown={handleKeydown}
>
  <span class="sp-st">
    {#if meta.spin}
      <Spin />
    {:else if meta.icon}
      <Icon name={meta.icon} size={10} />
    {/if}
    {meta.label}
  </span>
  <span class="ttl">{title}</span>
  <span class="who">{who}</span>
  <span
    class="t-meter"
    class:is-warn={contextWarn}
    role="meter"
    aria-valuenow={barPct}
    aria-valuemin={0}
    aria-valuemax={100}
    aria-label="context {barPct}%"
  >
    <span class="bar"><i style="--w:{barPct}%"></i></span>{barPct}% · {contextTokens}
  </span>
</div>

<style>
  .t-meter {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .t-meter .bar {
    width: 56px;
    height: 3px;
    background: var(--bg3);
    border-radius: var(--r-full);
    overflow: hidden;
  }
  .t-meter .bar i {
    display: block;
    width: var(--w, 0%);
    height: 100%;
    background: var(--tx3);
  }
  .t-meter.is-warn .bar i {
    background: var(--st-warn);
  }

  .sp-row {
    display: grid;
    grid-template-columns: 74px 1fr auto;
    grid-template-rows: auto auto;
    gap: 1px 8px;
    padding: 5px 10px 6px;
    border-left: 2px solid transparent;
    cursor: pointer;
    min-width: 0;
    align-items: center;
  }
  .sp-row:hover {
    background: var(--bg2);
  }
  .sp-row.is-active {
    background: var(--bg2);
    border-left-color: var(--acc);
  }
  .sp-row[data-st='blocked'] {
    border-left-color: var(--st-warn);
    background: color-mix(in oklab, var(--st-warn) 6%, transparent);
  }
  .sp-st {
    grid-row: 1 / 3;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font: 500 var(--fs-2xs) / 1 var(--font-mono);
    color: var(--c, var(--tx3));
  }
  .sp-row[data-st='working'] .sp-st {
    --c: var(--st-ok);
  }
  .sp-row[data-st='blocked'] .sp-st {
    --c: var(--st-warn);
    font-weight: 700;
  }
  .sp-row[data-st='idle'] .sp-st {
    --c: var(--tx3);
  }
  .sp-row[data-st='done'] .sp-st {
    --c: var(--st-mute);
  }
  .sp-row .ttl {
    font: 500 var(--fs-sm) / 1.3 var(--font-ui);
    color: var(--tx1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .sp-row .who {
    grid-column: 2;
    font: 400 var(--fs-2xs) / 1.4 var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .sp-row .t-meter {
    grid-row: 1 / 3;
    grid-column: 3;
    flex-direction: column;
    align-items: flex-end;
    gap: 3px;
  }
  .sp-row .t-meter .bar {
    width: 44px;
  }
</style>
