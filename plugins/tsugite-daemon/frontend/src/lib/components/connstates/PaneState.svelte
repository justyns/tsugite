<script lang="ts">
  // "Every list renders one of five things: rows, skeleton (loading), empty
  // (with the way forward), error (with retry + inspect), or permission.
  // There is no blank div in this system." - one component, kind prop.
  import type { Snippet } from 'svelte';
  import { paneSkeletonWidths } from './format';

  let {
    kind,
    title = '',
    hint,
    icon,
    actions,
    lines = 4,
  }: {
    kind: 'empty' | 'loading' | 'error' | 'permission';
    /** ignored for kind="loading" */
    title?: string;
    /** ignored for kind="loading"; can hold rich content e.g. <span class="mono"> */
    hint?: Snippet;
    /** ignored for kind="loading"; caller supplies a bare <svg viewBox="0 0 16 16">, sized/colored by this component */
    icon?: Snippet;
    /** ignored for kind="loading" */
    actions?: Snippet;
    /** kind="loading" only - number of skeleton bars */
    lines?: number;
  } = $props();

  const widths = $derived(paneSkeletonWidths(lines));
</script>

{#if kind === 'loading'}
  <div class="pane-loading" aria-busy="true" aria-label="Loading">
    <div class="t-skel">
      {#each widths as w, i (i)}
        <i style="--w:{w}%"></i>
      {/each}
    </div>
  </div>
{:else}
  <div class="t-pane" class:t-pane--err={kind === 'error'}>
    {#if icon}
      <span class="pane-ic">{@render icon()}</span>
    {/if}
    <span class="pt">{title}</span>
    {#if hint}
      <span class="ph">{@render hint()}</span>
    {/if}
    {#if actions}
      <span class="pa">{@render actions()}</span>
    {/if}
  </div>
{/if}

<style>
  .t-pane {
    border: 1px dashed var(--bd1);
    border-radius: var(--r-lg);
    padding: 22px 16px;
    display: grid;
    justify-items: center;
    gap: 6px;
    text-align: center;
    color: var(--tx2);
  }
  .pane-ic {
    width: 18px;
    height: 18px;
    color: var(--tx3);
  }
  .pane-ic :global(svg) {
    width: 100%;
    height: 100%;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .t-pane .pt {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx1);
  }
  .t-pane .ph {
    font: 400 var(--fs-xs) / 1.5 var(--font-ui);
    /* tx2, not tx3: 11px hint text on the tinted pane background
       misses the 4.5:1 contrast contract. */
    color: var(--tx2);
    max-width: 30ch;
    text-wrap: pretty;
  }
  .t-pane .pa {
    margin-top: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .t-pane--err {
    border-style: solid;
    border-color: color-mix(in oklab, var(--st-err) 35%, transparent);
    background: color-mix(in oklab, var(--st-err) 5%, transparent);
  }
  .t-pane--err .pane-ic,
  .t-pane--err .pt {
    color: var(--st-err);
  }
  .pane-loading {
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
    padding: 14px 16px;
  }
  .t-skel {
    display: grid;
    gap: 9px;
    padding: 6px 0;
  }
  .t-skel i {
    display: block;
    width: var(--w, 100%);
    height: 11px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--bg3) 25%, var(--bg4) 45%, var(--bg3) 65%);
    background-size: 200% 100%;
    animation: tshimmer 1.4s linear infinite;
  }
  @keyframes tshimmer {
    to {
      background-position: -200% 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .t-skel i {
      animation: none;
    }
  }
</style>
