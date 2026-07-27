<script lang="ts">
  // Terminal status pill. The library Pill (buttons/Pill.svelte) only carries
  // the 5 session states; a terminal's 6-state machine folds onto the
  // job/state-language pill buckets (queued/running/done/errored/cancelled/
  // stuck) via terminalPill(). Same `.t-pill` contract, covering exactly
  // the buckets this view needs.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { startSpin } from '$lib/components/buttons/spin';
  import { terminalPill } from './termState';
  import type { TerminalState } from '$lib/stores/terminals.svelte';

  let {
    // Named `st`, not `state`: a prop literally named `state` shadows the
    // `$state` rune sigil and svelte-check misreads `$state(...)` below as a
    // store auto-subscription (same reason buttons/Pill.svelte uses `st`).
    st,
    exitCode = null,
  }: {
    st: TerminalState;
    exitCode?: number | null;
  } = $props();

  const spec = $derived(terminalPill(st, exitCode));

  let frame = $state('⠋');
  $effect(() => {
    if (!spec.spin) return;
    return startSpin((glyph) => (frame = glyph));
  });
</script>

<span class="t-pill" data-st={spec.st}>
  {#if spec.spin}
    <span class="t-spin" aria-hidden="true">{frame}</span>
  {:else}
    <Icon name={spec.icon} />
  {/if}
  {spec.label}
</span>

<style>
  /* .t-pill for the terminal buckets */
  .t-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 8px 0 7px;
    border-radius: var(--r-full);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    letter-spacing: 0.02em;
    white-space: nowrap;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 13%, transparent);
    border: 1px solid color-mix(in oklab, var(--c) 32%, transparent);
  }
  .t-pill :global(.ic) {
    width: 11px;
    height: 11px;
  }
  .t-pill[data-st='queued'] {
    --c: var(--st-queue);
  }
  .t-pill[data-st='running'] {
    --c: var(--st-ok);
  }
  .t-pill[data-st='errored'] {
    --c: var(--st-err);
  }
  .t-pill[data-st='done'] {
    --c: var(--st-mute);
  }
  .t-pill[data-st='stuck'] {
    --c: var(--st-warn);
  }
  .t-pill[data-st='cancelled'] {
    --c: var(--st-mute);
    opacity: 0.75;
  }

  .t-spin {
    font-family: var(--font-mono);
    font-weight: 600;
    display: inline-block;
    width: 1.1ch;
    line-height: 1;
    flex: none;
  }
</style>
