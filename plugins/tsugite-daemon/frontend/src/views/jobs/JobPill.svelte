<script lang="ts">
  // Job-state pill. The shared `Pill` component only models session states
  // (idle/busy/streaming/...); the job state-language is a separate set, so it
  // is rendered here off `jobPillMeta`. Pills use `.t-pill` styling with
  // per-state `data-st` colour rules, showing colour + icon + text together.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { startSpin } from '$lib/components/buttons/spin';
  import { jobPillMeta, jobPillState } from './jobPill';

  // Named `st` (not `state`) so it can't shadow the `$state` rune below.
  let { st }: { st: string } = $props();

  const meta = $derived(jobPillMeta(st));
  const token = $derived(jobPillState(st));

  let frame = $state('⠋');
  $effect(() => {
    if (!meta.spin) return;
    return startSpin((glyph) => (frame = glyph));
  });
</script>

<span class="t-pill" data-st={token}>
  {#if meta.spin}
    <span class="t-spin" aria-hidden="true">{frame}</span>
  {:else}
    <Icon name={meta.icon} />
  {/if}
  {meta.label}
</span>

<style>
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
  .t-pill[data-st='verifying'] {
    --c: var(--st-verify);
  }
  .t-pill[data-st='awaiting'] {
    --c: var(--st-warn);
  }
  .t-pill[data-st='stuck'] {
    --c: var(--st-warn);
  }
  .t-pill[data-st='errored'] {
    --c: var(--st-err);
  }
  .t-pill[data-st='done'] {
    --c: var(--st-mute);
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
    color: var(--spin-c, currentColor);
    flex: none;
  }
</style>
