<script lang="ts">
  // Pill.
  // The canonical state-language form: color + icon + text together ("nothing
  // relies on color alone"). `st` drives the color and icon; the visible label
  // defaults to the state word itself.
  import type { PillState } from './pill-state';
  import { startSpin } from './spin';

  let {
    // Named `st` (matching the `data-st` attribute), not `state` —
    // a prop literally named `state` shadows the `$state` rune sigil and
    // svelte-check misreads `$state(...)` below as a legacy store
    // auto-subscription of it.
    st,
    /** Overrides the visible label; defaults to the state word (e.g. "busy"). */
    label,
  }: {
    st: PillState;
    label?: string;
  } = $props();

  let frame = $state('⠋');

  $effect(() => {
    if (st !== 'busy') return;
    return startSpin((glyph) => (frame = glyph));
  });

  const text = $derived(label ?? st);
</script>

<span class="t-pill" data-st={st}>
  {#if st === 'idle'}
    <svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="4.5" /></svg>
  {:else if st === 'busy'}
    <span class="t-spin" aria-hidden="true">{frame}</span>
  {:else if st === 'streaming'}
    <span class="ic-stream" aria-hidden="true"><i></i><i></i><i></i></span>
  {:else if st === 'compacting'}
    <svg class="ic" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M4 3l4 3.4L12 3" /><path d="M4 13l4-3.4 4 3.4" />
    </svg>
  {:else if st === 'interrupted'}
    <svg class="ic" viewBox="0 0 16 16" aria-hidden="true"
      ><path d="M6 4v8M10 4v8" stroke-width="2" /></svg
    >
  {/if}
  {text}
</span>

<style>
  /* ---- status pill ---- */
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
  .t-pill .ic {
    width: 11px;
    height: 11px;
  }
  .t-pill[data-st='idle'] {
    /* tx2, not st-mute: --c is also the pill's 11px label color, and st-mute
       misses 4.5:1 on the tinted chip background. */
    --c: var(--tx2);
  }
  .t-pill[data-st='busy'] {
    --c: var(--st-ok);
  }
  .t-pill[data-st='streaming'] {
    --c: var(--st-info);
  }
  .t-pill[data-st='compacting'] {
    --c: var(--st-warn);
  }
  .t-pill[data-st='interrupted'] {
    --c: var(--st-warn);
  }

  /* icon base */
  .ic {
    flex: none;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  /* spinner */
  .t-spin {
    font-family: var(--font-mono);
    font-weight: 600;
    display: inline-block;
    width: 1.1ch;
    line-height: 1;
    color: var(--spin-c, currentColor);
    flex: none;
  }

  /* streaming bars */
  .ic-stream {
    display: inline-flex;
    align-items: flex-end;
    gap: 1.5px;
    height: 10px;
    width: 11px;
  }
  .ic-stream i {
    width: 2px;
    background: currentColor;
    border-radius: 1px;
    animation: tbars 1s var(--ease) infinite;
  }
  .ic-stream i:nth-child(1) {
    height: 40%;
    animation-delay: 0ms;
  }
  .ic-stream i:nth-child(2) {
    height: 90%;
    animation-delay: 160ms;
  }
  .ic-stream i:nth-child(3) {
    height: 60%;
    animation-delay: 320ms;
  }
  @keyframes tbars {
    0%,
    100% {
      transform: scaleY(0.5);
    }
    50% {
      transform: scaleY(1);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .ic-stream i {
      animation: none;
      transform: none;
    }
  }
</style>
