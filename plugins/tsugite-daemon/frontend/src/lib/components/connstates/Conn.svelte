<script lang="ts">
  // Nav-rail connection indicator.
  // Mirrors the store's on-the-wire state 1:1 - never busier or calmer than the daemon.
  let {
    state,
    reconnectAttempt = 0,
    onRetry,
  }: {
    state: 'on' | 're' | 'off';
    /** Shown as "(n)" next to "reconnecting…" once a retry has actually fired. */
    reconnectAttempt?: number;
    /** "retry now" only renders (and is only reachable) once the stream has given up. */
    onRetry?: () => void;
  } = $props();
</script>

<span class="t-conn" data-st={state} role="status">
  <span class="t-dot"></span>
  <span class="lbl-on">connected</span>
  <span class="lbl-re"
    >reconnecting…{#if reconnectAttempt > 0}{' '}<span class="mono">({reconnectAttempt})</span
      >{/if}</span
  >
  <span class="lbl-off"
    >offline{#if onRetry}{' '}<button type="button" class="retry" onclick={onRetry}
        >retry now</button
      >{/if}</span
  >
</span>

<style>
  .t-conn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font: 500 var(--fs-xs)/1 var(--font-mono);
    color: var(--tx2);
    white-space: nowrap;
  }
  .t-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--c, var(--st-mute));
    flex: none;
    display: inline-block;
  }
  .t-conn .t-dot {
    width: 7px;
    height: 7px;
  }
  .t-conn[data-st='on'] .t-dot {
    --c: var(--st-ok);
  }
  .t-conn[data-st='on'] .lbl-re,
  .t-conn[data-st='on'] .lbl-off {
    display: none;
  }
  .t-conn[data-st='re'] .t-dot {
    --c: var(--st-warn);
    animation: tpulse 1.1s var(--ease) infinite;
  }
  .t-conn[data-st='re'] {
    color: var(--st-warn);
  }
  .t-conn[data-st='re'] .lbl-on,
  .t-conn[data-st='re'] .lbl-off {
    display: none;
  }
  .t-conn[data-st='off'] .t-dot {
    --c: var(--st-err);
  }
  .t-conn[data-st='off'] {
    color: var(--st-err);
  }
  .t-conn[data-st='off'] .lbl-on,
  .t-conn[data-st='off'] .lbl-re {
    display: none;
  }
  @keyframes tpulse {
    0%,
    100% {
      box-shadow: 0 0 0 0 color-mix(in oklab, var(--c, var(--acc)) 45%, transparent);
    }
    55% {
      box-shadow: 0 0 0 5px transparent;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .t-conn[data-st='re'] .t-dot {
      animation: none;
    }
  }
  .mono {
    font-family: var(--font-mono);
  }
  .retry {
    margin-left: 2px;
    padding: 0;
    border: 0;
    background: none;
    color: var(--acc);
    font: inherit;
    cursor: pointer;
  }
  .retry:hover {
    color: color-mix(in oklab, var(--acc) 78%, var(--tx0));
    text-decoration: underline;
  }
</style>
