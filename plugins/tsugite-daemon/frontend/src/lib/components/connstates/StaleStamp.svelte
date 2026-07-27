<script lang="ts">
  // Stamped on a live pane/header while the SSE stream is down. Nothing shown
  // is trusted until resync - never busier or calmer than the daemon.
  // Presence = visible: the caller decides
  // whether to mount this at all (no internal is-stale gating to coordinate
  // with other groups' pane markup).
  import Icon from '$lib/components/icon/Icon.svelte';
  import { formatStale } from './format';

  let {
    since = null,
  }: {
    /** epoch ms the connection went stale; omit for the bare "stale" label. */
    since?: number | null;
  } = $props();

  let now = $state(Date.now());

  $effect(() => {
    if (since == null) return;
    const id = setInterval(() => {
      now = Date.now();
    }, 1000);
    return () => clearInterval(id);
  });

  const label = $derived(formatStale(since, now));
</script>

<span class="t-stale">
  <Icon name="clock" size={9} />
  <span class="st-el">{label}</span>
</span>

<style>
  .t-stale {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font: 500 var(--fs-2xs)/1 var(--font-mono);
    color: var(--st-warn);
    background: color-mix(in oklab, var(--st-warn) 12%, transparent);
    border: 1px solid color-mix(in oklab, var(--st-warn) 30%, transparent);
    border-radius: var(--r-full);
    padding: 2px 7px;
    white-space: nowrap;
  }
</style>
