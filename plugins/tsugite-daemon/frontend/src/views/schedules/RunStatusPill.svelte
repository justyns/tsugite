<script lang="ts">
  // Run-status pill for a schedule's last run. The shared <Pill> is scoped to
  // session states (idle/busy/streaming/...), so this defines a `.t-pill` for
  // the schedule/run vocabulary instead. Color + icon + text together, never
  // color alone (state-language contract).
  import Icon from '$lib/components/icon/Icon.svelte';
  import type { IconName } from '$lib/components/icon/icons';
  import type { RunStatus } from './schedulesView';

  let { status }: { status: RunStatus } = $props();

  // `st` selects the color token; label + icon carry the meaning for
  // anyone who can't perceive the tint.
  const MAP: Record<RunStatus, { st: string; icon: IconName; label: string }> = {
    done: { st: 'done', icon: 'check', label: 'done' },
    errored: { st: 'errored', icon: 'x', label: 'errored' },
    skipped: { st: 'idle', icon: 'dot', label: 'skipped' },
    queued: { st: 'queued', icon: 'clock', label: 'waiting' },
    off: { st: 'idle', icon: 'ring', label: 'off' },
  };
  const m = $derived(MAP[status]);
</script>

<span class="t-pill" data-st={m.st}>
  <Icon name={m.icon} size={11} />
  {m.label}
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
  .t-pill[data-st='errored'] {
    --c: var(--st-err);
  }
  .t-pill[data-st='done'] {
    --c: var(--st-mute);
  }
  .t-pill[data-st='queued'] {
    --c: var(--st-queue);
  }
  .t-pill[data-st='idle'] {
    --c: var(--st-mute);
  }
</style>
