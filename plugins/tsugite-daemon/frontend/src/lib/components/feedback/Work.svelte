<script lang="ts">
  // The working/streaming line.
  // "The working block is the only spinner the chat ever shows - it always
  // names the operation, ticks elapsed time, and offers Stop. A lonely
  // spinner is banned by design." Stop renders when a handler is given; a
  // caller whose surface already owns a Stop control (the chat composer)
  // omits it rather than doubling the affordance.
  //
  // `detail` is this group's progress-detail extension:
  // an optional trailing clause like "turn 3 · 2 tools · tool: bash", styled
  // as a dimmer aside after the operation name. Caller pre-formats the
  // string (same "·"-joined convention used for meta text elsewhere,
  // e.g. .t-exec-hd .meta) - kept a plain string so this stays a
  // two-prop addition instead of a speculative structured schema.
  import Spin from './Spin.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import { formatElapsed } from './format';

  let {
    verb = 'running',
    operation,
    detail,
    startedAt,
    reconnecting = false,
    onStop,
  }: {
    verb?: string;
    operation: string;
    detail?: string;
    /** epoch ms the operation started; the elapsed readout ticks from here. */
    startedAt: number;
    /** SSE dropped mid-run - flips the border/spinner to the warn color and,
     * since state is never color-alone, adds a "reconnecting…" status flag. */
    reconnecting?: boolean;
    onStop?: () => void;
  } = $props();

  let elapsedLabel = $state(formatElapsed(0));

  $effect(() => {
    const tick = () => {
      elapsedLabel = formatElapsed((Date.now() - startedAt) / 1000);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  });
</script>

<div class="t-work" class:is-re={reconnecting}>
  <Spin color={reconnecting ? 'var(--st-warn)' : 'var(--st-ok)'} />
  {#if reconnecting}<span class="re-flag" role="status">reconnecting…</span>{/if}
  <span
    >{verb} <b>{operation}</b>{#if detail}<span class="wk-detail">{` · ${detail}`}</span>{/if}</span
  >
  <span class="el">{elapsedLabel}</span>
  <span class="grow"></span>
  {#if onStop}
    <Button variant="danger" size="sm" onclick={onStop}>
      {#snippet icon()}<Icon name="stop" />{/snippet}
      Stop
    </Button>
  {/if}
</div>

<style>
  .t-work {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 7px 11px;
    border: 1px dashed color-mix(in oklab, var(--st-ok) 40%, transparent);
    border-radius: var(--r-md);
    background: color-mix(in oklab, var(--st-ok) 7%, transparent);
    font: 500 var(--fs-sm) var(--font-mono);
    color: var(--tx1);
  }
  /* Ideally `.t-work .t-spin{color:...}`; Spin is a child component so
     scoped CSS can't reach its internals - the color is passed as a prop
     above instead (see Toast.svelte for the same issue with its icon). */
  .t-work .el {
    color: var(--tx3);
    font-variant-numeric: tabular-nums;
  }
  .t-work.is-re {
    border-color: color-mix(in oklab, var(--st-warn) 45%, transparent);
    background: color-mix(in oklab, var(--st-warn) 8%, transparent);
  }
  /* role="status" lives on this flag alone, not the whole .t-work - the
     block's own .el elapsed readout ticks every second, and an atomic live
     region wrapping it would re-announce the full sentence every tick. */
  .re-flag {
    color: var(--st-warn);
    font-weight: 600;
  }
  .wk-detail {
    color: var(--tx3);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
</style>
