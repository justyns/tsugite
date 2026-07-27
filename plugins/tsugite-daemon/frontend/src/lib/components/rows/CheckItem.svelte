<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import { checkStatePrefix, type CheckState } from './rowState';

  let {
    label,
    state,
    note,
  }: {
    label: string;
    state: CheckState;
    /** Verifier's reason. Design contract: a fail with no note is a dead end,
     * so callers should always supply one for `state: 'fail'`. */
    note?: string;
  } = $props();
</script>

<div class="t-check" data-st={state}>
  <span class="box">
    {#if state === 'active'}
      <Spin />
    {:else if state === 'pass'}
      <Icon name="check" size={9} />
    {:else if state === 'fail'}
      <Icon name="x" size={9} />
    {/if}
  </span>
  <span class="lb"
    ><span class="vh">{checkStatePrefix(state)}: </span>{label}{#if note}<span class="note"
        >{note}</span
      >{/if}</span
  >
</div>

<style>
  .vh {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }

  .t-check {
    display: flex;
    gap: 8px;
    padding: 5px 0;
    font-size: var(--fs-sm);
    align-items: flex-start;
  }
  .t-check .box {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1.5px solid var(--bd1);
    display: grid;
    place-items: center;
    flex: none;
    margin-top: 1px;
    color: transparent;
  }
  .t-check .box :global(.t-spin) {
    font-size: 10px;
    width: auto;
  }
  .t-check .lb {
    color: var(--tx2);
    line-height: 1.45;
  }
  .t-check .note {
    display: block;
    font: 400 var(--fs-xs) / 1.45 var(--font-mono);
    margin-top: 2px;
    color: var(--tx3);
  }
  .t-check[data-st='pass'] .box {
    border-color: var(--st-ok);
    background: color-mix(in oklab, var(--st-ok) 16%, transparent);
    color: var(--st-ok);
  }
  .t-check[data-st='pass'] .lb {
    color: var(--tx1);
  }
  .t-check[data-st='fail'] .box {
    border-color: var(--st-err);
    background: color-mix(in oklab, var(--st-err) 14%, transparent);
    color: var(--st-err);
  }
  .t-check[data-st='fail'] .lb {
    color: var(--tx1);
  }
  .t-check[data-st='fail'] .note {
    color: var(--st-err);
  }
  .t-check[data-st='active'] .box {
    border-color: var(--st-verify);
    color: var(--st-verify);
  }
  .t-check[data-st='active'] .lb {
    color: var(--tx0);
  }
  .t-check[data-st='active'] .lb::after {
    content: ' · verifying…';
    font: 400 var(--fs-xs) var(--font-mono);
    color: var(--st-verify);
  }
</style>
