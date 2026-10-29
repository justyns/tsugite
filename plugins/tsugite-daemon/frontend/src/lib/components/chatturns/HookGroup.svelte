<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import ExecBlock from './ExecBlock.svelte';
  import type { ExecBlock as Hook } from '../../../views/chats/turns';

  let { hooks, meta }: { hooks: Hook[]; meta?: string } = $props();

  let open = $state(false);
</script>

<div class="t-hooks" class:is-open={open}>
  <button type="button" class="hd" aria-expanded={open} onclick={() => (open = !open)}>
    <span class="chev"><Icon name="chev-r" size={10} /></span>
    <Icon name="term" size={13} color="var(--tx3)" />
    <span class="title">{hooks.length} hooks</span>
    {#if meta}<span class="meta">· {meta}</span>{/if}
  </button>
  {#if open}
    <div class="rows">
      {#each hooks as hook, i (i)}
        <ExecBlock
          command={hook.command}
          status={hook.status}
          exitCode={hook.exitCode}
          output={hook.output}
          meta={hook.meta}
        />
      {/each}
    </div>
  {/if}
</div>

<style>
  .t-hooks {
    font-family: var(--font-mono);
    font-size: var(--fs-sm);
  }
  .hd {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 6px 9px;
    background: none;
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    color: var(--tx1);
    font: inherit;
    text-align: left;
    cursor: pointer;
    user-select: none;
  }
  .hd:hover {
    background: var(--bg2);
  }
  .chev {
    color: var(--tx3);
    display: inline-flex;
    transition: rotate var(--t-2) var(--ease);
  }
  .is-open .chev {
    rotate: 90deg;
  }
  .meta {
    color: var(--tx3);
    font-size: var(--fs-2xs);
    font-variant-numeric: tabular-nums;
  }
  .rows {
    display: grid;
    gap: 4px;
    margin-top: 4px;
    padding-left: 14px;
  }
  @media (prefers-reduced-motion: reduce) {
    .chev {
      transition: none;
    }
  }
</style>
