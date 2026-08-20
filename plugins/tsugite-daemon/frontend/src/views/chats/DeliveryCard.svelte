<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Prose from '$lib/components/chatturns/Prose.svelte';
  import { TESTID } from '$lib/testids';
  import type { DeliveryBlock } from './turns';

  let {
    block,
    outstanding,
    onDismiss,
  }: {
    block: DeliveryBlock;
    outstanding: boolean;
    onDismiss: () => void;
  } = $props();
</script>

<div
  class="dlv"
  class:is-ack={block.needsAck}
  role={block.needsAck ? 'alert' : 'status'}
  data-testid={TESTID.chatDelivery}
>
  <div class="dlv-hd">
    <Icon name={block.needsAck ? 'alert' : 'down'} size={12} />
    <span>{block.source || 'delivery'}</span>
    {#if outstanding}<span class="dlv-tag">needs you</span>{/if}
  </div>
  {#if block.title}<div class="dlv-title">{block.title}</div>{/if}
  <Prose content={block.message} />
  {#if outstanding}
    <Button size="sm" data-testid={TESTID.chatDeliveryDismiss} onclick={onDismiss}>
      {#snippet icon()}<Icon name="check" size={11} />{/snippet}
      Dismiss
    </Button>
  {/if}
</div>

<style>
  .dlv {
    justify-self: start;
    max-width: 100%;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 11px;
    border: 1px solid var(--bd0);
    border-left: 2px solid var(--tx3);
    background: var(--bg1);
    border-radius: var(--r-md);
  }
  .dlv.is-ack {
    border-left-color: var(--st-warn);
    background: color-mix(in oklab, var(--st-warn) 7%, var(--bg1));
  }
  .dlv-hd {
    display: flex;
    align-items: center;
    gap: 7px;
    color: var(--tx3);
    font: 500 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .dlv.is-ack .dlv-hd :global(svg) {
    color: var(--st-warn);
  }
  .dlv-tag {
    color: var(--st-warn);
  }
  .dlv-title {
    color: var(--tx0);
    font: 600 var(--fs-sm) var(--font-ui);
  }
  .dlv :global(.t-btn) {
    align-self: flex-start;
  }
</style>
