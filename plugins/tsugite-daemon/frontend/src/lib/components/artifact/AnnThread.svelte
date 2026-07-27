<script lang="ts">
  import { untrack } from 'svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  // A margin-anchored annotation thread. Anchors to a highlighted span in the
  // artifact and carries the comment plus its lifecycle: open (reply/resolve),
  // editing (textarea + save/cancel), resolved (dimmed, reopenable). State is
  // signalled in text too, never colour alone.
  // `status` (not `state`) so the lifecycle prop doesn't shadow the $state rune.
  let {
    author,
    anchor,
    when,
    body,
    status = 'open',
    onReply,
    onResolve,
    onSave,
    onCancel,
    onReopen,
  }: {
    author: string;
    anchor?: string;
    when: string;
    body: string;
    status?: 'open' | 'editing' | 'resolved';
    onReply?: () => void;
    onResolve?: () => void;
    onSave?: (text: string) => void;
    onCancel?: () => void;
    onReopen?: () => void;
  } = $props();

  // editable copy seeded once from the incoming comment
  let draft = $state(untrack(() => body));
</script>

<div
  class="ann-card"
  class:is-editing={status === 'editing'}
  class:is-resolved={status === 'resolved'}
>
  <div class="aw">
    <b>{author}</b>{#if anchor}
      · on “{anchor}”{/if} · {when}{#if status === 'editing'}
      · editing{:else if status === 'resolved'}
      · resolved{/if}
  </div>

  {#if status === 'editing'}
    <textarea class="at_in" aria-label="Edit annotation" bind:value={draft}></textarea>
    <div class="fx">
      <Button variant="pri" size="sm" onclick={() => onSave?.(draft)}>
        {#snippet icon()}<Icon name="check" size={11} />{/snippet}Save
      </Button>
      <Button variant="ghost" size="sm" onclick={() => onCancel?.()}>Cancel</Button>
    </div>
  {:else}
    <div class="at_">{body}</div>
    <div class="fx">
      {#if status === 'resolved'}
        <span class="mk"
          ><svg class="ic" viewBox="0 0 16 16" aria-hidden="true"
            ><path d="M3.5 8.5l3 3 6-6.5" /></svg
          >Resolved</span
        >
        <Button variant="ghost" size="sm" onclick={() => onReopen?.()}>Reopen</Button>
      {:else}
        <Button variant="ghost" size="sm" onclick={() => onReply?.()}>Reply</Button>
        <Button variant="ghost" size="sm" onclick={() => onResolve?.()}>
          {#snippet icon()}<Icon name="check" size={11} />{/snippet}Resolve
        </Button>
      {/if}
    </div>
  {/if}
</div>

<style>
  .ann-card {
    border: 1px solid color-mix(in oklab, var(--st-warn) 35%, transparent);
    border-left: 3px solid var(--st-warn);
    border-radius: var(--r-md);
    background: var(--bg2);
    padding: 8px 10px;
    display: grid;
    gap: 5px;
    font-size: var(--fs-sm);
  }
  .aw {
    display: flex;
    gap: 7px;
    align-items: center;
    flex-wrap: wrap;
    font: 600 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .aw b {
    color: var(--st-warn);
  }
  .at_ {
    color: var(--tx1);
    line-height: 1.5;
  }
  .at_in {
    width: 100%;
    min-height: 44px;
    resize: vertical;
    margin: 0;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 7px 9px;
    color: var(--tx0);
    font: 400 var(--fs-md) var(--font-ui);
    line-height: 1.5;
  }
  .at_in:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
  .fx {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .mk {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font: 600 var(--fs-2xs) var(--font-mono);
    color: var(--st-mute);
  }
  /* editing: accent frame */
  .ann-card.is-editing {
    border-left-color: var(--acc);
    border-color: color-mix(in oklab, var(--acc) 45%, transparent);
    background: color-mix(in oklab, var(--acc) 5%, var(--bg2));
  }
  .ann-card.is-editing .aw b {
    color: var(--acc);
  }
  /* resolved: muted + quiet */
  .ann-card.is-resolved {
    border-color: var(--bd0);
    border-left-color: var(--st-mute);
    opacity: 0.72;
  }
  .ann-card.is-resolved .aw b {
    color: var(--st-mute);
  }

  .ic {
    width: 13px;
    height: 13px;
    flex: none;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .mk .ic {
    width: 11px;
    height: 11px;
  }
</style>
