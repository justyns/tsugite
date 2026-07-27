<script lang="ts">
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  // Review footer for an artifact pane: the reviewer resolves with Approve or
  // Request changes; once resolved the buttons give way to the verdict, which
  // is announced (role="status") and carries an icon + text, not colour alone.
  let {
    state = 'pending',
    approveLabel = 'Approve plan',
    changesLabel = 'Request changes',
    note,
    approvedText = 'Approved',
    changesText = 'Changes requested',
    onApprove,
    onRequestChanges,
  }: {
    state?: 'pending' | 'approved' | 'changes';
    approveLabel?: string;
    changesLabel?: string;
    note?: string;
    approvedText?: string;
    changesText?: string;
    onApprove?: () => void;
    onRequestChanges?: () => void;
  } = $props();
</script>

<div class="art-ft" class:is-approved={state === 'approved'} class:is-changes={state === 'changes'}>
  {#if state === 'pending'}
    {#if note}<span class="n">{note}</span>{/if}
  {:else}
    <span class="verdict" role="status">
      {#if state === 'approved'}
        <svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-6.5" /></svg
        >{approvedText}
      {:else}
        <svg class="ic" viewBox="0 0 16 16" aria-hidden="true"
          ><path d="M3 13l.9-3L10.4 3.5l2.1 2.1L6 12.1z" /></svg
        >{changesText}
      {/if}
    </span>
  {/if}
  <div class="grow"></div>
  {#if state === 'pending'}
    <Button size="sm" onclick={() => onRequestChanges?.()}>
      {#snippet icon()}<Icon name="edit" size={11} />{/snippet}{changesLabel}
    </Button>
    <Button variant="pri" size="sm" onclick={() => onApprove?.()}>
      {#snippet icon()}<Icon name="check" size={11} />{/snippet}{approveLabel}
    </Button>
  {/if}
</div>

<style>
  .art-ft {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 9px 11px;
    border-top: 1px solid var(--bd0);
    background: var(--bg2);
    flex-wrap: wrap;
  }
  .art-ft.is-approved {
    background: color-mix(in oklab, var(--st-ok) 10%, var(--bg2));
  }
  .art-ft.is-changes {
    background: color-mix(in oklab, var(--st-warn) 10%, var(--bg2));
  }
  .n {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .verdict {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font: 600 var(--fs-sm) var(--font-mono);
  }
  .art-ft.is-approved .verdict {
    color: var(--st-ok);
  }
  .art-ft.is-changes .verdict {
    color: var(--st-warn);
  }

  .ic {
    width: 11px;
    height: 11px;
    flex: none;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
</style>
