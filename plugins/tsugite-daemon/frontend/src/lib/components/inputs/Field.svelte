<script lang="ts">
  // Label + control + hint/error wrapper (.t-field). Field is presentational
  // only - it doesn't own the control, so it hands the computed message id
  // to the `children` snippet for the caller's control to wire onto its own
  // `aria-describedby` (and to set `aria-invalid` itself when `error` is set,
  // same as the card: the ring + icon + text all live on the real control).
  import type { Snippet } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  let {
    id,
    label,
    hint,
    error,
    children,
  }: {
    id: string;
    label: string;
    hint?: string;
    error?: string;
    children: Snippet<[describedBy: string | undefined]>;
  } = $props();

  const describedBy = $derived(error || hint ? `${id}-msg` : undefined);
</script>

<div class="t-field">
  <label for={id}>{label}</label>
  {@render children(describedBy)}
  {#if error}
    <span class="msg" id={describedBy}>
      <Icon name="alert" size={10} />
      {error}
    </span>
  {:else if hint}
    <span class="hint" id={describedBy}>{hint}</span>
  {/if}
</div>

<style>
  .t-field {
    display: grid;
    gap: 5px;
  }
  .t-field label {
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  .msg {
    font-size: var(--fs-xs);
    color: var(--st-err);
    display: flex;
    gap: 4px;
    align-items: center;
  }
  .hint {
    font: 400 var(--fs-2xs) var(--font-ui);
    color: var(--tx3);
    line-height: 1.5;
  }
</style>
