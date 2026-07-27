<script lang="ts">
  // Native select dressed as .t-selectw (custom chevron, no native arrow).
  // Options are plain strings - value === label for every option
  // (model id, reasoning effort, scope); a { value, label } shape would be
  // speculative until a real screen needs it.
  import Icon from '$lib/components/icon/Icon.svelte';

  let {
    options,
    value = $bindable(options[0] ?? ''),
    id,
    ariaLabel,
    onchange,
  }: {
    options: string[];
    value?: string;
    id?: string;
    ariaLabel?: string;
    /** Fired on a user selection (not on programmatic `value` writes). */
    onchange?: (value: string) => void;
  } = $props();
</script>

<span class="t-selectw">
  <select {id} aria-label={ariaLabel} bind:value onchange={() => onchange?.(value)}>
    {#each options as opt (opt)}
      <option value={opt}>{opt}</option>
    {/each}
  </select>
  <Icon name="chev-d" />
</span>

<style>
  .t-selectw {
    position: relative;
    display: inline-flex;
    align-items: center;
  }
  .t-selectw select {
    appearance: none;
    height: 24px;
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    color: var(--tx1);
    font: 500 var(--fs-xs) var(--font-mono);
    padding: 0 22px 0 8px;
    cursor: pointer;
  }
  .t-selectw select:hover {
    border-color: var(--tx3);
    color: var(--tx0);
  }
  .t-selectw :global(.ic) {
    position: absolute;
    right: 6px;
    width: 10px;
    height: 10px;
    color: var(--tx3);
    pointer-events: none;
  }
</style>
