<script lang="ts">
  // Enable/disable toggle idiom (.t-sw) for schedules, skills, webhooks.
  // A real <button role="switch"> gets native Enter/Space activation for
  // free; state is announced via aria-checked, never via the thumb color
  // shift alone (the thumb also slides position).
  let {
    checked = $bindable(false),
    ariaLabel,
    onCheckedChange,
  }: {
    checked?: boolean;
    ariaLabel: string;
    /** Fires after a user toggle with the new value. Lets a controlled consumer
     *  (server-backed rows: schedules, skills, webhooks) persist the change -
     *  bind:checked alone offers no seam for that. */
    onCheckedChange?: (checked: boolean) => void;
  } = $props();

  function toggle() {
    checked = !checked;
    onCheckedChange?.(checked);
  }
</script>

<button
  type="button"
  class="t-sw"
  role="switch"
  aria-checked={checked}
  aria-label={ariaLabel}
  onclick={toggle}
></button>

<style>
  .t-sw {
    position: relative;
    width: 30px;
    height: 17px;
    border-radius: var(--r-full);
    background: var(--bg4);
    border: 1px solid var(--bd1);
    cursor: pointer;
    transition: background var(--t-2) var(--ease);
    flex: none;
    padding: 0;
  }
  .t-sw::after {
    content: '';
    position: absolute;
    top: 1.5px;
    left: 2px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--tx1);
    transition:
      translate var(--t-2) var(--ease),
      background var(--t-2);
  }
  .t-sw[aria-checked='true'] {
    background: color-mix(in oklab, var(--st-ok) 55%, var(--bg4));
    border-color: transparent;
  }
  .t-sw[aria-checked='true']::after {
    translate: 12px 0;
    background: var(--bg0);
  }
</style>
