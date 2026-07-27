<script lang="ts">
  // Segmented control (.t-seg): a single-choice button group (reasoning
  // effort, layout, etc). A plain `role="group"`
  // of buttons has no aria-selected state - we add `aria-pressed` so the
  // active segment isn't signaled by background color alone, and roving
  // tabindex + arrow keys so it behaves like the radiogroup it represents.
  import { nextRovingIndex } from '$lib/actions/rovingNav';

  let {
    options,
    value = $bindable(options[0]),
    ariaLabel,
    onchange,
  }: {
    options: string[];
    value?: string;
    ariaLabel: string;
    /** Fired on a user selection (not on programmatic `value` writes). */
    onchange?: (value: string) => void;
  } = $props();

  let groupEl: HTMLDivElement;

  function select(opt: string) {
    value = opt;
    onchange?.(opt);
  }

  function onKeydown(e: KeyboardEvent, current: number) {
    const next = nextRovingIndex(current, e.key, options.length);
    const target = next === null ? undefined : options[next];
    if (next === null || target === undefined) return;
    e.preventDefault();
    select(target);
    const buttons = groupEl.querySelectorAll<HTMLButtonElement>('button[data-seg]');
    buttons[next]?.focus();
  }
</script>

<div class="t-seg" role="group" aria-label={ariaLabel} bind:this={groupEl}>
  {#each options as opt, i (opt)}
    {@const active = opt === value}
    <button
      type="button"
      data-seg
      class:is-active={active}
      aria-pressed={active}
      tabindex={active ? 0 : -1}
      onclick={() => select(opt)}
      onkeydown={(e) => onKeydown(e, i)}
    >
      {opt}
    </button>
  {/each}
</div>
