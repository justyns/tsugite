<script lang="ts">
  // Agent-generated inline UI - the model authors an ephemeral control (a
  // choice here) to collect one decision inline. Tagged "generated UI" so it is
  // never confused with host chrome. The pick returns to the agent to continue
  // the turn. Presentational + callback prop.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { nextRovingIndex } from '$lib/actions/rovingNav';

  let {
    question,
    choices,
    selected = null,
    label = 'model-authored · ephemeral',
    onPick,
  }: {
    question: string;
    choices: string[];
    selected?: number | null;
    label?: string;
    onPick?: (index: number) => void;
  } = $props();

  let picked = $state<number | null>(selected);
  const done = $derived(picked !== null);
  // Roving focus starts on the picked choice, else the first.
  let focusIndex = $state(selected ?? 0);
  let btns = $state<(HTMLButtonElement | undefined)[]>([]);

  function pick(i: number) {
    if (done) return;
    picked = i;
    onPick?.(i);
  }

  function focusAt(index: number) {
    focusIndex = index;
    btns[index]?.focus();
  }

  function onKeydown(e: KeyboardEvent) {
    if (done) return;
    const next = nextRovingIndex(focusIndex, e.key, choices.length);
    if (next !== null) {
      e.preventDefault();
      focusAt(next);
      return;
    }
    // Digit-jump: keys 1..N pick the matching choice directly.
    const d = Number(e.key);
    if (Number.isInteger(d) && d >= 1 && d <= choices.length) {
      e.preventDefault();
      pick(d - 1);
    }
  }
</script>

<div class="t-genui">
  <div class="gu-hd">
    <Icon name="sparkle" />generated UI<span class="ep">{label}</span>
  </div>
  <div class="gu-body" class:is-done={done}>
    <div class="gu-q">{question}</div>
    <!-- Roving tabindex lives on the radios; the group itself is not a tab stop. -->
    <!-- svelte-ignore a11y_interactive_supports_focus -->
    <div class="gu-choices" role="radiogroup" aria-label={question} onkeydown={onKeydown}>
      {#each choices as choice, i (choice)}
        <button
          type="button"
          class="gu-choice"
          class:is-sel={picked === i}
          role="radio"
          aria-checked={picked === i}
          tabindex={i === focusIndex ? 0 : -1}
          bind:this={btns[i]}
          onclick={() => pick(i)}><span class="num">{i + 1}</span>{choice}</button
        >
      {/each}
    </div>
  </div>
</div>

<style>
  /* ===== agent-generated inline UI (owned here) ===== */
  .t-genui {
    border: 1px solid color-mix(in oklab, var(--brand) 38%, var(--bd1));
    background: color-mix(in oklab, var(--brand) 5%, var(--bg1));
    border-radius: var(--r-lg);
    overflow: hidden;
    max-width: 540px;
  }
  .t-genui .gu-hd {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 6px 11px;
    border-bottom: 1px solid color-mix(in oklab, var(--brand) 18%, var(--bd0));
    font: 600 var(--fs-2xs) var(--font-mono);
    color: var(--brand);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .t-genui .gu-hd :global(.ic) {
    width: 12px;
    height: 12px;
  }
  .t-genui .gu-hd .ep {
    margin-left: auto;
    color: var(--tx3);
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0;
  }
  .t-genui .gu-body {
    padding: 12px;
    display: grid;
    gap: 11px;
  }
  .t-genui .gu-q {
    font-size: var(--fs-md);
    color: var(--tx0);
    line-height: 1.45;
    text-wrap: pretty;
  }
  .t-genui .gu-choices {
    display: grid;
    gap: 6px;
  }
  .t-genui .gu-choice {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 8px 11px;
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    background: var(--bg1);
    cursor: pointer;
    font-size: var(--fs-sm);
    color: var(--tx1);
    text-align: left;
    width: 100%;
    font-family: inherit;
  }
  .t-genui .gu-choice:hover {
    border-color: var(--brand);
    background: var(--bg2);
  }
  .t-genui .gu-choice.is-sel {
    border-color: var(--brand);
    background: color-mix(in oklab, var(--brand) 10%, var(--bg1));
  }
  .t-genui .gu-choice .num {
    width: 18px;
    height: 18px;
    flex: none;
    display: grid;
    place-items: center;
    border-radius: var(--r-sm);
    background: var(--bg3);
    color: var(--tx3);
    font: 600 var(--fs-2xs) var(--font-mono);
  }
  .t-genui .gu-choice.is-sel .num {
    background: var(--brand);
    color: var(--on-brand);
  }
  .t-genui .gu-body.is-done .gu-choice:not(.is-sel) {
    opacity: 0.5;
  }
  .t-genui .gu-body.is-done .gu-choice {
    cursor: default;
    pointer-events: none;
  }
</style>
