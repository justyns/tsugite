<script lang="ts">
  // Theme segmented control (.t-seg) wired straight to the theme store, so the
  // top bar and the settings drawer share one source of truth and stay in sync.
  // The store's .set() persists + reskins; active is derived from .current, so
  // no local mirror is needed. Roving tabindex + arrow keys match Seg's a11y.
  import { theme } from '$lib/stores/theme.svelte';
  import { nextRovingIndex } from '$lib/actions/rovingNav';

  let { testid }: { testid?: string } = $props();

  let groupEl: HTMLDivElement;

  function onKeydown(e: KeyboardEvent, current: number) {
    const next = nextRovingIndex(current, e.key, theme.list.length);
    if (next === null) return;
    const target = theme.list[next];
    if (!target) return;
    e.preventDefault();
    theme.set(target);
    groupEl.querySelectorAll<HTMLButtonElement>('button')[next]?.focus();
  }
</script>

<div
  class="t-seg t-seg--ui"
  role="group"
  aria-label="Theme"
  bind:this={groupEl}
  data-testid={testid}
>
  {#each theme.list as name, i (name)}
    {@const active = name === theme.current}
    <button
      type="button"
      class:is-active={active}
      aria-pressed={active}
      tabindex={active ? 0 : -1}
      onclick={() => theme.set(name)}
      onkeydown={(e) => onKeydown(e, i)}>{name}</button
    >
  {/each}
</div>
