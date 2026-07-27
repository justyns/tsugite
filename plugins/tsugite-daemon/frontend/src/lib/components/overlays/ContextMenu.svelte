<script lang="ts" module>
  export interface ContextMenuItem {
    label: string;
    disabled?: boolean;
    danger?: boolean;
    run: () => void;
  }
</script>

<script lang="ts">
  // Shared right-click menu: fixed at the pointer, clamped to the viewport,
  // dismissed by outside-mousedown / Escape. The parent owns the open state
  // ({x, y, ...context} | null) and renders this only while open. An item's
  // `run` executes BEFORE the menu closes - handlers may read state that the
  // close would tear down (lazily-evaluated {@const} in the parent, e.g.).
  let {
    x,
    y,
    label,
    items,
    onclose,
  }: {
    x: number;
    y: number;
    /** aria-label for the menu (also what tests target). */
    label: string;
    items: ContextMenuItem[];
    onclose: () => void;
  } = $props();

  let el = $state<HTMLElement>();
  let cx = $state(0);
  let cy = $state(0);
  $effect(() => {
    cx = x;
    cy = y;
  });
  // Clamp after paint so the menu never opens half off-screen (bottom rows,
  // right edge).
  $effect(() => {
    void cx;
    void cy;
    if (!el) return;
    const r = el.getBoundingClientRect();
    if (r.right > window.innerWidth - 4) cx = Math.max(4, window.innerWidth - r.width - 4);
    if (r.bottom > window.innerHeight - 4) cy = Math.max(4, y - r.height);
  });

  $effect(() => {
    // Contains-check, not stopPropagation: the item's own mousedown must never
    // race the dismiss listener.
    const down = (e: MouseEvent) => {
      if (el && el.contains(e.target as Node)) return;
      onclose();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onclose();
    };
    window.addEventListener('mousedown', down);
    window.addEventListener('keydown', key);
    return () => {
      window.removeEventListener('mousedown', down);
      window.removeEventListener('keydown', key);
    };
  });

  function pick(item: ContextMenuItem) {
    if (item.disabled) return;
    item.run();
    onclose();
  }
</script>

<div
  class="ctxmenu"
  role="menu"
  aria-label={label}
  tabindex="-1"
  style="--x:{cx}px;--y:{cy}px"
  bind:this={el}
>
  {#each items as item (item.label)}
    <button
      type="button"
      role="menuitem"
      class:is-danger={item.danger}
      disabled={item.disabled}
      onclick={() => pick(item)}
    >
      {item.label}
    </button>
  {/each}
</div>

<style>
  .ctxmenu {
    position: fixed;
    left: var(--x);
    top: var(--y);
    z-index: 80;
    min-width: 132px;
    display: flex;
    flex-direction: column;
    padding: 4px;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
  }
  .ctxmenu button {
    display: flex;
    align-items: center;
    padding: 6px 9px;
    border: 0;
    border-radius: var(--r-sm);
    background: transparent;
    color: var(--tx1);
    font: 500 var(--fs-sm) var(--font-ui);
    text-align: left;
    cursor: pointer;
    white-space: nowrap;
  }
  .ctxmenu button:hover:not(:disabled),
  .ctxmenu button:focus-visible {
    background: var(--bg4);
    color: var(--tx0);
    outline: none;
  }
  .ctxmenu button:disabled {
    color: var(--tx3);
    cursor: default;
  }
  .ctxmenu button.is-danger {
    color: var(--st-err);
  }
  .ctxmenu button.is-danger:hover:not(:disabled) {
    background: color-mix(in oklab, var(--st-err) 14%, transparent);
    color: var(--st-err);
  }
</style>
