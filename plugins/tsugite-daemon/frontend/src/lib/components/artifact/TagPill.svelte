<script lang="ts">
  // Workspace tag chip: `#tag` that indexes across files. Renders as a link
  // when navigable, a button when it only fires a callback, or a removable
  // filter pill with an `x`. Transcribes the .t-chip primitive inline.
  // Kept inline rather than the shared Chip: this renders as <a>/<button> when
  // navigable (Chip is always a <span>), and its remove-× glyph is 10px where
  // Chip's is 9px - swapping would change the element type and shrink the ×.
  let {
    tag,
    href,
    count,
    removable = false,
    onSelect,
    onRemove,
  }: {
    tag: string;
    href?: string;
    /** Cross-file occurrence count, shown inside the chip as `#tag · N`. */
    count?: number;
    removable?: boolean;
    onSelect?: (tag: string) => void;
    onRemove?: (tag: string) => void;
  } = $props();

  const interactive = $derived(!removable && (href != null || onSelect != null));
</script>

{#snippet label()}#{tag}{#if count != null}<span class="cnt"> · {count}</span>{/if}{/snippet}

{#if removable}
  <span class="t-chip">
    {@render label()}
    <button
      type="button"
      class="x"
      aria-label={`Remove tag ${tag}`}
      onclick={() => onRemove?.(tag)}
    >
      <svg class="ic" viewBox="0 0 16 16" aria-hidden="true"
        ><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" /></svg
      >
    </button>
  </span>
{:else if href}
  <a class="t-chip" {href} onclick={() => onSelect?.(tag)}>{@render label()}</a>
{:else if interactive}
  <button type="button" class="t-chip" onclick={() => onSelect?.(tag)}>{@render label()}</button>
{:else}
  <span class="t-chip">{@render label()}</span>
{/if}

<style>
  .t-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
    cursor: default;
  }
  .cnt {
    color: var(--tx3);
  }
  button.t-chip,
  a.t-chip {
    cursor: pointer;
  }
  a.t-chip:hover,
  button.t-chip:hover {
    border-color: var(--acc);
    color: var(--acc);
    text-decoration: none;
  }
  .x {
    cursor: pointer;
    color: var(--tx3);
    display: inline-flex;
    background: none;
    border: 0;
    padding: 0;
    margin-right: -2px;
  }
  .x:hover {
    color: var(--st-err);
  }
  .ic {
    width: 10px;
    height: 10px;
    flex: none;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
</style>
