<script module lang="ts">
  export interface PickItem {
    /** Returned to onPick when the row is chosen. */
    value: string;
    /** Primary text, matched against and rendered. */
    label: string;
    /** Optional trailing meta shown right-aligned. */
    detail?: string;
  }
</script>

<script lang="ts">
  import Scrim from '$lib/components/overlays/Scrim.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { pwmIgnore } from '$lib/components/inputs/pwmIgnore';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { matchItem, type MatchResult } from '$lib/components/palette/palette-match';
  import { TESTID } from '$lib/testids';

  // Generic searchable-list overlay: a modal search box over a caller-supplied
  // list, returning the picked value. No command/session/api coupling - a
  // provider with a large option set opens this instead of an inline submenu.
  let {
    items,
    title,
    placeholder = 'search…',
    onPick,
    onClose,
  }: {
    items: PickItem[];
    title?: string;
    placeholder?: string;
    onPick: (value: string) => void;
    onClose: () => void;
  } = $props();

  const uid = $props.id();
  const listId = `${uid}-ls`;
  const optionId = (i: number) => `${uid}-opt-${i}`;

  const ui = $state({ query: '', selected: 0 });
  let listEl = $state<HTMLDivElement | undefined>();

  // Filter + score + order via the shared palette matcher (an empty query keeps
  // every item in source order). Ties fall back to source order for stability.
  const matches = $derived.by(() => {
    const q = ui.query.trim();
    return items
      .map((item, order) => ({ item, order, match: matchItem(q, item.label) }))
      .filter((m): m is typeof m & { match: MatchResult } => m.match !== null)
      .sort((a, b) => b.match.score - a.match.score || a.order - b.order);
  });
  const hasResults = $derived(matches.length > 0);
  const trimmed = $derived(ui.query.trim());

  // Keep the roving highlight inside the (possibly shrunk) result set.
  $effect(() => {
    if (ui.selected > matches.length - 1) ui.selected = Math.max(0, matches.length - 1);
  });

  // Scroll the active option into view as selection moves.
  $effect(() => {
    void ui.selected;
    void matches;
    listEl?.querySelector('.is-sel')?.scrollIntoView({ block: 'nearest' });
  });

  function move(delta: number) {
    const max = matches.length - 1;
    if (max < 0) return;
    ui.selected = Math.min(Math.max(ui.selected + delta, 0), max);
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      move(1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      move(-1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const m = matches[ui.selected];
      if (m) onPick(m.item.value);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  }

  function autofocus(node: HTMLInputElement) {
    requestAnimationFrame(() => node.focus());
  }
</script>

<Scrim open onclose={onClose}>
  <div
    class="t-picker"
    role="dialog"
    aria-modal="true"
    aria-label={title ?? 'Picker'}
    tabindex="-1"
    data-testid={TESTID.picker}
    use:trapFocus
  >
    {#if title}<div class="pk-title">{title}</div>{/if}

    <div class="pk-in">
      <Icon name="search" />
      <input
        type="search"
        {placeholder}
        {...pwmIgnore}
        spellcheck="false"
        role="combobox"
        aria-label={title ?? 'Search'}
        aria-autocomplete="list"
        aria-controls={listId}
        aria-expanded={hasResults}
        aria-activedescendant={hasResults ? optionId(ui.selected) : undefined}
        bind:value={ui.query}
        oninput={() => (ui.selected = 0)}
        onkeydown={onKeydown}
        use:autofocus
      />
      <span class="t-kbd">esc</span>
    </div>

    <div class="pk-ls" id={listId} role="listbox" aria-label="Results" bind:this={listEl}>
      {#if hasResults}
        {#each matches as m, i (m.item.value)}
          {@const isSel = ui.selected === i}
          <!-- Listbox option in the combobox+activedescendant pattern: it stays
               non-focusable (tabindex=-1, tracked by the input's
               aria-activedescendant), and Arrow/Enter selection lives on the
               combobox input above. onclick is a pointer convenience; a per-option
               key handler would be dead (options never take focus) and a <button>
               would join the shared trapFocus Tab cycle. Suppression is correct: -->
          <!-- svelte-ignore a11y_click_events_have_key_events -->
          <div
            class="pk-it"
            class:is-sel={isSel}
            id={optionId(i)}
            role="option"
            aria-selected={isSel}
            tabindex="-1"
            data-testid={TESTID.pickerOption(m.item.value)}
            onclick={() => onPick(m.item.value)}
            onmousemove={() => (ui.selected = i)}
          >
            <span class="lbl"
              >{#if m.match.highlight}{@const hl = m.match.highlight}{m.item.label.slice(
                  0,
                  hl[0],
                )}<b>{m.item.label.slice(hl[0], hl[1])}</b>{m.item.label.slice(hl[1])}{:else}{m.item
                  .label}{/if}</span
            >
            {#if m.item.detail}<span class="detail">{m.item.detail}</span>{/if}
          </div>
        {/each}
      {:else}
        <div class="pk-empty">
          {#if trimmed}No matches for “{trimmed}”.{:else}Nothing to pick.{/if}
        </div>
      {/if}
    </div>
  </div>
</Scrim>

<style>
  .t-picker {
    width: min(520px, 100%);
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-3);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    max-height: min(460px, 76vh);
    animation: toastin var(--t-3) var(--ease);
  }
  .pk-title {
    padding: 11px 13px 0;
    font: 600 var(--fs-md) / 1.3 var(--font-ui);
    color: var(--tx0);
    flex: none;
  }
  .pk-in {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 11px 13px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
  }
  .pk-in :global(.ic) {
    color: var(--tx3);
    width: 14px;
    height: 14px;
  }
  .pk-in input {
    flex: 1;
    min-width: 0;
    background: none;
    border: 0;
    outline: none;
    color: var(--tx0);
    font: 400 var(--fs-lg) var(--font-ui);
  }
  .pk-in input::placeholder {
    color: var(--tx3);
  }
  /* type=search opts out of Chromium's password manager but pulls in the UA
     clear button; drop it so the field looks the same as before. */
  .pk-in input::-webkit-search-cancel-button,
  .pk-in input::-webkit-search-decoration {
    -webkit-appearance: none;
    appearance: none;
  }
  .pk-ls {
    overflow-y: auto;
    padding: 4px;
    flex: 1;
  }
  .pk-it {
    display: flex;
    gap: 9px;
    align-items: center;
    padding: 6px 10px;
    border-radius: var(--r-md);
    cursor: pointer;
    color: var(--tx1);
    font-size: var(--fs-md);
    min-height: 32px;
  }
  .pk-it .lbl {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .pk-it .lbl b {
    color: var(--acc);
    font-weight: 600;
  }
  .pk-it.is-sel {
    background: color-mix(in oklab, var(--acc) 14%, transparent);
    color: var(--tx0);
  }
  .pk-it .detail {
    margin-left: auto;
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    white-space: nowrap;
    flex: none;
  }
  .pk-empty {
    padding: 20px;
    text-align: center;
    color: var(--tx3);
    font-size: var(--fs-sm);
  }

  @keyframes toastin {
    from {
      translate: 0 8px;
      opacity: 0;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .t-picker {
      animation-duration: 0.01ms;
    }
  }
</style>
