<script lang="ts">
  import { buildRows, type PaletteItem } from './palette-match';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { pwmIgnore } from '$lib/components/inputs/pwmIgnore';
  import type { IconName } from '$lib/components/icon/icons';
  import { trapFocus } from '$lib/actions/trapFocus';

  let {
    open = $bindable(false),
    items = [],
    sessionItems = [],
    onSelect,
    placeholder = 'jump to session, job, pty, schedule, file…',
    inline = false,
    initialQuery = '',
  }: {
    /** Overlay visibility. Two-way: the app opens it (⌘K), the palette closes itself. */
    open?: boolean;
    /** Searchable index. */
    items?: PaletteItem[];
    /** Chat sessions, query-only: hidden on the default list, surfaced under their
     *  own header once a query matches a title/topic. */
    sessionItems?: PaletteItem[];
    /** Fired when a row is chosen; the app handles navigation / the quick action. */
    onSelect?: (item: PaletteItem) => void;
    placeholder?: string;
    /** Render the panel in-flow (no backdrop, always shown) - for embedding / the gallery. */
    inline?: boolean;
    /** Seed the query (used for gallery states and pre-filled opens). */
    initialQuery?: string;
  } = $props();

  // Unique per instance so ids don't collide when several palettes coexist (gallery).
  const uid = `pal-${Math.random().toString(36).slice(2, 9)}`;
  const listId = `${uid}-ls`;
  const optionId = (i: number) => `${uid}-opt-${i}`;

  const ui = $state({ query: '', selected: 0 });
  let listEl: HTMLDivElement | undefined = $state();
  let opener: HTMLElement | null = null;

  const visible = $derived(inline || open);
  const rows = $derived(buildRows(items, ui.query, sessionItems));
  const itemRows = $derived(rows.filter((r) => r.kind === 'item'));
  const hasResults = $derived(itemRows.length > 0);
  const trimmed = $derived(ui.query.trim());

  // Seed the query: inline embeds show it straight away; the overlay (re)seeds and
  // captures the opener each time it opens, BEFORE its input steals focus.
  $effect.pre(() => {
    if (inline) {
      ui.query = initialQuery;
      return;
    }
    if (!open) return;
    opener = (document.activeElement as HTMLElement) ?? null;
    ui.query = initialQuery;
    ui.selected = 0;
    return () => opener?.focus?.();
  });

  // Keep the active option scrolled into view as selection moves.
  $effect(() => {
    void ui.selected;
    void rows;
    listEl?.querySelector('.is-sel')?.scrollIntoView({ block: 'nearest' });
  });

  function move(delta: number) {
    const max = itemRows.length - 1;
    if (max < 0) return;
    ui.selected = Math.min(Math.max(ui.selected + delta, 0), max);
  }

  function choose(item: PaletteItem) {
    if (!inline) open = false;
    onSelect?.(item);
  }

  function onKeydown(e: KeyboardEvent) {
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === 'ArrowDown' || (mod && e.key.toLowerCase() === 'j')) {
      e.preventDefault();
      move(1);
    } else if (e.key === 'ArrowUp' || (mod && e.key.toLowerCase() === 'k')) {
      e.preventDefault();
      move(-1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const row = itemRows[ui.selected];
      if (row?.kind === 'item') choose(row.item);
    } else if (e.key === 'Escape' && !inline) {
      e.preventDefault();
      open = false;
    }
  }

  function onBackdrop(e: MouseEvent) {
    if (e.target === e.currentTarget) open = false;
  }

  // Focus the fresh input when the overlay opens (never in inline/embedded mode).
  function autofocus(node: HTMLInputElement, enabled: boolean) {
    if (enabled) requestAnimationFrame(() => node.focus());
  }
</script>

{#snippet panel(asDialog: boolean)}
  <div class="t-pal-panel">
    <div class="t-pal-in">
      <Icon name="search" />
      <input
        type="search"
        {placeholder}
        {...pwmIgnore}
        spellcheck="false"
        role="combobox"
        aria-label="Command palette search"
        aria-autocomplete="list"
        aria-controls={listId}
        aria-expanded={hasResults}
        aria-activedescendant={hasResults ? optionId(ui.selected) : undefined}
        bind:value={ui.query}
        oninput={() => (ui.selected = 0)}
        onkeydown={onKeydown}
        use:autofocus={asDialog}
      />
      <span class="t-kbd">esc</span>
    </div>

    <div class="t-pal-ls" id={listId} role="listbox" aria-label="Results" bind:this={listEl}>
      {#if hasResults}
        {#each rows as row, i (i)}
          {#if row.kind === 'group'}
            <div class="t-pal-g">{row.label}</div>
          {:else}
            {@const isSel = ui.selected === row.index}
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <div
              class="t-pal-it"
              class:is-sel={isSel}
              id={optionId(row.index)}
              role="option"
              aria-selected={isSel}
              tabindex="-1"
              onclick={() => choose(row.item)}
              onmousemove={() => (ui.selected = row.index)}
            >
              <Icon name={row.item.icon as IconName} />
              <span class="lbl"
                >{#if row.highlight}{@const hl = row.highlight}{row.item.label.slice(0, hl[0])}<b
                    >{row.item.label.slice(hl[0], hl[1])}</b
                  >{row.item.label.slice(hl[1])}{:else}{row.item.label}{/if}</span
              >
              {#if row.item.meta}<span class="meta">{row.item.meta}</span>{/if}
            </div>
          {/if}
        {/each}
      {:else}
        <div class="t-pal-empty">
          {#if trimmed}No matches for “{trimmed}” — try a job id, command, or file name.{:else}Nothing
            to jump to yet.{/if}
        </div>
      {/if}
    </div>

    <div class="t-pal-ft">
      <span><span class="t-kbd">↑↓</span> navigate</span>
      <span><span class="t-kbd">⏎</span> open</span>
      <span><span class="t-kbd">esc</span> close</span>
    </div>
  </div>
{/snippet}

{#if visible}
  {#if inline}
    <div class="pal-inline">{@render panel(false)}</div>
  {:else}
    <!-- backdrop click closes (mouse convenience); Escape on the input is the keyboard path -->
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div
      class="t-pal is-open"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      tabindex="-1"
      onclick={onBackdrop}
      use:trapFocus
    >
      {@render panel(true)}
    </div>
  {/if}
{/if}

<style>
  @keyframes toastin {
    from {
      translate: 0 8px;
      opacity: 0;
    }
  }

  /* ── command palette ── */
  .t-pal {
    position: fixed;
    inset: 0;
    z-index: 280;
    background: color-mix(in oklab, var(--bg0) 58%, transparent);
    backdrop-filter: blur(3px);
    display: none;
    align-items: flex-start;
    justify-content: center;
    padding: 11vh 16px 16px;
  }
  .t-pal.is-open {
    display: flex;
  }
  .t-pal-panel {
    width: min(580px, 100%);
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
  .t-pal-in {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 11px 13px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
  }
  .t-pal-in :global(.ic) {
    color: var(--tx3);
    width: 14px;
    height: 14px;
  }
  .t-pal-in input {
    flex: 1;
    min-width: 0;
    background: none;
    border: 0;
    outline: none;
    color: var(--tx0);
    font: 400 var(--fs-lg) var(--font-ui);
  }
  .t-pal-in input::placeholder {
    color: var(--tx3);
  }
  /* type=search opts out of Chromium's password manager but pulls in the UA
     clear button; drop it so the field looks the same as before. */
  .t-pal-in input::-webkit-search-cancel-button,
  .t-pal-in input::-webkit-search-decoration {
    -webkit-appearance: none;
    appearance: none;
  }
  .t-pal-ls {
    overflow-y: auto;
    padding: 4px;
    flex: 1;
  }
  .t-pal-g {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
    padding: 9px 10px 4px;
  }
  .t-pal-it {
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
  .t-pal-it :global(.ic) {
    width: 13px;
    height: 13px;
    color: var(--tx3);
    flex: none;
  }
  .t-pal-it .lbl {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .t-pal-it .lbl b {
    color: var(--acc);
    font-weight: 600;
  }
  .t-pal-it.is-sel {
    background: color-mix(in oklab, var(--acc) 14%, transparent);
    color: var(--tx0);
  }
  .t-pal-it.is-sel :global(.ic) {
    color: var(--acc);
  }
  .t-pal-it .meta {
    margin-left: auto;
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    white-space: nowrap;
    flex: none;
  }
  .t-pal-empty {
    padding: 20px;
    text-align: center;
    color: var(--tx3);
    font-size: var(--fs-sm);
  }
  .t-pal-ft {
    display: flex;
    gap: 14px;
    align-items: center;
    padding: 7px 13px;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    flex: none;
  }

  /* Under 640px the overlay becomes a full-screen sheet. Scoped to .t-pal so
     the inline/embedded variant keeps its box. */
  @media (max-width: 640px) {
    .t-pal {
      padding: 0;
    }
    .t-pal .t-pal-panel {
      width: 100%;
      height: 100%;
      max-height: none;
      border-radius: 0;
      border: 0;
    }
  }

  /* ── inline/embedded variant (gallery, docs) ── */
  .pal-inline {
    width: min(560px, 100%);
  }
  .pal-inline .t-pal-panel {
    max-height: 340px;
  }
</style>
