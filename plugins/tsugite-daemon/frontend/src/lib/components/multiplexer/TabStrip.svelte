<script module lang="ts">
  import type { DotColor } from '$lib/components/buttons/dot-colors';

  // A docked session/terminal, shown as one tab. `state` drives the status dot
  // (color + accessible label); `closable` defaults to true.
  export type TabState = 'busy' | 'idle' | 'streaming' | 'blocked' | 'error' | 'done';

  export interface PaneTab {
    id: string;
    label: string;
    state?: TabState;
    closable?: boolean;
    /** VSCode-style preview tab: rendered italic; a double-click pins it (onPin). */
    ephemeral?: boolean;
  }

  // Dot color per state, as a shared Dot color name (never hardcoded hex).
  const DOT_COLOR: Record<TabState, DotColor> = {
    busy: 'ok',
    idle: 'mute',
    streaming: 'info',
    blocked: 'warn',
    error: 'err',
    done: 'mute',
  };

  // State spoken to assistive tech, so the dot is never color-alone.
  const STATE_LABEL: Record<TabState, string> = {
    busy: 'busy',
    idle: 'idle',
    streaming: 'streaming',
    blocked: 'needs attention',
    error: 'error',
    done: 'done',
  };
</script>

<script lang="ts">
  import Dot from '$lib/components/buttons/Dot.svelte';
  import ContextMenu, { type ContextMenuItem } from '$lib/components/overlays/ContextMenu.svelte';
  import { nextRovingIndex } from '$lib/actions/rovingNav';

  let {
    tabs,
    activeId,
    onSelect,
    onPin,
    onClose,
    onCloseOthers,
    onCloseAll,
    onNew,
    onSplit,
    onReorder,
    onDragTab,
    label = 'Pane sessions',
    newLabel = 'New tab',
    panelId,
  }: {
    tabs: PaneTab[];
    activeId?: string;
    onSelect?: (id: string) => void;
    /** Pin a preview tab (double-click); when omitted the double-click is inert. */
    onPin?: (id: string) => void;
    onClose?: (id: string) => void;
    /** Context-menu bulk actions; the menu only offers what's wired. */
    onCloseOthers?: (id: string) => void;
    onCloseAll?: () => void;
    onNew?: () => void;
    /** Split affordance at the strip's end (was the pane header's). */
    onSplit?: () => void;
    /** Omitting this leaves tabs undraggable. */
    onReorder?: (tabId: string, insertAt: number) => void;
    onDragTab?: (tabId: string, dataTransfer: DataTransfer) => void;
    label?: string;
    newLabel?: string;
    /** DOM id of the tabpanel these tabs control (WAI-ARIA tabs pattern).
     *  Also prefixes each tab's own id so the panel can point back via
     *  aria-labelledby (`{panelId}-tab-{tab.id}`). */
    panelId?: string;
  } = $props();

  // Roving tabindex: exactly one tab is in the tab order (the active one, or the
  // first when nothing is active), the rest are reached with the arrow keys.
  const rovingId = $derived(activeId ?? tabs[0]?.id);

  let listEl: HTMLDivElement;

  function tabEls(): HTMLElement[] {
    return listEl ? [...listEl.querySelectorAll<HTMLElement>('[data-tab-id]')] : [];
  }

  function onKeydown(event: KeyboardEvent, id: string) {
    // One DOM query per keypress; both the roving arithmetic and the extra keys
    // read the same snapshot.
    const els = tabEls();
    const index = els.indexOf(event.currentTarget as HTMLElement);
    if (index === -1) return;

    const next = nextRovingIndex(index, event.key, els.length);
    if (next !== null) {
      event.preventDefault();
      const el = els[next];
      el?.focus();
      if (el?.dataset.tabId) onSelect?.(el.dataset.tabId);
      return;
    }

    switch (event.key) {
      case 'Enter':
      case ' ':
        event.preventDefault();
        onSelect?.(id);
        break;
      case 'Delete':
      case 'Backspace': {
        const tab = tabs.find((t) => t.id === id);
        if (tab && tab.closable !== false && onClose) {
          event.preventDefault();
          onClose(id);
        }
        break;
      }
    }
  }

  // Right-click menu on a tab: close / close others / close all.
  let menu = $state<{ x: number; y: number; tabId: string } | null>(null);
  function openMenu(event: MouseEvent, tabId: string) {
    if (!onClose && !onCloseOthers && !onCloseAll) return;
    event.preventDefault();
    menu = { x: event.clientX, y: event.clientY, tabId };
  }
  const menuItems = $derived.by<ContextMenuItem[]>(() => {
    const target = menu?.tabId;
    if (!target) return [];
    return [
      ...(onClose ? [{ label: 'Close', run: () => onClose?.(target) }] : []),
      ...(onCloseOthers
        ? [
            {
              label: 'Close others',
              disabled: tabs.length < 2,
              run: () => onCloseOthers?.(target),
            },
          ]
        : []),
      ...(onCloseAll ? [{ label: 'Close all', run: () => onCloseAll?.() }] : []),
    ];
  });

  // `dragover` exposes the payload's types but not its contents.
  let dragging = $state<string | null>(null);
  let dropAt = $state<number | null>(null);

  function endDrag() {
    dragging = null;
    dropAt = null;
  }

  function onTabDragStart(event: DragEvent, tab: PaneTab) {
    dragging = tab.id;
    if (event.dataTransfer) onDragTab?.(tab.id, event.dataTransfer);
  }

  // A tab from another pane leaves `dragging` unset and falls through to the
  // pane's drop handler. Ours must stop propagation or the pane splits itself.
  function onTabDragOver(event: DragEvent, tab: PaneTab) {
    if (!dragging || !onReorder) return;
    event.preventDefault();
    event.stopPropagation();
    const r = (event.currentTarget as HTMLElement).getBoundingClientRect();
    dropAt =
      tabs.findIndex((t) => t.id === tab.id) + (event.clientX < r.left + r.width / 2 ? 0 : 1);
  }

  function onTabDrop(event: DragEvent) {
    if (!dragging || dropAt === null || !onReorder) return;
    event.preventDefault();
    event.stopPropagation();
    const from = tabs.findIndex((t) => t.id === dragging);
    const id = dragging;
    const insertAt = dropAt;
    endDrag();
    if (insertAt !== from && insertAt !== from + 1) onReorder(id, insertAt);
  }

  // Middle-click closes a tab (the mousedown preventDefault suppresses the
  // browser's autoscroll widget; auxclick carries the actual close).
  function onTabAuxclick(event: MouseEvent, tab: PaneTab) {
    if (event.button !== 1 || tab.closable === false || !onClose) return;
    event.preventDefault();
    onClose(tab.id);
  }

  function tabAria(tab: PaneTab): string {
    const base = tab.state ? `${tab.label}, ${STATE_LABEL[tab.state]}` : tab.label;
    // The close control is pointer-only (hidden from the a11y tree, see below);
    // Delete is the accessible path, so the tab announces it.
    return tab.closable !== false && onClose ? `${base} (Delete to close)` : base;
  }
</script>

<!-- The + new-tab button sits OUTSIDE the tablist element: role=tablist permits
     only tab children (axe aria-required-children). display:contents keeps the
     original single-row flex layout. -->
<div class="mux-tabs" bind:this={listEl}>
  <div class="mux-tablist" role="tablist" aria-orientation="horizontal" aria-label={label}>
    {#each tabs as tab, i (tab.id)}
      <div
        class="mux-tab"
        class:is-active={tab.id === activeId}
        class:is-preview={tab.ephemeral}
        class:is-dragging={dragging === tab.id}
        class:drop-before={dropAt === i}
        class:drop-after={dropAt === i + 1 && i === tabs.length - 1}
        role="tab"
        draggable={onReorder ? 'true' : undefined}
        ondragstart={(event) => onTabDragStart(event, tab)}
        ondragend={endDrag}
        ondragover={(event) => onTabDragOver(event, tab)}
        ondrop={onTabDrop}
        id={panelId ? `${panelId}-tab-${tab.id}` : undefined}
        aria-controls={panelId}
        data-tab-id={tab.id}
        aria-selected={tab.id === activeId}
        aria-label={tabAria(tab)}
        tabindex={tab.id === rovingId ? 0 : -1}
        onclick={() => onSelect?.(tab.id)}
        ondblclick={() => onPin?.(tab.id)}
        oncontextmenu={(event) => openMenu(event, tab.id)}
        onauxclick={(event) => onTabAuxclick(event, tab)}
        onmousedown={(event) => {
          if (event.button === 1) event.preventDefault();
        }}
        onkeydown={(event) => onKeydown(event, tab.id)}
      >
        {#if tab.state}
          <Dot color={DOT_COLOR[tab.state]} />
        {/if}
        <span class="lb">{tab.label}</span>
        {#if tab.closable !== false && onClose}
          <!-- Suppression is correct: pointer-only close, kept as an aria-hidden
               <span> because a real <button> nested in this role="tab" is an axe
               nested-interactive violation. The keyboard path is Delete on the
               focused tab (see onKeydown), announced by tabAria(). -->
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
          <span
            class="x"
            aria-hidden="true"
            onclick={(event) => {
              event.stopPropagation();
              onClose?.(tab.id);
            }}
          >
            <svg class="ic ic--close" viewBox="0 0 16 16" aria-hidden="true">
              <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />
            </svg>
          </span>
        {/if}
      </div>
    {/each}
  </div>
  {#if onNew}
    <button type="button" class="mux-tab mux-tab--new" aria-label={newLabel} onclick={onNew}>
      <svg class="ic ic--plus" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M8 3.5v9M3.5 8h9" />
      </svg>
    </button>
  {/if}
  {#if onSplit}
    <button type="button" class="mux-tab mux-tab--new" aria-label="Split pane" onclick={onSplit}>
      <svg class="ic ic--split" viewBox="0 0 16 16" aria-hidden="true">
        <rect x="2.5" y="3" width="4.6" height="10" />
        <rect x="9" y="3" width="4.6" height="10" />
      </svg>
    </button>
  {/if}
</div>

{#if menu}
  <ContextMenu
    x={menu.x}
    y={menu.y}
    label="Tab actions"
    items={menuItems}
    onclose={() => (menu = null)}
  />
{/if}

<style>
  /* Real flex wrapper, NOT display:contents - contents dissolves the element
     from the accessibility tree in practice, orphaning the role=tab children
     (axe aria-required-parent). Tabs scroll inside the tablist; the + button
     stays pinned after it. */
  .mux-tablist {
    display: flex;
    gap: 2px;
    min-width: 0;
    overflow-x: auto;
  }
  .mux-tabs {
    display: flex;
    gap: 2px;
    padding: 5px 6px 0;
    background: var(--bg1);
    border-bottom: 1px solid var(--bd0);
  }
  .mux-tab.is-dragging {
    opacity: 0.4;
  }
  .mux-tab.drop-before {
    box-shadow: inset 2px 0 0 0 var(--acc);
  }
  .mux-tab.drop-after {
    box-shadow: inset -2px 0 0 0 var(--acc);
  }
  .mux-tab {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 8px 6px;
    border: 1px solid transparent;
    border-bottom: 0;
    border-radius: var(--r-md) var(--r-md) 0 0;
    background: transparent;
    color: var(--tx2);
    font: 500 var(--fs-xs) / 1 var(--font-ui);
    cursor: pointer;
    white-space: nowrap;
    max-width: 190px;
  }
  .mux-tab .lb {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* Preview (ephemeral) tab: italic label, the VSCode signal that the next
     single-click replaces it in place until it is pinned. */
  .mux-tab.is-preview .lb {
    font-style: italic;
  }
  .mux-tab:hover {
    color: var(--tx0);
    background: var(--bg2);
  }
  /* Focusable div: suppress the UA ring a mouse click leaves behind; keyboard
     focus keeps a visible ring. */
  .mux-tab:focus {
    outline: none;
  }
  .mux-tab:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: -2px;
  }
  .mux-tab.is-active {
    background: var(--bg0);
    border-color: var(--bd0);
    color: var(--tx0);
    margin-bottom: -1px;
  }
  .mux-tab--new {
    padding: 5px 7px 6px;
  }
  .mux-tab :global(.t-dot) {
    width: 6px;
    height: 6px;
  }
  .mux-tab .x {
    display: inline-flex;
    color: var(--tx3);
    border-radius: 3px;
    padding: 1px;
    cursor: pointer;
  }
  .mux-tab .x:hover {
    color: var(--st-err);
    background: var(--bg3);
  }

  .ic {
    width: 13px;
    height: 13px;
    flex: none;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .ic--close {
    width: 9px;
    height: 9px;
  }
  .ic--split {
    width: 10px;
    height: 10px;
  }
  .ic--plus {
    width: 10px;
    height: 10px;
  }
</style>
