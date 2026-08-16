<script lang="ts">
  import Pane from '$lib/components/multiplexer/Pane.svelte';
  import TabStrip, { type PaneTab } from '$lib/components/multiplexer/TabStrip.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import type { LeafNode, PaneTabModel, SurfaceRef } from './layout';
  import {
    hasSurfaceDrag,
    readSurfaceDrag,
    readTabDrag,
    writeSurfaceDrag,
    writeTabDrag,
  } from './drag';
  import { type DropZone, dropIntent, dropZoneAt } from './dropzone';
  import type { MuxContent, MuxHandlers } from './types';
  import { TESTID } from '$lib/testids';

  let {
    pane,
    focused = false,
    showRing = false,
    narrow = false,
    content,
    ...handlers
  }: {
    pane: LeafNode;
    /** This pane is the focus target (where new surfaces dock). */
    focused?: boolean;
    /** Draw the focus ring (only meaningful when >1 pane is visible). */
    showRing?: boolean;
    /** Phone width: the split affordance is withheld (phones show one pane + a
     *  switcher, so splitting - which would hide a pane behind the switcher -
     *  is not offered). */
    narrow?: boolean;
    content?: MuxContent;
  } & MuxHandlers = $props();

  const activeTab = $derived(pane.tabs.find((t) => t.id === pane.activeTabId) ?? null);
  // Kind fallbacks are capitalized so an untitled surface labels like a titled
  // one ("Chat", never "chat" beside "Chat").
  const surfaceLabel = (t: PaneTabModel) =>
    t.title ?? t.kind.charAt(0).toUpperCase() + t.kind.slice(1);
  const stripTabs = $derived<PaneTab[]>(
    pane.tabs.map((t) => ({
      id: t.id,
      label: surfaceLabel(t),
      state: t.state,
      ephemeral: t.ephemeral,
    })),
  );
  const paneLabel = $derived(activeTab ? surfaceLabel(activeTab) : 'Empty pane');

  let slotEl = $state<HTMLElement>();
  let dropZone = $state<DropZone | null>(null);

  const DROP_LABEL: Record<DropZone, string> = {
    left: 'split left',
    right: 'split right',
    center: 'dock as tab',
  };

  function refOf(t: PaneTabModel): SurfaceRef {
    return {
      kind: t.kind,
      params: t.params,
      ...(t.title != null ? { title: t.title } : {}),
      ...(t.state != null ? { state: t.state } : {}),
    };
  }

  function ondragover(e: DragEvent) {
    if (!hasSurfaceDrag(e.dataTransfer)) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
    const r = slotEl?.getBoundingClientRect();
    if (!r) return;
    dropZone = dropZoneAt(r.width, e.clientX - r.left);
  }

  function ondragleave(e: DragEvent) {
    // Ignore leaves into descendant elements; only clear when the pane is exited.
    if (e.relatedTarget && slotEl?.contains(e.relatedTarget as Node)) return;
    dropZone = null;
  }

  function ondrop(e: DragEvent) {
    if (!hasSurfaceDrag(e.dataTransfer)) return;
    e.preventDefault();
    const ref = readSurfaceDrag(e.dataTransfer);
    const origin = readTabDrag(e.dataTransfer);
    const zone = dropZone;
    dropZone = null;
    if (!ref || !zone) return;
    const intent = dropIntent(zone);
    // Same-pane reorders never reach here; TabStrip stops them at the strip.
    // moveTab collapses the source pane if the move emptied it.
    if (intent.action === 'dock' && origin && origin.paneId !== pane.id) {
      handlers.onMoveTab?.(origin.paneId, origin.tabId, pane.id);
      return;
    }
    if (intent.action === 'dock') handlers.onDock?.(pane.id, ref);
    else handlers.onSplit?.(pane.id, intent.dir, ref, intent.position);
  }

  function onDragTab(tabId: string, dt: DataTransfer) {
    const tab = pane.tabs.find((t) => t.id === tabId);
    if (!tab) return;
    writeSurfaceDrag(dt, refOf(tab));
    writeTabDrag(dt, { paneId: pane.id, tabId });
  }

  function splitHere() {
    if (narrow) return;
    if (activeTab) handlers.onSplit?.(pane.id, 'row', refOf(activeTab), 'after');
  }
</script>

<!-- Progressive-enhancement wrapper: click-to-focus and drag-to-dock ride on
     this div, but the pane's accessible structure is the inner <section> landmark
     and its tab strip / controls, which keyboard users operate directly. -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  bind:this={slotEl}
  class="mux-slot"
  class:is-focused={showRing && focused}
  data-focused={focused}
  data-testid={TESTID.muxPane}
  tabindex="-1"
  onpointerdown={() => handlers.onFocusPane?.(pane.id)}
  onfocusin={() => handlers.onFocusPane?.(pane.id)}
  {ondragover}
  {ondragleave}
  {ondrop}
>
  <Pane
    label={paneLabel}
    panelId={pane.tabs.length > 0 ? `mux-panel-${pane.id}` : undefined}
    activeTabId={pane.activeTabId ?? undefined}
  >
    {#snippet tabs()}
      {#if pane.tabs.length > 0}
        <!-- The strip is the pane's only chrome: per-tab close, bulk actions on
             right-click, + and split at the end. Surfaces carry their own
             titles, so a second header bar would just repeat them. -->
        <TabStrip
          tabs={stripTabs}
          activeId={pane.activeTabId ?? undefined}
          panelId={`mux-panel-${pane.id}`}
          onSelect={(id) => handlers.onSelectTab?.(pane.id, id)}
          onPin={handlers.onPinTab ? (id) => handlers.onPinTab?.(pane.id, id) : undefined}
          onClose={(id) => handlers.onCloseTab?.(pane.id, id)}
          onCloseOthers={handlers.onCloseOtherTabs
            ? (id) => handlers.onCloseOtherTabs?.(pane.id, id)
            : undefined}
          onCloseAll={handlers.onCloseAllTabs
            ? () => handlers.onCloseAllTabs?.(pane.id)
            : undefined}
          onNew={handlers.onNewTab ? () => handlers.onNewTab?.(pane.id) : undefined}
          onSplit={narrow || !activeTab ? undefined : splitHere}
          onReorder={handlers.onMoveTab
            ? (id, at) => handlers.onMoveTab?.(pane.id, id, pane.id, at)
            : undefined}
          {onDragTab}
        />
      {/if}
    {/snippet}
    {#if activeTab}
      {#if content}
        {@render content(activeTab, () => handlers.onFocusPane?.(pane.id))}
      {/if}
    {:else}
      <div class="mux-empty">
        <p>Nothing open in this pane.</p>
        {#if handlers.onNewTab}
          <Button variant="pri" onclick={() => handlers.onNewTab?.(pane.id)}>
            {#snippet icon()}<Icon name="plus" />{/snippet}
            Open a surface
          </Button>
        {/if}
      </div>
    {/if}
  </Pane>

  {#if dropZone}
    <div class="mux-drop" data-zone={dropZone} aria-hidden="true">
      <span class="mux-drop-lb">{DROP_LABEL[dropZone]}</span>
    </div>
  {/if}
</div>

<style>
  .mux-slot {
    position: relative;
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    flex: 1;
  }
  /* The Pane fills the slot; `.mux-pane` carries no flex-grow of its own (it's
     the split container that stretches it), so the wrapper does it here -
     otherwise the pane collapses to content height and leaves dead space. */
  .mux-slot > :global(.mux-pane) {
    flex: 1;
    min-height: 0;
  }
  /* The slot is programmatically focused (tabindex=-1) only to recover keyboard
     focus after a pane/tab close; the focused pane is signalled by the accent ring
     below, so the container itself shows no separate UA outline. */
  .mux-slot:focus {
    outline: none;
  }
  /* Focus ring for the active pane - an inset accent outline, not a colour swap,
     so it reads independently of the surface's own state colours. */
  .mux-slot.is-focused::after {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 20;
    pointer-events: none;
    box-shadow: inset 0 0 0 2px var(--acc);
    border-radius: var(--r-sm);
  }

  .mux-empty {
    box-sizing: border-box;
    min-height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--sp-3);
    padding: var(--sp-5);
    text-align: center;
  }
  .mux-empty p {
    margin: 0;
    max-width: 34ch;
    font: 400 var(--fs-sm) / 1.6 var(--font-ui);
    color: var(--tx3);
  }

  /* Drop-zone affordance: the target region highlights (left/right half or the
     whole pane) with a dashed accent border and a text label of the action. */
  .mux-drop {
    position: absolute;
    z-index: 40;
    top: 0;
    bottom: 0;
    display: grid;
    place-items: center;
    pointer-events: none;
    border: 2px dashed var(--acc);
    border-radius: var(--r-md);
    background: color-mix(in oklab, var(--acc) 12%, transparent);
  }
  .mux-drop[data-zone='left'] {
    left: 0;
    width: 50%;
  }
  .mux-drop[data-zone='right'] {
    right: 0;
    width: 50%;
  }
  .mux-drop[data-zone='center'] {
    left: 0;
    right: 0;
  }
  .mux-drop-lb {
    background: var(--bg3);
    border: 1px solid var(--bd1);
    color: var(--tx0);
    font: 600 var(--fs-xs) var(--font-mono);
    padding: 4px 10px;
    border-radius: var(--r-full);
    box-shadow: var(--sh-2);
  }
</style>
