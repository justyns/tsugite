<script lang="ts">
  // Spaces switcher, mounted in the top bar. Presentational: the caller owns
  // every mutation.
  //
  // Toggle buttons in a labelled group, not a tablist: rename swaps a chip for
  // an <input> and close is a real button, both nested-interactive violations
  // inside a role="tab".
  import ContextMenu, { type ContextMenuItem } from '$lib/components/overlays/ContextMenu.svelte';
  import { TESTID } from '$lib/testids';

  interface SpaceItem {
    id: string;
    name: string;
  }

  let {
    spaces,
    activeId,
    onSelect,
    onAdd,
    onRename,
    onClose,
  }: {
    spaces: SpaceItem[];
    activeId: string;
    onSelect: (id: string) => void;
    onAdd: () => void;
    onRename: (id: string, name: string) => void;
    /** The store refuses to drop below one space, so withhold rather than no-op. */
    onClose: (id: string) => void;
  } = $props();

  const closable = $derived(spaces.length > 1);

  let editingId = $state<string | null>(null);
  let draft = $state('');
  let menu = $state<{ x: number; y: number; id: string } | null>(null);

  function startRename(space: SpaceItem) {
    editingId = space.id;
    draft = space.name;
  }

  function commitRename() {
    const id = editingId;
    if (!id) return;
    editingId = null;
    const name = draft.trim();
    if (name) onRename(id, name);
  }

  function onEditKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      event.preventDefault();
      commitRename();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      editingId = null;
    }
  }

  // Delete on a focused chip closes it, mirroring the pane tabs' keyboard path.
  function onChipKeydown(event: KeyboardEvent, id: string) {
    if ((event.key === 'Delete' || event.key === 'Backspace') && closable) {
      event.preventDefault();
      onClose(id);
    }
  }

  const menuItems = $derived.by<ContextMenuItem[]>(() => {
    const id = menu?.id;
    if (!id) return [];
    const space = spaces.find((s) => s.id === id);
    if (!space) return [];
    return [
      { label: 'Rename', run: () => startRename(space) },
      { label: 'Close', disabled: !closable, danger: true, run: () => onClose(id) },
    ];
  });
</script>

<div class="spacebar" role="group" aria-label="Spaces" data-testid={TESTID.spaceBar}>
  {#each spaces as space (space.id)}
    <span class="sp" class:is-active={space.id === activeId}>
      {#if editingId === space.id}
        <!-- svelte-ignore a11y_autofocus -->
        <input
          class="sp-edit"
          value={draft}
          oninput={(event) => (draft = event.currentTarget.value)}
          onkeydown={onEditKeydown}
          onblur={commitRename}
          aria-label="Rename space"
          autofocus
        />
      {:else}
        <button
          type="button"
          class="sp-name"
          aria-pressed={space.id === activeId}
          onclick={() => onSelect(space.id)}
          ondblclick={() => startRename(space)}
          oncontextmenu={(event) => {
            event.preventDefault();
            menu = { x: event.clientX, y: event.clientY, id: space.id };
          }}
          onkeydown={(event) => onChipKeydown(event, space.id)}
        >
          {space.name}
        </button>
        {#if closable}
          <button
            type="button"
            class="sp-x"
            aria-label="Close {space.name}"
            onclick={() => onClose(space.id)}
          >
            <svg class="ic" viewBox="0 0 16 16" aria-hidden="true">
              <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />
            </svg>
          </button>
        {/if}
      {/if}
    </span>
  {/each}
  <button type="button" class="sp-add" aria-label="New space" title="New space" onclick={onAdd}>
    <svg class="ic" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 3.5v9M3.5 8h9" />
    </svg>
  </button>
</div>

{#if menu}
  <ContextMenu
    x={menu.x}
    y={menu.y}
    label="Space actions"
    items={menuItems}
    onclose={() => (menu = null)}
  />
{/if}

<style>
  .spacebar {
    display: flex;
    align-items: center;
    gap: 2px;
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: none;
    padding-left: var(--sp-2);
  }
  .spacebar::-webkit-scrollbar {
    display: none;
  }
  .sp {
    display: inline-flex;
    align-items: center;
    flex: none;
    border: 1px solid transparent;
    border-radius: var(--r-md);
    background: var(--bg2);
  }
  .sp:hover {
    background: var(--bg3);
  }
  .sp.is-active {
    background: var(--bg4);
    border-color: var(--bd1);
  }
  .sp-name {
    max-width: 15ch;
    padding: 3px 8px;
    border: 0;
    background: none;
    border-radius: var(--r-md);
    color: var(--tx2);
    font: 500 var(--fs-xs) var(--font-ui);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: pointer;
  }
  /* Weight as well as surface: state never rides on colour alone. */
  .sp.is-active .sp-name {
    color: var(--tx0);
    font-weight: 600;
  }
  .sp-x,
  .sp-add {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    flex: none;
    padding: 0;
    border: 0;
    border-radius: var(--r-sm);
    background: none;
    color: var(--tx3);
    cursor: pointer;
  }
  .sp-x {
    margin-right: 3px;
    opacity: 0;
  }
  .sp:hover .sp-x,
  .sp:focus-within .sp-x {
    opacity: 1;
  }
  .sp-x:hover,
  .sp-add:hover {
    background: var(--bg3);
    color: var(--tx0);
  }
  .sp-name:focus-visible,
  .sp-x:focus-visible,
  .sp-add:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: -1px;
  }
  .sp-add {
    margin-left: 2px;
  }
  .ic {
    width: 11px;
    height: 11px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.5;
    stroke-linecap: round;
  }
  .sp-edit {
    width: 12ch;
    padding: 2px 7px;
    border: 1px solid var(--acc);
    border-radius: var(--r-md);
    background: var(--bg1);
    color: var(--tx0);
    font: 500 var(--fs-xs) var(--font-ui);
  }
  .sp-edit:focus {
    outline: none;
  }
  @media (max-width: 640px) {
    .sp-name {
      max-width: 9ch;
    }
  }
</style>
