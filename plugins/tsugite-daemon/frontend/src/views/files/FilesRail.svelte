<script lang="ts">
  // Files context rail: the workspace file tree + search + new-note. A file
  // click (or drag) opens it as a document surface in the focused pane via
  // `onOpenFile`. The tree + backlink index are shared with the open document
  // surfaces through the filesWorkspace store, so the expensive whole-workspace
  // walk happens once.
  import Icon from '$lib/components/icon/Icon.svelte';
  import SearchInput from '$lib/components/inputs/SearchInput.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Modal from '$lib/components/overlays/Modal.svelte';
  import ContextMenu, { type ContextMenuItem } from '$lib/components/overlays/ContextMenu.svelte';
  import { spaces } from '$lib/stores/spaces.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { writeSurfaceDrag } from '$lib/shell/mux/drag';
  import { files } from '$lib/stores/files.svelte';
  import { TESTID } from '$lib/testids';
  import TreeNode from './TreeNode.svelte';
  import { isMarkdown } from './load';
  import { filesWorkspace } from './workspace.svelte';

  let {
    focusedFilePath,
    onOpenFile,
    onPinFile,
  }: {
    focusedFilePath: string | null;
    onOpenFile: (path: string) => void;
    /** Double-click-to-keep: pins the file's preview into a permanent tab. */
    onPinFile: (path: string) => void;
  } = $props();

  let query = $state('');
  let expanded = $state<Set<string>>(new Set());

  const ws = $derived(filesWorkspace.ws);
  const idx = $derived(ws?.index ?? null);
  const wsName = $derived(ws?.workspaceDir.split('/').filter(Boolean).pop() ?? 'workspace');

  // Load the shared workspace, then expand every directory so the tree reads open.
  $effect(() => {
    void filesWorkspace.ensure().then(() => {
      if (filesWorkspace.ws)
        expanded = new Set(filesWorkspace.ws.entries.filter((e) => e.is_dir).map((e) => e.path));
    });
  });

  // Rail filter: empty -> tree; `#tag` -> files carrying that tag; else path substring.
  const tagQuery = $derived(query.trim().startsWith('#'));
  const filtered = $derived.by(() => {
    const q = query.trim();
    if (!q || !ws) return null;
    const fileEntries = ws.entries.filter((e) => !e.is_dir);
    if (q.startsWith('#')) {
      const tag = q.slice(1).toLowerCase();
      return fileEntries.filter((e) => (idx?.tagsByFile.get(e.path) ?? []).includes(tag));
    }
    const needle = q.toLowerCase();
    return fileEntries.filter((e) => e.path.toLowerCase().includes(needle));
  });

  // Tag search reads the content-derived index, which is lazy (it means reading
  // every note once) - a #query is the explicit ask that kicks the scan off.
  $effect(() => {
    if (tagQuery && filesWorkspace.indexState === 'none') void filesWorkspace.ensureIndex();
  });
  const scanning = $derived(tagQuery && filesWorkspace.indexState !== 'ready');

  function toggleDir(path: string) {
    const next = new Set(expanded);
    if (next.has(path)) next.delete(path);
    else next.add(path);
    expanded = next;
  }

  function open(path: string) {
    onOpenFile(path);
  }

  function pin(path: string) {
    onPinFile(path);
  }

  function fileDragStart(e: DragEvent, path: string, name: string) {
    if (!e.dataTransfer) return;
    writeSurfaceDrag(e.dataTransfer, { kind: 'file', params: { path }, title: name });
  }

  // Right-click menu on a file row (flat search results and tree nodes alike).
  let menu = $state<{ x: number; y: number; path: string } | null>(null);
  function openFileMenu(event: MouseEvent, path: string) {
    event.preventDefault();
    menu = { x: event.clientX, y: event.clientY, path };
  }
  const menuItems = $derived.by<ContextMenuItem[]>(() => {
    const path = menu?.path;
    if (!path) return [];
    return [
      {
        label: 'Open in new tab',
        run: () =>
          spaces.open({
            kind: 'file',
            params: { path },
            title: path.split('/').pop() ?? path,
          }),
      },
      {
        label: 'Copy path',
        run: () =>
          void navigator.clipboard
            ?.writeText(path)
            .then(() => toasts.push('ok', 'Path copied', { body: path }))
            .catch(() => toasts.push('err', 'Could not copy path')),
      },
    ];
  });

  // New-note flow lives in a Modal (never window.prompt) so the resolved path
  // shows before creating; names resolve against the workspace root here.
  let newNoteOpen = $state(false);
  let newNoteName = $state('untitled.md');
  const newNotePath = $derived.by(() => {
    const name = newNoteName.trim();
    if (!name) return '';
    return isMarkdown(name) ? name : `${name}.md`;
  });

  function newNote() {
    newNoteName = 'untitled.md';
    newNoteOpen = true;
  }

  async function createNote(event?: Event) {
    event?.preventDefault();
    const path = newNotePath;
    if (!path) return;
    const title = path
      .split('/')
      .pop()!
      .replace(/\.[^.]+$/, '');
    try {
      await files.write(path, `# ${title}\n\n`);
      newNoteOpen = false;
      await filesWorkspace.reload();
      open(path);
      toasts.push('ok', 'Note created', { body: path });
    } catch (err) {
      toasts.push('err', 'Could not create note', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }
</script>

<div class="wk-tree" data-testid={TESTID.filesTree}>
  <div class="sidebar-hd">
    <strong>Workspace</strong>
    <span class="cnt">{wsName}/</span>
    <div class="grow"></div>
    <button type="button" class="hd-btn" aria-label="New note" onclick={newNote}>
      <Icon name="plus" />
    </button>
  </div>
  <div class="sidebar-se" data-testid={TESTID.filesSearch}>
    <SearchInput
      bind:value={query}
      placeholder="search workspace…"
      ariaLabel="Search workspace"
      shortcutKey="/"
    />
  </div>
  <div class="wk-tree-ls">
    {#if filesWorkspace.loading && !ws}
      <PaneState kind="loading" lines={6} />
    {:else if filesWorkspace.error && !ws}
      <PaneState kind="error" title="Could not load workspace">
        {#snippet hint()}<span class="mono">{filesWorkspace.error}</span>{/snippet}
        {#snippet actions()}
          <Button size="sm" onclick={() => filesWorkspace.reload()}>Retry</Button>
        {/snippet}
      </PaneState>
    {:else if ws}
      {#if filtered}
        {#if filtered.length === 0}
          <p class="rail-empty">{scanning ? 'scanning workspace tags…' : 'No files match.'}</p>
        {:else}
          {#each filtered as e (e.path)}
            <button
              type="button"
              class="wk-file"
              class:is-active={e.path === focusedFilePath}
              data-testid={TESTID.fileNode(e.path)}
              draggable="true"
              ondragstart={(ev) => fileDragStart(ev, e.path, e.name)}
              onclick={() => open(e.path)}
              ondblclick={() => pin(e.path)}
              oncontextmenu={(ev) => openFileMenu(ev, e.path)}
            >
              <Icon name="file" />{e.path}
            </button>
          {/each}
        {/if}
      {:else}
        <TreeNode
          nodes={ws.tree}
          activePath={focusedFilePath ?? ''}
          {expanded}
          onToggle={toggleDir}
          onOpenFile={open}
          onPinFile={pin}
          onFileContextMenu={openFileMenu}
        />
      {/if}
    {/if}
  </div>
</div>

<Modal open={newNoteOpen} title="New note" onclose={() => (newNoteOpen = false)}>
  <form class="nn-form" onsubmit={createNote}>
    <Input bind:value={newNoteName} ariaLabel="Note name" placeholder="name, or nested/path.md" />
    <p class="nn-path">creates <span class="nn-mono">{newNotePath || '…'}</span></p>
  </form>
  {#snippet footer()}
    <Button onclick={() => (newNoteOpen = false)}>Cancel</Button>
    <Button variant="pri" onclick={createNote} disabled={!newNotePath}>Create</Button>
  {/snippet}
</Modal>

{#if menu}
  <ContextMenu
    x={menu.x}
    y={menu.y}
    label="File actions"
    items={menuItems}
    onclose={() => (menu = null)}
  />
{/if}

<style>
  .wk-tree {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    height: 100%;
    background: var(--bg1);
  }
  .sidebar-hd {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px 8px;
    flex: none;
  }
  .sidebar-hd strong {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
  }
  .sidebar-hd .cnt {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .sidebar-se {
    padding: 0 10px 8px;
    flex: none;
  }
  .wk-tree-ls {
    overflow-y: auto;
    flex: 1;
    padding: 4px 6px 10px;
    font: 400 var(--fs-sm) / 1 var(--font-mono);
    min-height: 0;
  }
  .rail-empty {
    margin: 8px;
    font: 400 var(--fs-xs) var(--font-mono);
    color: var(--tx3);
  }
  .wk-tree-ls .wk-file {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    text-align: left;
    background: none;
    border: 0;
    color: var(--tx2);
    font: inherit;
    padding: 4px 6px;
    border-radius: var(--r-sm);
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
  }
  .wk-tree-ls .wk-file:hover {
    background: var(--bg2);
    color: var(--tx0);
  }
  .wk-tree-ls .wk-file.is-active {
    background: color-mix(in oklab, var(--acc) 13%, transparent);
    color: var(--tx0);
  }
  .wk-tree-ls .wk-file:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: -2px;
  }
  .wk-tree-ls .wk-file :global(.ic) {
    width: 12px;
    height: 12px;
    color: var(--tx3);
    flex: none;
  }
  .hd-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border: 1px solid transparent;
    border-radius: var(--r-md);
    background: none;
    color: var(--tx2);
    cursor: pointer;
    flex: none;
  }
  .hd-btn:hover {
    background: var(--bg3);
    color: var(--tx0);
  }
  .hd-btn:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .hd-btn :global(.ic) {
    width: 13px;
    height: 13px;
  }
  .nn-form {
    display: grid;
    gap: var(--sp-2);
  }
  .nn-path {
    margin: 0;
    font: 400 var(--fs-xs) / 1.5 var(--font-ui);
    color: var(--tx2);
  }
  .nn-mono {
    font-family: var(--font-mono);
    color: var(--tx1);
  }
  .mono {
    font-family: var(--font-mono);
  }
</style>
