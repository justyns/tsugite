<script lang="ts">
  // Recursive workspace file rail node. Directories toggle open/closed; files
  // open the doc and are drag sources carrying a {kind:'file'} surface ref so
  // they can be docked elsewhere.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { writeSurfaceDrag } from '$lib/shell/mux/drag';
  import { TESTID } from '$lib/testids';
  import type { TreeNode } from './wiki';
  import Self from './TreeNode.svelte';

  let {
    nodes,
    activePath,
    expanded,
    onToggle,
    onOpenFile,
    onPinFile,
    onFileContextMenu,
  }: {
    nodes: TreeNode[];
    activePath: string;
    expanded: Set<string>;
    onToggle: (path: string) => void;
    onOpenFile: (path: string) => void;
    /** Double-click-to-keep: pins the file's preview into a permanent tab. */
    onPinFile: (path: string) => void;
    onFileContextMenu?: (event: MouseEvent, path: string) => void;
  } = $props();

  function onDragStart(e: DragEvent, node: TreeNode) {
    if (!e.dataTransfer) return;
    writeSurfaceDrag(e.dataTransfer, {
      kind: 'file',
      params: { path: node.path },
      title: node.name,
    });
  }
</script>

{#each nodes as node (node.path)}
  {#if node.isDir}
    {@const open = expanded.has(node.path)}
    <div class="wk-dir" class:is-open={open}>
      <button
        type="button"
        data-act="wk-dir"
        aria-expanded={open}
        onclick={() => onToggle(node.path)}
      >
        <span class="chev"><Icon name="chev-r" size={9} /></span>
        <Icon name="files" />{node.name}
      </button>
      {#if open}
        <div class="kids">
          <Self
            nodes={node.children}
            {activePath}
            {expanded}
            {onToggle}
            {onOpenFile}
            {onPinFile}
            {onFileContextMenu}
          />
        </div>
      {/if}
    </div>
  {:else}
    <button
      type="button"
      class="wk-file"
      class:is-active={node.path === activePath}
      data-testid={TESTID.fileNode(node.path)}
      aria-current={node.path === activePath ? 'true' : undefined}
      draggable="true"
      ondragstart={(e) => onDragStart(e, node)}
      onclick={() => onOpenFile(node.path)}
      ondblclick={() => onPinFile(node.path)}
      oncontextmenu={(e) => onFileContextMenu?.(e, node.path)}
    >
      <Icon name="file" />{node.name}
    </button>
  {/if}
{/each}

<style>
  .wk-dir > button,
  .wk-file {
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
  .wk-dir > button:hover,
  .wk-file:hover {
    background: var(--bg2);
    color: var(--tx0);
  }
  .wk-file.is-active {
    background: color-mix(in oklab, var(--acc) 13%, transparent);
    color: var(--tx0);
  }
  .wk-file:focus-visible,
  .wk-dir > button:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: -2px;
  }
  .chev {
    display: inline-flex;
    transition: rotate var(--t-2) var(--ease);
    color: var(--tx3);
  }
  .wk-dir.is-open > button .chev {
    rotate: 90deg;
  }
  .kids {
    margin-left: 14px;
    border-left: 1px solid var(--bd0);
    padding-left: 4px;
  }
  .wk-file :global(.ic),
  .wk-dir > button :global(.ic) {
    width: 12px;
    height: 12px;
    color: var(--tx3);
    flex: none;
  }
  @media (prefers-reduced-motion: reduce) {
    .chev {
      transition: none;
    }
  }
</style>
