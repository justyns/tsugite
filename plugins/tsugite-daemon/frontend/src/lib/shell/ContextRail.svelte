<script lang="ts">
  // The shared context rail (.work-rail): the second sidebar of the workspace. Its
  // header carries the active workspace view's title + the collapse control; its
  // body swaps between the sessions / terminals / files rails. Collapsing hides the
  // whole rail (the work-shell drops to a single column and shows the expand strip
  // in the work-main), so the expand affordance lives in App, not here.
  import Icon from '$lib/components/icon/Icon.svelte';
  import type { WorkspaceView } from '$lib/stores/shellView.svelte';
  import ChatsRail from '../../views/chats/ChatsRail.svelte';
  import TerminalsRail from '../../views/terminals/TerminalsRail.svelte';
  import FilesRail from '../../views/files/FilesRail.svelte';

  let {
    view,
    onCollapse,
    focusedSessionId,
    focusedTerminalId,
    focusedFilePath,
    onOpenChat,
    onOpenTerminal,
    onOpenFile,
    onPinFile,
  }: {
    view: WorkspaceView;
    onCollapse: () => void;
    focusedSessionId: string | null;
    focusedTerminalId: string | null;
    focusedFilePath: string | null;
    onOpenChat: (sessionId: string, agent?: string) => void;
    onOpenTerminal: (terminalId: string) => void;
    onOpenFile: (agent: string, path: string) => void;
    /** Pin the focused pane's file preview (a file's double-click-to-keep). */
    onPinFile: (agent: string, path: string) => void;
  } = $props();

  const TITLES: Record<WorkspaceView, string> = {
    chats: 'Chats',
    terminals: 'Terminals',
    files: 'Files',
  };
</script>

<aside class="work-rail" aria-label="Sidebar">
  <div class="rail-top">
    <strong class="rail-title">{TITLES[view]}</strong>
    <div class="grow"></div>
    <button
      type="button"
      class="railc"
      data-act="rail-collapse"
      aria-label="Collapse sidebar"
      onclick={onCollapse}
    >
      <Icon name="chev-r" />
    </button>
  </div>

  <div class="rail-body">
    {#if view === 'chats'}
      <ChatsRail {focusedSessionId} {onOpenChat} />
    {:else if view === 'terminals'}
      <TerminalsRail {focusedTerminalId} {onOpenTerminal} />
    {:else}
      <FilesRail {focusedFilePath} {onOpenFile} {onPinFile} />
    {/if}
  </div>
</aside>

<style>
  /* .work-rail - palette: bg1 panel, bd0 seam. Fills its grid column; the body
     scrolls, the header stays. */
  .work-rail {
    position: relative;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--bd0);
    background: var(--bg1);
    min-width: 0;
    min-height: 0;
    overflow: hidden;
  }
  .rail-top {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px 8px;
    flex: none;
  }
  .rail-title {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .railc {
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
  .railc:hover {
    background: var(--bg3);
    color: var(--tx0);
  }
  .railc:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .railc :global(.ic) {
    width: 13px;
    height: 13px;
    rotate: 180deg;
  }
  .rail-body {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  /* Phone is a full-screen drilldown, not a collapsible drawer, so the collapse
     control has nothing to collapse to - and tapping it would corrupt the persisted
     desktop rail state. */
  @media (max-width: 640px) {
    .railc {
      display: none;
    }
  }
</style>
