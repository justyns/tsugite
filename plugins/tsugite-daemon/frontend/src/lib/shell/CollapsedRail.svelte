<script lang="ts">
  // The collapsed context rail: a thin expand strip, plus the overlay a hover
  // peeks open over the work area. The overlay mounts its own ContextRail above
  // the strip, leaving the grid at one column and the persisted collapsed flag
  // untouched. Clicking the strip pins the rail open.
  import Icon from '$lib/components/icon/Icon.svelte';
  import ContextRail from './ContextRail.svelte';
  import type { WorkspaceView } from '$lib/stores/shellView.svelte';
  import { TESTID } from '$lib/testids';

  let {
    view,
    peeking = false,
    onHoverStart,
    onHoverEnd,
    onPin,
    focusedSessionId,
    focusedTerminalId,
    focusedFilePath,
    onOpenChat,
    onOpenTerminal,
    onOpenFile,
    onPinFile,
  }: {
    view: WorkspaceView;
    peeking?: boolean;
    onHoverStart: () => void;
    onHoverEnd: () => void;
    onPin: () => void;
    focusedSessionId: string | null;
    focusedTerminalId: string | null;
    focusedFilePath: string | null;
    onOpenChat: (sessionId: string) => void;
    onOpenTerminal: (terminalId: string) => void;
    onOpenFile: (path: string) => void;
    onPinFile: (path: string) => void;
  } = $props();

  let root: HTMLElement | undefined = $state();

  function onFocusOut(event: FocusEvent) {
    const next = event.relatedTarget;
    if (next instanceof Node && root?.contains(next)) return;
    onHoverEnd();
  }
</script>

<div
  class="peek-host"
  bind:this={root}
  onmouseenter={onHoverStart}
  onmouseleave={onHoverEnd}
  onfocusin={onHoverStart}
  onfocusout={onFocusOut}
  role="presentation"
>
  <button
    type="button"
    class="rail-expand"
    data-act="rail-collapse"
    data-testid={TESTID.railExpand}
    aria-label="Show sidebar"
    title="Show sidebar"
    onclick={onPin}
  >
    <Icon name="chev-r" />
  </button>

  {#if peeking}
    <div class="rail-peek" data-testid={TESTID.railPeek}>
      <ContextRail
        {view}
        peeking
        onCollapse={onPin}
        {focusedSessionId}
        {focusedTerminalId}
        {focusedFilePath}
        {onOpenChat}
        {onOpenTerminal}
        {onOpenFile}
        {onPinFile}
      />
    </div>
  {/if}
</div>

<style>
  /* The host is only a hover/focus boundary. The strip and the panel each position
     themselves against .work-shell. */
  .peek-host {
    display: contents;
  }
  .rail-expand {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 20px;
    z-index: 25;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg1);
    border: 0;
    border-right: 1px solid var(--bd1);
    color: var(--tx3);
    cursor: pointer;
    padding: 0;
  }
  .rail-expand:hover {
    color: var(--acc);
    background: var(--bg2);
  }
  .rail-expand :global(.ic) {
    width: 13px;
    height: 13px;
  }
  .rail-peek {
    position: absolute;
    inset: 0 auto 0 0;
    width: clamp(200px, var(--w-work), 30%);
    z-index: 26;
    display: flex;
    box-shadow: var(--sh-2);
    animation: peek-in var(--t-2) var(--ease);
  }
  .rail-peek > :global(.work-rail) {
    flex: 1;
    min-width: 0;
  }
  @keyframes peek-in {
    from {
      translate: -102% 0;
    }
    to {
      translate: 0 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .rail-peek {
      animation: none;
    }
  }
</style>
