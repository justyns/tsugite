<script lang="ts">
  // Session lifecycle menu. A ghost dots button opens a role=menu popover of the
  // actions the gap audit enumerated: rename, edit topic, pin/unpin, set primary,
  // mark complete, cancel, restart. Which items show is gated by the session's
  // state (restart only from failed/cancelled; primary hidden when already
  // primary; cancel only while a run is in flight; complete only for live
  // sessions).
  import { tick } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import type { IconName } from '$lib/components/icon/icons';
  import { TESTID } from '$lib/testids';

  interface Action {
    id: string;
    label: string;
    icon: IconName;
    danger?: boolean;
    run: () => void;
  }

  let {
    pinned,
    isPrimary,
    canRestart,
    canComplete,
    canCancel,
    onRename,
    onEditTopic,
    onPin,
    onUnpin,
    onSetPrimary,
    onCopyId,
    onComplete,
    onCancel,
    onRestart,
  }: {
    pinned: boolean;
    isPrimary: boolean;
    canRestart: boolean;
    /** The session is live (not already completed/failed/cancelled). */
    canComplete: boolean;
    /** A run is in flight (busy or streaming) that a cancel would stop. */
    canCancel: boolean;
    onRename: () => void;
    onEditTopic: () => void;
    onPin: () => void;
    onUnpin: () => void;
    onSetPrimary: () => void;
    /** Copy the session id to the clipboard. */
    onCopyId: () => void;
    onComplete: () => void;
    onCancel: () => void;
    onRestart: () => void;
  } = $props();

  let open = $state(false);
  let root: HTMLElement | undefined;
  let menuEl = $state<HTMLElement>();

  const actions = $derived<Action[]>([
    { id: 'rename', label: 'Rename', icon: 'edit', run: onRename },
    { id: 'topic', label: 'Edit topic', icon: 'link', run: onEditTopic },
    { id: 'copyid', label: 'Copy session id', icon: 'copy', run: onCopyId },
    pinned
      ? { id: 'unpin', label: 'Unpin', icon: 'pin', run: onUnpin }
      : { id: 'pin', label: 'Pin', icon: 'pin', run: onPin },
    ...(isPrimary
      ? []
      : [
          {
            id: 'primary',
            label: 'Set as primary',
            icon: 'sparkle' as IconName,
            run: onSetPrimary,
          },
        ]),
    ...(canComplete
      ? [{ id: 'complete', label: 'Mark complete', icon: 'check' as IconName, run: onComplete }]
      : []),
    ...(canCancel
      ? [
          {
            id: 'cancel',
            label: 'Cancel run',
            icon: 'cancel' as IconName,
            danger: true,
            run: onCancel,
          },
        ]
      : []),
    ...(canRestart
      ? [{ id: 'restart', label: 'Restart', icon: 'retry' as IconName, run: onRestart }]
      : []),
  ]);

  async function toggle() {
    open = !open;
    if (open) {
      await tick();
      menuEl?.querySelector<HTMLElement>('[role="menuitem"]')?.focus();
    }
  }

  function choose(action: Action) {
    open = false;
    action.run();
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && open) {
      open = false;
      root?.querySelector<HTMLElement>('button')?.focus();
      return;
    }
    if (!open || (e.key !== 'ArrowDown' && e.key !== 'ArrowUp')) return;
    e.preventDefault();
    const items = Array.from(menuEl?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? []);
    const i = items.indexOf(document.activeElement as HTMLElement);
    const next = e.key === 'ArrowDown' ? i + 1 : i - 1;
    items[(next + items.length) % items.length]?.focus();
  }

  $effect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root && !root.contains(e.target as Node)) open = false;
    };
    window.addEventListener('mousedown', onDown);
    return () => window.removeEventListener('mousedown', onDown);
  });
</script>

<!-- Keyboard nav is delegated from the wrapper to the real button + menuitems it
     contains; the wrapper itself is not an interactive control. -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="menu-anchor" bind:this={root} onkeydown={onKeydown}>
  <Button
    variant="ghost"
    size="sm"
    iconOnly
    aria-label="Session menu"
    aria-haspopup="menu"
    aria-expanded={open}
    data-testid={TESTID.chatSessionMenuTrigger}
    onclick={toggle}
  >
    {#snippet icon()}<Icon name="dots" />{/snippet}
  </Button>
  {#if open}
    <div class="menu" role="menu" bind:this={menuEl} data-testid={TESTID.chatSessionMenu}>
      {#each actions as action (action.id)}
        <button
          type="button"
          role="menuitem"
          class="menu-item"
          class:is-danger={action.danger}
          onclick={() => choose(action)}
        >
          <Icon name={action.icon} size={12} />
          {action.label}
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .menu-anchor {
    position: relative;
    display: inline-flex;
  }
  .menu {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    z-index: 60;
    min-width: 168px;
    display: flex;
    flex-direction: column;
    padding: 4px;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
  }
  .menu-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 9px;
    border: 0;
    border-radius: var(--r-sm);
    background: transparent;
    color: var(--tx1);
    font: 500 var(--fs-sm) var(--font-ui);
    text-align: left;
    cursor: pointer;
  }
  .menu-item:hover,
  .menu-item:focus-visible {
    background: var(--bg4);
    color: var(--tx0);
    outline: none;
  }
  .menu-item.is-danger {
    color: var(--st-err);
  }
  .menu-item.is-danger:hover,
  .menu-item.is-danger:focus-visible {
    background: color-mix(in oklab, var(--st-err) 14%, transparent);
  }
</style>
