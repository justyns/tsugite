<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import {
    buildSessionRowAriaLabel,
    sessionStateMeta,
    sourceTypeLabel,
    type SessionSourceType,
    type SessionState,
  } from './rowState';

  let {
    title,
    when,
    description,
    state,
    sourceType,
    isActive = false,
    isPinned = false,
    isUnread = false,
    activeJobCount = 0,
    waitingOnCount = 0,
    onSelect,
    onOpenNewTab,
  }: {
    title: string;
    when: string;
    description?: string;
    state: SessionState;
    sourceType: SessionSourceType;
    /** The row for the session currently open in the main pane (cool left edge). */
    isActive?: boolean;
    isPinned?: boolean;
    isUnread?: boolean;
    activeJobCount?: number;
    waitingOnCount?: number;
    onSelect?: () => void;
    onOpenNewTab?: () => void;
  } = $props();

  const meta = $derived(sessionStateMeta(state));
  // Only idle/done have no ambient glyph to protect, so unread can take the slot.
  const showUnreadDot = $derived(isUnread && (state === 'idle' || state === 'done'));
  const ariaLabel = $derived(buildSessionRowAriaLabel({ title, state, isUnread }));
  const hasMarkers = $derived(
    state === 'needs-you' || activeJobCount > 0 || waitingOnCount > 0 || isPinned,
  );
  const isEnded = $derived(state === 'done' || state === 'failed');

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect?.();
    }
  }
</script>

<div
  class="t-srow"
  class:is-active={isActive}
  class:is-attn={state === 'needs-you'}
  class:is-unread={isUnread}
  class:is-ended={isEnded}
  role="button"
  tabindex="0"
  aria-current={isActive ? 'true' : undefined}
  aria-label={ariaLabel}
  onclick={() => onSelect?.()}
  ondblclick={() => onOpenNewTab?.()}
  onkeydown={handleKeydown}
>
  <span class="ind">
    {#if showUnreadDot}
      <span class="t-dot t-dot--unread" aria-hidden="true"></span>
    {:else if meta.spin}
      <Spin color={meta.color} />
    {:else if meta.icon}
      <Icon name={meta.icon} color={meta.color} />
    {/if}
  </span>
  <span class="ttl">{title}</span><span class="when">{when}</span>
  <span class="sub">
    <span class="t-type" data-k={sourceType}>{sourceTypeLabel(sourceType)}</span>
    {#if description}<span class="desc">{description}</span>{/if}
    {#if hasMarkers}
      <span class="mk">
        {#if state === 'needs-you'}
          <Icon name="alert" color="var(--st-warn)" size={10} />
        {/if}
        {#if activeJobCount > 0}
          <span
            class="t-badge"
            aria-label="{activeJobCount} active job{activeJobCount === 1 ? '' : 's'}"
            >{activeJobCount}&#9656;</span
          >
        {/if}
        {#if waitingOnCount > 0}
          <span
            class="wait"
            aria-label="waiting on {waitingOnCount} session{waitingOnCount === 1 ? '' : 's'}"
          >
            <Icon name="clock" size={10} />{waitingOnCount}
          </span>
        {/if}
        {#if isPinned}
          <Icon name="pin" size={10} />
        {/if}
      </span>
    {/if}
  </span>
</div>

<style>
  .t-srow {
    display: grid;
    grid-template-columns: 14px 1fr auto;
    grid-template-rows: auto auto;
    gap: 1px 8px;
    padding: 6px 10px 7px;
    border-left: 2px solid transparent;
    cursor: pointer;
    min-width: 0;
    position: relative;
  }
  .t-srow:hover {
    background: var(--bg2);
  }
  .t-srow.is-active {
    background: var(--bg2);
    border-left-color: var(--acc);
  }
  .t-srow.is-attn {
    border-left-color: var(--st-warn);
  }
  .t-srow .ind {
    grid-row: 1 / 3;
    display: flex;
    align-items: flex-start;
    padding-top: 4px;
    justify-content: center;
    color: var(--tx3);
  }
  .t-srow .ttl {
    font: 500 var(--fs-md) / 1.3 var(--font-ui);
    color: var(--tx1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .t-srow.is-unread .ttl {
    font-weight: 600;
    color: var(--tx0);
  }
  /* Finished rows read muted so they never look live. Placed after is-unread so
     an ended-but-unread row still dims rather than bolding. */
  .t-srow.is-ended {
    opacity: 0.72;
  }
  .t-srow.is-ended .ttl {
    font-weight: 500;
    color: var(--tx2);
  }
  .t-srow.is-ended:hover,
  .t-srow.is-ended.is-active {
    opacity: 1;
  }
  .t-srow .when {
    font: 500 var(--fs-2xs) / 1.5 var(--font-mono);
    /* tx2, not tx3: faint text on a raised (bg2) row misses the
       4.5:1 contrast contract at this size. */
    color: var(--tx2);
    white-space: nowrap;
  }
  .t-srow .sub {
    grid-column: 2 / 4;
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }
  .t-srow .sub .desc {
    font-size: var(--fs-xs);
    color: var(--tx2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .t-srow .mk {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--tx3);
    flex: none;
  }

  /* t-type (session source tag) */
  .t-type {
    display: inline-block;
    font: 600 var(--fs-2xs) / 1.5 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0 4px;
    border-radius: 3px;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 15%, transparent);
    flex: none;
  }
  .t-type[data-k='code'] {
    --c: var(--acc);
  }
  .t-type[data-k='ops'] {
    --c: var(--st-warn);
  }
  .t-type[data-k='research'] {
    --c: var(--st-queue);
  }
  .t-type[data-k='chat'] {
    /* tx2, not st-mute: the badge renders --c as 10px text on a tinted chip,
       and st-mute lands under 3:1 there. */
    --c: var(--tx2);
  }

  /* t-badge (active-job count marker) - compacted to 14px for this row's tight
     marker slot. */
  .t-badge {
    display: inline-grid;
    place-items: center;
    min-width: 14px;
    height: 14px;
    padding: 0 5px;
    border-radius: var(--r-sm);
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    background: var(--bg3);
    border: 1px solid var(--bd1);
    color: var(--tx2);
  }

  .t-srow .wait {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    color: var(--tx2);
    flex: none;
  }

  /* t-dot (unread marker) */
  .t-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--c, var(--st-mute));
    flex: none;
    display: inline-block;
  }
  .t-dot--unread {
    --c: var(--acc);
    width: 7px;
    height: 7px;
  }
</style>
