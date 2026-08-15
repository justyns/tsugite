<script lang="ts">
  // The chat surface's sessions rail: a filterable, grouped list of SessionRow
  // components (pinned / active / recent buckets), a token-grammar search that
  // also drives the server-side full-store merge, quick filter pills, a needs-you
  // count, and a new-session button. Each row is a mux drag source so it can be
  // dropped into another pane ({kind:'chat', params:{sessionId}}).
  import SearchInput from '$lib/components/inputs/SearchInput.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import SessionRow from '$lib/components/rows/SessionRow.svelte';
  import ContextMenu, { type ContextMenuItem } from '$lib/components/overlays/ContextMenu.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { spaces } from '$lib/stores/spaces.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { writeSurfaceDrag } from '$lib/shell/mux/drag';
  import { readLocal, writeLocal } from '$lib/storage';
  import type { SessionRow as Row } from '$lib/stores/sessions.svelte';
  import { TESTID } from '$lib/testids';
  import {
    parseSessionFilter,
    sessionMatchesFilter,
    filterFreeText,
    isActiveFilter,
    type SessionFilterRow,
  } from './sessionFilter';
  import {
    groupSessions,
    sessionSourceType,
    sessionRowState,
    isFinishedSession,
    formatWhen,
    sessionTopic,
  } from './sessionModel';
  import { attachRecordToChat, copyReference } from './attachRecord';
  import { chatRouteParams } from './chatLink';

  let {
    rows,
    agent,
    agents = [],
    selectedId,
    attn,
    loading = false,
    onSelect,
    onNew,
    onAgentChange,
    onServerSearch,
  }: {
    rows: Row[];
    agent: string;
    /** All configured agents; >1 renders the agent picker. */
    agents?: string[];
    selectedId: string | null;
    /** Session ids with an outstanding ask_user (drives needs-you). */
    attn: Set<string>;
    loading?: boolean;
    onSelect: (id: string) => void;
    onNew: () => void;
    onAgentChange?: (agent: string) => void;
    /** Free-text portion of the query, for the store's server-merge search. */
    onServerSearch: (q: string) => void;
  } = $props();

  let search = $state('');
  let quick = $state<'all' | 'needs-you' | 'pinned'>('all');
  let lastServerQuery = '';

  // Right-click menu on a row: tab-independent actions (open elsewhere, pin,
  // copy id, lifecycle). Rename/topic stay in the conversation header menu -
  // they need its inline edit field.
  let menu = $state<{ x: number; y: number; row: Row } | null>(null);
  function openRowMenu(event: MouseEvent, row: Row) {
    event.preventDefault();
    menu = { x: event.clientX, y: event.clientY, row };
  }
  function openInNewTab(row: Row) {
    spaces.open({
      kind: 'chat',
      params: chatRouteParams(row.id, agent),
      title: row.title ?? 'Chat',
    });
  }
  async function copyId(id: string) {
    try {
      await navigator.clipboard?.writeText(id);
      toasts.push('ok', 'Session id copied', { body: id });
    } catch {
      toasts.push('err', 'Could not copy session id');
    }
  }
  async function completeRow(id: string) {
    try {
      await sessions.complete(id);
      toasts.push('ok', 'Session marked complete');
      await sessions.load(agent);
    } catch (err) {
      toasts.push('err', 'Could not mark complete', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }
  const menuItems = $derived.by<ContextMenuItem[]>(() => {
    const row = menu?.row;
    if (!row) return [];
    return [
      {
        label: 'Open in new tab',
        run: () => openInNewTab(row),
      },
      row.pinned
        ? { label: 'Unpin', run: () => void sessions.unpin(row.id) }
        : { label: 'Pin', run: () => void sessions.pin(row.id) },
      { label: 'Copy session id', run: () => void copyId(row.id) },
      { label: 'Add to chat', run: () => void attachRecordToChat('session', row.id) },
      { label: 'Copy reference', run: () => void copyReference('session', row.id) },
      ...(!isFinishedSession(row)
        ? [{ label: 'Mark complete', run: () => void completeRow(row.id) }]
        : []),
    ];
  });

  function toFilterRow(r: Row): SessionFilterRow {
    return {
      id: r.id,
      title: r.title,
      topic: typeof r.metadata?.topic === 'string' ? r.metadata.topic : undefined,
      label: r.label,
      agent,
      status: r.status,
      pinned: r.pinned,
      unread: r.unread,
      isPrimary: r.is_primary,
      // A finished session never flags needs-you - a stray ask can't linger past
      // the end, and is:needs-you must not surface it.
      needsYou: attn.has(r.id) && !isFinishedSession(r),
    };
  }

  const effectiveQuery = $derived(
    quick === 'all'
      ? search
      : `${quick === 'pinned' ? 'is:pinned' : 'is:needs-you'} ${search}`.trim(),
  );
  const filter = $derived(parseSessionFilter(effectiveQuery));
  const filtered = $derived(rows.filter((r) => sessionMatchesFilter(toFilterRow(r), filter)));
  const groups = $derived(groupSessions(filtered, { attn }));
  const needsYouCount = $derived(
    rows.filter((r) => attn.has(r.id) && !isFinishedSession(r)).length,
  );
  const pinnedCount = $derived(rows.filter((r) => r.pinned).length);

  // The three live buckets render in the recency flow; ended renders below them
  // as a collapsible, muted section (its own {#if} block, not this loop).
  const GROUP_LABELS: [keyof typeof groups, string][] = [
    ['pinned', 'pinned'],
    ['active', 'active'],
    ['recent', 'recent'],
  ];
  const anyResults = $derived(
    groups.pinned.length + groups.active.length + groups.recent.length + groups.ended.length > 0,
  );

  // Ended section collapse: default-closed, choice persisted like other rail
  // state. An active filter force-opens it so a status: search reaches its rows.
  const ENDED_OPEN_KEY = 'tsugite_rail_ended_open';
  let endedExpanded = $state(readLocal(ENDED_OPEN_KEY) === '1');
  const endedOpen = $derived(endedExpanded || isActiveFilter(filter));
  function toggleEnded() {
    // Toggle against the visible state so the click always does the opposite of
    // what's shown (under a filter the section is force-open and stays so).
    endedExpanded = !endedOpen;
    writeLocal(ENDED_OPEN_KEY, endedExpanded ? '1' : '0');
  }

  // Hand the free-text half to the server so a query reaches the full store, not
  // just the loaded rows. Facet-only queries don't need a server round-trip.
  $effect(() => {
    const q = filterFreeText(filter);
    if (q !== lastServerQuery) {
      lastServerQuery = q;
      onServerSearch(q);
    }
  });

  function onRowDragStart(e: DragEvent, row: Row) {
    if (!e.dataTransfer) return;
    writeSurfaceDrag(e.dataTransfer, {
      kind: 'chat',
      params: { sessionId: row.id },
      title: row.title ?? 'chat',
    });
  }
</script>

<aside class="chat-rail" data-testid={TESTID.chatRail} aria-label="Sessions">
  <div class="rail-hd">
    <div class="rail-search" data-testid={TESTID.chatSearch}>
      <SearchInput
        bind:value={search}
        ariaLabel="Filter sessions"
        placeholder="filter — agent:x status:y"
        shortcutKey="/"
      />
    </div>
    {#if agents.length > 1}
      <Select
        options={agents}
        value={agent}
        ariaLabel="Chat agent"
        onchange={(a) => onAgentChange?.(a)}
      />
    {/if}
    <Button
      variant="pri"
      size="sm"
      iconOnly
      aria-label="New session"
      data-testid={TESTID.chatNewSession}
      onclick={onNew}
    >
      {#snippet icon()}<Icon name="plus" />{/snippet}
    </Button>
  </div>

  <div class="fpills" role="group" aria-label="Quick filters">
    <button
      type="button"
      class="fpill"
      class:is-active={quick === 'all'}
      onclick={() => (quick = 'all')}
    >
      all <span class="n">{rows.length}</span>
    </button>
    <button
      type="button"
      class="fpill fpill--attn"
      class:is-active={quick === 'needs-you'}
      data-testid={TESTID.chatNeedsYou}
      aria-pressed={quick === 'needs-you'}
      onclick={() => (quick = quick === 'needs-you' ? 'all' : 'needs-you')}
    >
      needs you <span class="n">{needsYouCount}</span>
    </button>
    <button
      type="button"
      class="fpill"
      class:is-active={quick === 'pinned'}
      onclick={() => (quick = quick === 'pinned' ? 'all' : 'pinned')}
    >
      pinned <span class="n">{pinnedCount}</span>
    </button>
  </div>

  {#snippet rowItem(row: Row)}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="srow-drag"
      draggable="true"
      ondragstart={(e) => onRowDragStart(e, row)}
      oncontextmenu={(e) => openRowMenu(e, row)}
    >
      <SessionRow
        title={row.title ?? 'Untitled session'}
        when={formatWhen(row.last_active ?? row.created_at)}
        description={sessionTopic(row)}
        state={sessionRowState(row, { pendingAsk: attn.has(row.id) })}
        sourceType={sessionSourceType(row)}
        isActive={row.id === selectedId}
        isPinned={row.pinned}
        isUnread={row.unread}
        onSelect={() => onSelect(row.id)}
        onOpenNewTab={() => openInNewTab(row)}
      />
    </div>
  {/snippet}

  <div class="rail-scroll">
    {#if loading && rows.length === 0}
      <p class="rail-note">loading sessions…</p>
    {:else if rows.length === 0}
      <p class="rail-note">No sessions yet. Start one with the + button.</p>
    {:else if !anyResults}
      <p class="rail-note">No sessions match this filter.</p>
    {:else}
      {#each GROUP_LABELS as [key, label] (key)}
        {#if groups[key].length > 0}
          <div class="sb-group">{label}</div>
          {#each groups[key] as row (row.id)}
            {@render rowItem(row)}
          {/each}
        {/if}
      {/each}

      {#if groups.ended.length > 0}
        <div class="sb-ended">
          <button
            type="button"
            class="sb-ended-hd"
            class:is-open={endedOpen}
            aria-expanded={endedOpen}
            onclick={toggleEnded}
          >
            <span class="chev"><Icon name="chev-r" size={10} /></span>
            ended <span class="n">{groups.ended.length}</span>
          </button>
          {#if endedOpen}
            {#each groups.ended as row (row.id)}
              {@render rowItem(row)}
            {/each}
          {/if}
        </div>
      {/if}
    {/if}
  </div>
</aside>

{#if menu}
  <ContextMenu
    x={menu.x}
    y={menu.y}
    label="Session actions"
    items={menuItems}
    onclose={() => (menu = null)}
  />
{/if}

<style>
  .chat-rail {
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
    background: var(--bg1);
    border-right: 1px solid var(--bd0);
  }
  .rail-hd {
    display: flex;
    gap: 6px;
    align-items: center;
    padding: 8px 10px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
  }
  .rail-search {
    flex: 1;
    min-width: 0;
  }
  .fpills {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    padding: 8px 10px;
    flex: none;
  }
  .fpill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 23px;
    padding: 0 9px;
    border-radius: var(--r-full);
    border: 1px solid var(--bd1);
    background: transparent;
    color: var(--tx2);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    cursor: pointer;
  }
  .fpill:hover {
    color: var(--tx0);
    border-color: var(--tx3);
  }
  .fpill.is-active {
    background: var(--bg3);
    color: var(--tx0);
    border-color: var(--bd1);
  }
  .fpill .n {
    color: var(--tx3);
    font-size: var(--fs-2xs);
  }
  .fpill.is-active .n {
    color: var(--tx1);
  }
  .fpill--attn.is-active {
    background: color-mix(in oklab, var(--st-warn) 15%, transparent);
    border-color: color-mix(in oklab, var(--st-warn) 45%, transparent);
    color: var(--st-warn);
  }
  .rail-scroll {
    flex: 1;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding-bottom: 8px;
  }
  /* sb-group - group header. */
  .sb-group {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
    padding: 10px 12px 4px;
  }
  /* Ended section: a hairline sets the finished chats apart from the live flow,
     and the header doubles as a disclosure toggle (sb-group typography). */
  .sb-ended {
    margin-top: 4px;
    border-top: 1px solid var(--bd0);
  }
  .sb-ended-hd {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    background: none;
    border: 0;
    cursor: pointer;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
    padding: 10px 12px 6px;
  }
  .sb-ended-hd:hover {
    color: var(--tx2);
  }
  .sb-ended-hd .chev {
    display: inline-flex;
    transition: rotate var(--t-2) var(--ease);
  }
  .sb-ended-hd.is-open .chev {
    rotate: 90deg;
  }
  .sb-ended-hd .n {
    color: var(--tx3);
    letter-spacing: 0;
  }
  .srow-drag {
    cursor: grab;
  }
  .rail-note {
    padding: 18px 14px;
    text-align: center;
    color: var(--tx3);
    font: 400 var(--fs-xs) var(--font-ui);
  }
</style>
