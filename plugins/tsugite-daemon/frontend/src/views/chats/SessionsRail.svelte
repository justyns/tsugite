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
  import Spin from '$lib/components/feedback/Spin.svelte';
  import SessionRow from '$lib/components/rows/SessionRow.svelte';
  import AgentDot from '$lib/components/rows/AgentDot.svelte';
  import ContextMenu, { type ContextMenuItem } from '$lib/components/overlays/ContextMenu.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { spaces } from '$lib/stores/spaces.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { writeSurfaceDrag } from '$lib/shell/mux/drag';
  import { moveItem } from '$lib/reorder';
  import { ReorderDrag } from '$lib/reorderDrag.svelte';
  import { readLocal, writeLocal } from '$lib/storage';
  import type { SessionRow as Row } from '$lib/stores/sessions.svelte';
  import type { Agent } from '$lib/stores/agents.svelte';
  import type { Room } from '$lib/stores/rooms.svelte';
  import type { SessionJobTally } from '$lib/stores/jobsFilter';
  import { TESTID } from '$lib/testids';
  import {
    parseSessionFilter,
    sessionMatchesFilter,
    filterFreeText,
    isActiveFilter,
    type SessionFilterRow,
  } from './sessionFilter';
  import {
    collapseScheduleRuns,
    groupByAgent,
    groupSessions,
    agentSessionList,
    UNKNOWN_AGENT,
    sessionSourceType,
    sessionRowState,
    isFinishedSession,
    formatWhen,
    sessionTopic,
  } from './sessionModel';
  import { sessionStateMeta } from '$lib/components/rows/rowState';
  import { attachRecordToChat, copyReference } from './attachRecord';

  let {
    rows: loadedRows,
    agents = [],
    agentsError = null,
    onRetryAgents,
    rooms = [],
    roomsError = null,
    onRetryRooms,
    busyRooms = new Set(),
    selectedRoomId = null,
    onOpenRoom,
    selectedId,
    attn,
    jobCounts = new Map(),
    loading = false,
    onSelect,
    onNew,
    onOpenAgent,
    onServerSearch,
    onOpenSchedule,
  }: {
    rows: Row[];
    /** More than one agent nests the sessions under their agent. */
    agents?: Agent[];
    agentsError?: string | null;
    onRetryAgents?: () => void;
    /** All rooms, closed ones included. Open rooms get their own section, and every room badges its participants' rows. */
    rooms?: Room[];
    roomsError?: string | null;
    onRetryRooms?: () => void;
    busyRooms?: Set<string>;
    selectedRoomId?: string | null;
    onOpenRoom?: (roomId: string) => void;
    selectedId: string | null;
    attn: Set<string>;
    jobCounts?: Map<string, SessionJobTally>;
    loading?: boolean;
    onSelect: (id: string) => void;
    onNew: () => void;
    onOpenAgent?: (agentId: string) => void;
    /** Free-text portion of the query, for the store's server-merge search. */
    onServerSearch: (q: string) => void;
    /** Opens the Schedules view with this schedule's drawer open. */
    onOpenSchedule: (scheduleId: string) => void;
  } = $props();

  const collapsed = $derived(collapseScheduleRuns(loadedRows, attn));
  const rows = $derived(collapsed.rows);

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
      params: { sessionId: row.id },
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
      await sessions.load();
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
      ...(row.needs_attention === true
        ? [{ label: 'Dismiss attention', run: () => void sessions.dismissAttention(row.id) }]
        : []),
      { label: 'Copy session id', run: () => void copyId(row.id) },
      { label: 'Add to chat', run: () => void attachRecordToChat('session', row.id) },
      { label: 'Copy reference', run: () => void copyReference('session', row.id) },
      ...(!isFinishedSession(row)
        ? [{ label: 'Mark complete', run: () => void completeRow(row.id) }]
        : []),
    ];
  });

  function rowNeedsYou(r: Row): boolean {
    return attn.has(r.id) && !isFinishedSession(r);
  }

  function toFilterRow(r: Row): SessionFilterRow {
    return {
      id: r.id,
      title: r.title,
      topic: typeof r.metadata?.topic === 'string' ? r.metadata.topic : undefined,
      label: r.label,
      status: r.status,
      pinned: r.pinned,
      unread: r.unread,
      isPrimary: r.is_primary,
      // A finished session never flags needs-you - a stray ask can't linger past
      // the end, and is:needs-you must not surface it.
      needsYou: rowNeedsYou(r),
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
  const needsYouCount = $derived(rows.filter(rowNeedsYou).length);
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

  // Ended sections and agent groups persist their open/closed choice like other
  // rail state. An active filter force-opens both.
  const ENDED_OPEN_KEY = 'tsugite_rail_ended_open';
  const AGENT_OPEN_KEY = 'tsugite_rail_agent_open';
  let railChoice = $state<Record<string, string>>({});
  function storedChoice(key: string): string | null {
    return railChoice[key] ?? readLocal(key);
  }
  function storeChoice(key: string, open: boolean) {
    railChoice[key] = open ? '1' : '0';
    writeLocal(key, railChoice[key]);
  }

  function endedOpen(key: string): boolean {
    return isActiveFilter(filter) || storedChoice(key) === '1';
  }

  const roomById = $derived(new Map(rooms.map((r) => [r.id, r])));
  const liveRooms = $derived(
    isActiveFilter(filter) ? [] : rooms.filter((r) => !isFinishedSession(r)),
  );

  const agentGroups = $derived(agents.length > 1 ? groupByAgent(agents, rows, attn) : null);
  const agentById = $derived(new Map(agents.map((a) => [a.id, a])));
  const filteredIds = $derived(new Set(filtered.map((r) => r.id)));
  const hiddenMainCount = $derived(
    agentGroups?.filter((g) => g.main && !g.main.pinned).length ?? 0,
  );

  const selectedRowId = $derived(collapsed.rowOf.get(selectedId ?? '') ?? selectedId);
  const selectedGroupId = $derived.by(() => {
    const row = rows.find((r) => r.id === selectedRowId);
    if (!row || row.pinned || agentGroups?.some((g) => g.main === row)) return null;
    return agentById.has(row.agent_id) ? row.agent_id : UNKNOWN_AGENT.id;
  });
  // Opening a session expands its agent until the user collapses it, without
  // writing the stored choice.
  let autoOpenAgent = $derived(selectedId === null ? null : selectedGroupId);
  // Likewise the open ended session expands its agent's ended row.
  let autoOpenEnded = $derived(selectedRowId);

  const AGENT_ROW_CAP = 5;
  let agentShowAll = $state<Record<string, boolean>>({});

  function agentOpen(agent: Agent): boolean {
    if (isActiveFilter(filter) || autoOpenAgent === agent.id) return true;
    const stored = storedChoice(`${AGENT_OPEN_KEY}:${agent.id}`);
    return stored === null ? agent.is_default : stored === '1';
  }
  function toggleAgent(agent: Agent) {
    const open = agentOpen(agent);
    if (autoOpenAgent === agent.id) autoOpenAgent = null;
    storeChoice(`${AGENT_OPEN_KEY}:${agent.id}`, !open);
  }
  function toggleEnded(key: string, isOpen: boolean) {
    autoOpenEnded = null;
    storeChoice(key, !isOpen);
  }
  function agentLabel(name: string, busy: boolean, needsYou: number): string {
    return [name, busy && 'working', needsYou > 0 && `${needsYou} needs you`]
      .filter(Boolean)
      .join(', ');
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
    if (row.pinned && canReorderPins) pinDrag.start(row.id);
  }

  const pinnedIds = $derived(groups.pinned.map((p) => p.id));
  const pinDrag = new ReorderDrag((id) => pinnedIds.indexOf(id), { axis: 'y' });

  // A filter hides part of the pin set, and the daemon rewrites pin_position
  // 0..N-1 for exactly the ids it is sent - reordering what is visible would
  // drop the hidden pins onto positions the visible ones now hold.
  const canReorderPins = $derived(!isActiveFilter(filter));

  function onPinnedDragOver(e: DragEvent, row: Row) {
    if (!row.pinned) return;
    pinDrag.over(e, row.id);
  }

  function onPinnedDrop(e: DragEvent, row: Row) {
    if (!row.pinned) return;
    const move = pinDrag.drop(e);
    if (move)
      void sessions.reorderPins(moveItem(pinnedIds, pinnedIds.indexOf(move.id), move.insertAt));
  }
</script>

<aside class="chat-rail" data-testid={TESTID.chatRail} aria-label="Sessions">
  <div class="rail-hd">
    <div class="rail-search" data-testid={TESTID.chatSearch}>
      <SearchInput
        bind:value={search}
        ariaLabel="Filter sessions"
        placeholder="filter by status:x"
        shortcutKey="/"
      />
    </div>
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
      all <span class="n">{rows.length - hiddenMainCount}</span>
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
    {@const display = sessionRowState(row, { needsYou: rowNeedsYou(row) })}
    {@const room = roomById.get(String(row.metadata?.room_id ?? ''))}
    {@const index = pinnedIds.indexOf(row.id)}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="srow-drag"
      class:is-dragging={pinDrag.dragging === row.id}
      class:drop-above={row.pinned && pinDrag.dropAt === index}
      class:drop-below={row.pinned &&
        pinDrag.dropAt === index + 1 &&
        index === pinnedIds.length - 1}
      draggable="true"
      ondragstart={(e) => onRowDragStart(e, row)}
      ondragend={() => pinDrag.end()}
      ondragover={(e) => onPinnedDragOver(e, row)}
      ondrop={(e) => onPinnedDrop(e, row)}
      oncontextmenu={(e) => openRowMenu(e, row)}
    >
      <SessionRow
        title={row.title ?? 'Untitled session'}
        when={formatWhen(row.last_active ?? row.created_at)}
        description={sessionTopic(row)}
        state={display.state}
        needsYou={display.needsYou}
        sourceType={sessionSourceType(row)}
        isActive={row.id === selectedRowId}
        isPinned={row.pinned}
        isUnread={row.unread}
        activeJobCount={jobCounts.get(row.id)?.open ?? 0}
        waitingOnCount={row.waiting_on?.length ?? 0}
        runCount={collapsed.runs.get(row.id) ?? 0}
        agent={agentGroups && row.pinned
          ? (agentById.get(row.agent_id) ?? UNKNOWN_AGENT)
          : undefined}
        onSelect={() => onSelect(row.id)}
        onOpenSchedule={() => onOpenSchedule(row.parent_id!)}
        onOpenNewTab={() => openInNewTab(row)}
        {room}
        onOpenRoom={() => onOpenRoom?.(room!.id)}
      />
    </div>
  {/snippet}

  {#snippet endedSection(ended: Row[], key: string, nested = false)}
    {@const isOpen = endedOpen(key) || (nested && ended.some((r) => r.id === autoOpenEnded))}
    {#if ended.length > 0 && nested}
      <button
        type="button"
        class="agent-more"
        class:is-open={isOpen}
        aria-expanded={isOpen}
        onclick={() => toggleEnded(key, isOpen)}
      >
        <span class="chev"><Icon name="chev-r" size={10} /></span>
        {ended.length} ended
      </button>
    {:else if ended.length > 0}
      <div class="sb-ended">
        <button
          type="button"
          class="sb-ended-hd"
          class:is-open={isOpen}
          aria-expanded={isOpen}
          onclick={() => storeChoice(key, !isOpen)}
        >
          <span class="chev"><Icon name="chev-r" size={10} /></span>
          ended <span class="n">{ended.length}</span>
        </button>
      </div>
    {/if}
    {#if ended.length > 0 && isOpen}
      {#each ended as row (row.id)}
        {@render rowItem(row)}
      {/each}
    {/if}
  {/snippet}

  <div class="rail-scroll">
    {#if liveRooms.length > 0}
      <div class="sb-group">rooms</div>
      {#each liveRooms as room (room.id)}
        {@const n = room.participants.length}
        <SessionRow
          title={room.title}
          when={formatWhen(room.created_at)}
          description="{n} agent{n === 1 ? '' : 's'}"
          state={busyRooms.has(room.id) ? 'running' : 'idle'}
          needsYou={room.needs_attention}
          sourceType="room"
          isActive={room.id === selectedRoomId}
          onSelect={() => onOpenRoom?.(room.id)}
        />
      {/each}
    {/if}
    {#if roomsError}
      <p class="rail-note">
        Couldn't load rooms
        <Button size="sm" onclick={onRetryRooms}>Retry</Button>
      </p>
    {/if}
    {#if agentsError}
      <p class="rail-note">
        Couldn't load agents
        <Button size="sm" onclick={onRetryAgents}>Retry</Button>
      </p>
    {/if}
    {#if loading && rows.length === 0}
      <p class="rail-note">loading sessions…</p>
    {:else if rows.length === 0}
      <p class="rail-note">No sessions yet. Start one with the + button.</p>
    {:else if !anyResults}
      <p class="rail-note">No sessions match this filter.</p>
    {:else}
      {#if agentGroups}
        {#if groups.pinned.length > 0}
          <div class="sb-group">pinned</div>
          {#each groups.pinned as row (row.id)}
            {@render rowItem(row)}
          {/each}
        {/if}
        {#each agentGroups as group (group.agent.id)}
          {@const shown = group.rows.filter(
            (r) => !r.pinned && r !== group.main && filteredIds.has(r.id),
          )}
          {@const mainShown = group.main !== undefined && filteredIds.has(group.main.id)}
          {#if shown.length > 0 || mainShown || !isActiveFilter(filter)}
            {@const open = agentOpen(group.agent)}
            {@const active = group.main !== undefined && group.main.id === selectedRowId}
            <div class="agent-hd" class:is-open={open} class:is-active={active}>
              <button
                type="button"
                class="agent-toggle"
                aria-expanded={open}
                aria-label={agentLabel(group.agent.name, group.busy, group.needsYou)}
                onclick={() => toggleAgent(group.agent)}
              >
                <span class="chev"><Icon name="chev-r" size={10} /></span>
              </button>
              <button
                type="button"
                class="agent-open"
                aria-label="Open {group.agent.name} main chat"
                aria-current={active ? 'true' : undefined}
                onclick={() => onOpenAgent?.(group.agent.id)}
              >
                <AgentDot color={group.agent.color} />
                <span class="agent-name">{group.agent.name}</span>
                {#if group.busy}<Spin color={sessionStateMeta('running').color} />{/if}
                {#if group.needsYou > 0}
                  <span class="agent-attn">{group.needsYou}</span>
                {/if}
              </button>
            </div>
            {#if open}
              {@const list = agentSessionList(shown, attn)}
              {@const cap =
                isActiveFilter(filter) || agentShowAll[group.agent.id]
                  ? list.live.length
                  : AGENT_ROW_CAP}
              {@const visible = list.live.filter((r, i) => i < cap || r.id === selectedRowId)}
              {#each visible as row (row.id)}
                {@render rowItem(row)}
              {/each}
              {#if list.live.length > visible.length}
                <button
                  type="button"
                  class="agent-more"
                  onclick={() => (agentShowAll[group.agent.id] = true)}
                >
                  show {list.live.length - visible.length} more
                </button>
              {/if}
              {@render endedSection(list.ended, `${ENDED_OPEN_KEY}:${group.agent.id}`, true)}
            {/if}
          {/if}
        {/each}
      {:else}
        {#each GROUP_LABELS as [key, label] (key)}
          {#if groups[key].length > 0}
            <div class="sb-group">{label}</div>
            {#each groups[key] as row (row.id)}
              {@render rowItem(row)}
            {/each}
          {/if}
        {/each}
        {@render endedSection(groups.ended, ENDED_OPEN_KEY)}
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
  .sb-ended-hd .chev,
  .agent-hd .chev,
  .agent-more .chev {
    display: inline-flex;
    transition: rotate var(--t-2) var(--ease);
  }
  .sb-ended-hd.is-open .chev,
  .agent-hd.is-open .chev,
  .agent-more.is-open .chev {
    rotate: 90deg;
  }
  .agent-hd {
    display: flex;
    align-items: stretch;
    min-width: 0;
    border-top: 1px solid var(--bd0);
    border-left: 2px solid transparent;
  }
  .agent-hd.is-active {
    background: var(--bg2);
    border-left-color: var(--acc);
  }
  .agent-toggle,
  .agent-open {
    display: flex;
    align-items: center;
    background: none;
    border: 0;
    cursor: pointer;
    color: var(--tx1);
  }
  .agent-toggle {
    flex: none;
    padding: 8px 4px 8px 10px;
    border-radius: var(--r-sm);
  }
  .agent-toggle:hover {
    background: var(--bg3);
  }
  .agent-open:hover {
    background: var(--bg2);
  }
  .agent-open:hover .agent-name {
    text-decoration: underline;
  }
  .agent-open {
    flex: 1;
    min-width: 0;
    gap: 6px;
    font: 600 var(--fs-sm) / 1.3 var(--font-ui);
    padding: 8px 12px 8px 2px;
  }
  .agent-name {
    flex: 1;
    min-width: 0;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .agent-more {
    display: flex;
    align-items: center;
    gap: 4px;
    width: 100%;
    background: none;
    border: 0;
    cursor: pointer;
    text-align: left;
    font: 500 var(--fs-xs) / 1.3 var(--font-ui);
    color: var(--tx2);
    padding: 4px 12px 6px 34px;
  }
  .agent-more:hover {
    color: var(--tx1);
  }
  .agent-attn {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    color: var(--tx0);
    padding: 1px 5px;
    border: 1px solid var(--st-warn);
    border-radius: var(--r-full);
    background: color-mix(in oklab, var(--st-warn) 18%, var(--bg1));
    flex: none;
  }
  .sb-ended-hd .n {
    color: var(--tx3);
    letter-spacing: 0;
  }
  .srow-drag {
    cursor: grab;
  }
  .srow-drag.is-dragging {
    opacity: 0.4;
  }
  .srow-drag.drop-above {
    box-shadow: inset 0 2px 0 0 var(--acc);
  }
  .srow-drag.drop-below {
    box-shadow: inset 0 -2px 0 0 var(--acc);
  }
  .rail-note {
    padding: 18px 14px;
    text-align: center;
    color: var(--tx3);
    font: 400 var(--fs-xs) var(--font-ui);
  }
</style>
