<script lang="ts">
  // Room view. Shows the shared transcript of agents and the person, participant
  // chips, the round counter, and a composer that posts to the room.
  import { untrack } from 'svelte';
  import { rooms, type RoomParticipant } from '$lib/stores/rooms.svelte';
  import { agents } from '$lib/stores/agents.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { jobs } from '$lib/stores/jobs.svelte';
  import { jobDrawerRequest } from '../jobs/jobDrawerSignal.svelte';
  import { conn } from '$lib/stores/conn.svelte';
  import { buildHash } from '$lib/router.svelte';
  import { clockTime } from '$lib/relativeTime';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import Msg from '$lib/components/chatturns/Msg.svelte';
  import Prose from '$lib/components/chatturns/Prose.svelte';
  import Composer from '$lib/components/composer/Composer.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import AgentDot from '$lib/components/rows/AgentDot.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import PhoneBack from '$lib/shell/PhoneBack.svelte';
  import JobPill from '../jobs/JobPill.svelte';
  import { sessionStateMeta } from '$lib/components/rows/rowState';
  import { goBackToWorkspaceList } from '$lib/shell/phoneNav';
  import type { RefItem } from '$lib/components/composer/types';
  import { ScrollFollow } from './scrollFollow.svelte';
  import { fetchEventsPage } from './conversationController.svelte';
  import { isFinishedSession, liveSessionId } from './sessionModel';
  import {
    USER_AUTHOR,
    cardHref,
    messageTargets,
    participantBusy,
    pendingCapCard,
    roomEvents,
    workingParticipants,
    type RoomEvent,
  } from './roomModel';

  let { params, dense = false }: { params?: Record<string, string>; dense?: boolean } = $props();

  const CAP_STEP = 12;
  const PAGE = 300;
  const MEMBER_VERB = { join: 'joined', leave: 'left', mute: 'was muted', unmute: 'was unmuted' };

  const roomId = $derived(params?.roomId ?? '');
  const room = $derived(rooms.byId(roomId));
  const agentById = $derived(new Map(agents.list.map((a) => [a.id, a])));
  const open = $derived(room !== undefined && !isFinishedSession(room));
  const working = $derived(
    room ? workingParticipants(room, sessions.ordered).map((p) => label(p.agent_id)) : [],
  );
  const busy = $derived(working.length > 0);
  const job = $derived(room?.job_id ? jobs.jobs.find((j) => j.job_id === room.job_id) : undefined);

  let events = $state<RoomEvent[]>([]);
  let hasEarlier = $state(false);
  let loadingEarlier = $state(false);
  let loadError = $state<string | null>(null);
  let syncError = $state<string | null>(null);
  let draft = $state('');
  let composer = $state<ReturnType<typeof Composer>>();
  let oldestId: number | null = null;
  // Bumped on every room switch. A response for an earlier opening is dropped.
  let generation = 0;

  const atCap = $derived(
    room !== undefined &&
      (room.rounds_used >= room.max_rounds || pendingCapCard(events) !== undefined),
  );
  const targets = $derived(room ? messageTargets(draft, room) : null);
  const refItems = $derived<RefItem[]>(
    (room?.participants ?? []).map((p) => ({
      id: p.agent_id,
      kind: 'agent',
      label: `@${p.agent_id}`,
      detail: [label(p.agent_id), p.lead && 'lead'].filter(Boolean).join(' · '),
    })),
  );

  function label(agentId: string): string {
    return agentById.get(agentId)?.name ?? agentId;
  }

  function message(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
  }

  function add(list: { type?: unknown }[]) {
    const seen = new Set(events.map((e) => e.id));
    const fresh = roomEvents(list).filter((e) => !seen.has(e.id));
    if (fresh.length) events = [...events, ...fresh].sort((a, b) => a.id - b.id);
  }

  async function openRoom() {
    const gen = generation;
    loadError = null;
    try {
      const page = await fetchEventsPage(roomId, { limit: PAGE });
      if (gen !== generation) return;
      add(page.events);
      hasEarlier = page.hasMore;
      oldestId = page.oldestId;
    } catch (err) {
      if (gen === generation) loadError = message(err);
    }
  }

  async function resync() {
    const gen = generation;
    syncError = null;
    try {
      const page = await fetchEventsPage(roomId, { afterId: events.at(-1)?.id ?? null });
      if (gen === generation) add(page.events);
    } catch (err) {
      if (gen === generation) syncError = message(err);
    }
  }

  async function loadEarlier() {
    const gen = generation;
    loadingEarlier = true;
    try {
      await follow.preserveAcross(async () => {
        const page = await fetchEventsPage(roomId, { beforeId: oldestId, limit: PAGE });
        if (gen !== generation) return;
        add(page.events);
        hasEarlier = page.hasMore;
        oldestId = page.oldestId ?? oldestId;
      });
    } catch (err) {
      if (gen === generation)
        toasts.push('err', 'Could not load earlier messages', { body: message(err) });
    } finally {
      if (gen === generation) loadingEarlier = false;
    }
  }

  $effect(() => {
    const id = roomId;
    untrack(() => {
      generation++;
      events = [];
      hasEarlier = false;
      oldestId = null;
      loadError = null;
      syncError = null;
      if (id) void openRoom();
    });
    if (!id) return;
    // Live frames have no timestamp. They arrive as the event is recorded.
    return sessions.bindConversation(id, (frame) =>
      add([
        Object.assign({ timestamp: new Date().toISOString() }, frame, { type: frame.event_type }),
      ]),
    );
  });
  $effect(() => () => generation++);

  // Frames sent while the event stream was down are fetched on reconnect.
  let prevConn = conn.status;
  $effect(() => {
    const status = conn.status;
    const reconnected = status === 'live' && (prevConn === 'reconnecting' || prevConn === 'lost');
    prevConn = status;
    if (reconnected) untrack(() => void (events.length ? resync() : openRoom()));
  });

  const jobId = $derived(room?.job_id ?? null);
  $effect(() => {
    const id = jobId;
    if (id && !untrack(() => job)) jobs.fetch(id).catch(() => {});
  });

  $effect(() => {
    if (!rooms.loaded) rooms.load().catch(() => {});
    void agents.load();
    void sessions.load();
  });

  let scrollEl = $state<HTMLElement>();
  const follow = new ScrollFollow();
  $effect(() => {
    if (!scrollEl) return;
    return follow.attach(scrollEl);
  });
  $effect(() => {
    void events.length;
    follow.sync();
  });

  async function act(run: () => Promise<void>, failure: string) {
    try {
      await run();
    } catch (err) {
      toasts.push('err', failure, { body: message(err) });
    }
  }

  function send(text: string) {
    follow.repin();
    void act(async () => {
      try {
        await rooms.post(roomId, text);
      } catch (err) {
        if (!draft) draft = text;
        throw err;
      }
    }, 'Could not post to the room');
  }

  function raiseCap() {
    if (!room) return;
    const maxRounds = Math.max(room.max_rounds, room.rounds_used) + CAP_STEP;
    void act(() => rooms.setMaxRounds(room.id, maxRounds), 'Could not raise the round cap');
  }

  function mention(agentId: string) {
    draft = `${draft}${draft && !/\s$/.test(draft) ? ' ' : ''}@${agentId} `;
    composer?.focus();
  }

  function sessionHref(p: RoomParticipant): string {
    return buildHash('chats', { sessionId: liveSessionId(sessions.ordered, p.session_id) });
  }

  function toggleMute(p: RoomParticipant) {
    const name = label(p.agent_id);
    void act(
      () => rooms.setMuted(roomId, p.agent_id, !p.muted),
      p.muted ? `Could not unmute ${name}` : `Could not mute ${name}`,
    );
  }
</script>

<section class="room" data-density={dense ? 'dense' : undefined} aria-label="Room">
  <header class="room-hd">
    <PhoneBack onBack={() => goBackToWorkspaceList('chats')} label="Back to chats" />
    <h2>{room?.title ?? 'Room'}</h2>
    {#if room && !open}<span class="closed">closed</span>{/if}
    {#if job}
      <a class="job" href="#jobs" onclick={() => jobDrawerRequest.request(job.job_id)}>
        <span class="job-title">{job.prompt}</span>
        <JobPill st={job.state} />
      </a>
    {/if}
    <span class="grow"></span>
    {#if room}
      <span class="rounds" class:is-cap={atCap}>
        rounds {room.rounds_used}/{room.max_rounds}
      </span>
      {#if atCap && open}
        <Button size="sm" onclick={raiseCap}>+{CAP_STEP} rounds</Button>
      {/if}
    {/if}
  </header>

  {#if room}
    <div class="chips" role="group" aria-label="Participants">
      {#each room.participants as p (p.agent_id)}
        {@const agent = agentById.get(p.agent_id)}
        {@const name = label(p.agent_id)}
        {#snippet chipBody()}
          <AgentDot color={agent?.color ?? null} />
          <span>{name}</span>
          <span class="pmodel">{agent?.model ?? 'default'}</span>
          {#if p.lead}<span class="ptag">lead</span>{/if}
          {#if participantBusy(p, sessions.ordered)}
            <span class="pbusy" title="working"
              ><Spin color={sessionStateMeta('running').color} /></span
            >
          {/if}
        {/snippet}
        <span class="pchip" class:is-muted={p.muted}>
          {#if p.lead || !open}
            <a
              class="pmain"
              href={sessionHref(p)}
              title="@{p.agent_id}"
              aria-label="Open {name} session">{@render chipBody()}</a
            >
          {:else}
            <button
              type="button"
              class="pmain"
              title="@{p.agent_id}"
              aria-label="Mention {name}"
              onclick={() => mention(p.agent_id)}>{@render chipBody()}</button
            >
            <a class="popen" href={sessionHref(p)} aria-label="Open {name} session"
              ><Icon name="chev-r" size={11} /></a
            >
            <button
              type="button"
              class="pmute"
              aria-pressed={p.muted}
              aria-label="Mute {name}"
              onclick={() => toggleMute(p)}>{p.muted ? 'muted' : 'mute'}</button
            >
          {/if}
        </span>
      {/each}
    </div>
  {:else if rooms.error}
    <p class="sys is-err">
      Couldn't load rooms: {rooms.error}
      <Button size="sm" onclick={() => rooms.load().catch(() => {})}>Retry</Button>
    </p>
  {/if}

  <div class="room-scroll" bind:this={scrollEl}>
    {#if loadError && events.length === 0}
      <p class="room-empty is-err">
        {loadError}
        <Button size="sm" onclick={() => void openRoom()}>Retry</Button>
      </p>
    {:else if events.length === 0}
      <p class="room-empty">No messages yet.</p>
    {/if}
    {#if hasEarlier}
      <button
        type="button"
        class="load-earlier"
        disabled={loadingEarlier}
        onclick={() => void loadEarlier()}
      >
        {loadingEarlier ? 'loading…' : 'load earlier messages'}
      </button>
    {/if}
    {#each events as e (e.id)}
      {#if e.type === 'room_message'}
        {@const user = e.author === USER_AUTHOR}
        <Msg
          role={user ? 'user' : 'ai'}
          who={user ? 'you' : e.role === 'opener' ? `${label(e.author)} (opener)` : label(e.author)}
          agentColor={user ? undefined : (agentById.get(e.author)?.color ?? null)}
          at={clockTime(e.timestamp)}
        >
          <Prose content={e.content} breaks={user} />
          {#if !user && e.private_ref != null && e.participant_session_id}
            <a class="worked" href={buildHash('chats', { sessionId: e.participant_session_id })}
              >worked in {label(e.author)} ›</a
            >
          {/if}
        </Msg>
      {:else if e.type === 'room_member'}
        <p class="sys">{label(e.agent_id)} {MEMBER_VERB[e.action]}</p>
      {:else}
        {@const href = cardHref(e)}
        {@const text = e.agent_id ? `${label(e.agent_id)}: ${e.text}` : e.text}
        <p
          class="sys"
          class:is-ok={e.kind === 'job_done'}
          class:is-err={e.kind === 'turn_failed' || e.kind === 'job_failed'}
          class:is-warn={e.kind === 'round_cap' || e.kind === 'interrupted'}
          class:is-quiet={e.kind === 'turn_skipped'}
        >
          {#if href}<a {href} onclick={() => e.job_id && jobDrawerRequest.request(e.job_id)}
              >{text}</a
            >{:else}{text}{/if}
        </p>
      {/if}
    {/each}
    {#if syncError}
      <p class="sys is-err">
        Couldn't refresh: {syncError}
        <Button size="sm" onclick={() => void resync()}>Retry</Button>
      </p>
    {/if}
  </div>

  {#if open && targets}
    {#if busy}
      <p class="working" role="status">
        <Spin color={sessionStateMeta('running').color} />
        {new Intl.ListFormat('en').format(working)}
        {working.length === 1 ? 'is' : 'are'} working…
      </p>
    {/if}
    <p class="target" aria-live="polite">
      to
      {#if targets.mentioned.length > 0}
        {targets.mentioned.map(label).join(', ')}, then lead {label(targets.lead)}
      {:else}
        lead {label(targets.lead)}
      {/if}
    </p>
    <Composer
      bind:this={composer}
      bind:value={draft}
      {dense}
      {busy}
      {refItems}
      placeholder="message the room · @ to address an agent"
      onSend={send}
      onStop={() => void act(() => rooms.stop(roomId), 'Could not stop the room')}
    />
  {/if}
</section>

<style>
  .room {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
    min-height: 0;
    background: var(--bg0);
  }
  .room-hd {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 14px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
    min-width: 0;
  }
  .room-hd h2 {
    margin: 0;
    min-width: 0;
    color: var(--tx0);
    font: 600 var(--fs-lg) / 1.2 var(--font-ui);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .grow {
    flex: 1;
  }
  .closed,
  .rounds {
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
  }
  .rounds.is-cap {
    color: var(--st-warn);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 7px 14px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
  }
  .pchip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
    height: 22px;
    padding: 0 4px 0 0;
    border: 1px solid var(--bd1);
    border-radius: var(--r-full);
    background: var(--bg1);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
  }
  .pchip.is-muted {
    opacity: 0.55;
  }
  .pmain {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
    align-self: stretch;
    padding: 0 2px 0 8px;
    background: none;
    border: 0;
    border-radius: var(--r-full);
    color: inherit;
    font: inherit;
    text-decoration: none;
    cursor: pointer;
  }
  .pmain:hover {
    color: var(--tx0);
  }
  .popen {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    align-self: stretch;
    padding: 0 3px;
    border-radius: var(--r-full);
    color: var(--tx3);
  }
  .popen:hover {
    background: var(--bg3);
    color: var(--tx0);
  }
  @media (pointer: coarse) {
    .pchip {
      height: 30px;
    }
    .popen,
    .pmute {
      min-width: 28px;
    }
  }
  .pmodel,
  .ptag {
    color: var(--tx3);
    font-size: var(--fs-2xs);
  }
  .pbusy {
    display: inline-flex;
  }
  .pmute {
    background: none;
    border: 0;
    border-radius: var(--r-full);
    padding: 1px 6px;
    cursor: pointer;
    color: var(--tx2);
    font: inherit;
    font-size: var(--fs-2xs);
  }
  .pmute:hover,
  .pmute[aria-pressed='true'] {
    background: var(--bg3);
    color: var(--tx0);
  }
  .room-scroll {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    overscroll-behavior: contain;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .room-empty {
    margin: auto;
    padding: 40px 20px;
    color: var(--tx3);
    font: 400 var(--fs-sm) var(--font-ui);
  }
  .room-empty.is-err,
  .sys.is-err {
    color: var(--st-err);
  }
  .sys {
    margin: 0;
    padding: 6px 18px;
    text-align: center;
    color: var(--tx3);
    font: 500 var(--fs-2xs) / 1.5 var(--font-mono);
    border-top: 1px solid color-mix(in oklab, var(--bd0) 55%, transparent);
  }
  .load-earlier {
    align-self: center;
    margin: 10px 0 4px;
    background: none;
    border: 0;
    padding: 4px 10px;
    border-radius: var(--r-full);
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    cursor: pointer;
  }
  .load-earlier:hover:not(:disabled) {
    color: var(--acc);
    background: var(--bg2);
  }
  .sys.is-warn {
    color: var(--st-warn);
  }
  .sys.is-quiet {
    opacity: 0.7;
  }
  .sys.is-ok {
    color: var(--st-ok);
  }
  .sys a {
    color: inherit;
  }
  .job {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    color: var(--tx2);
    font: 500 var(--fs-xs) var(--font-ui);
    text-decoration: none;
  }
  .job:hover {
    color: var(--tx0);
  }
  .job-title {
    min-width: 0;
    max-width: 24ch;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .worked {
    justify-self: start;
    padding: 2px 7px;
    border: 1px solid var(--bd0);
    border-radius: var(--r-sm);
    background: var(--bg1);
    color: var(--tx3);
    font: 500 var(--fs-2xs) / 1.4 var(--font-mono);
    text-decoration: none;
  }
  .worked:hover {
    color: var(--tx1);
    border-color: var(--bd1);
  }
  .working {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0;
    padding: 5px 14px 0;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    color: var(--tx2);
    font: 500 var(--fs-2xs) var(--font-mono);
    flex: none;
  }
  .working + .target {
    border-top: 0;
  }
  .target {
    margin: 0;
    padding: 5px 14px 0;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    flex: none;
  }
  .target + :global(.composer) {
    border-top: 0;
  }
</style>
