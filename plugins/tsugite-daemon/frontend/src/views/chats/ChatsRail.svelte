<script lang="ts">
  // Chats context rail. Loads sessions and rooms and renders the filterable
  // SessionsRail. A row click opens or focuses that session or room in the focused
  // pane via `onOpenChat`. The highlight follows the focused chat surface (its
  // explicit param, or the resolved default) or the focused room.
  import { sessions } from '$lib/stores/sessions.svelte';
  import { agents } from '$lib/stores/agents.svelte';
  import { rooms } from '$lib/stores/rooms.svelte';
  import { spaces } from '$lib/stores/spaces.svelte';
  import { focusedSurface } from '$lib/shell/shellNav';
  import { navigate } from '$lib/router.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { jobs } from '$lib/stores/jobs.svelte';
  import { jobTallyBySession } from '$lib/stores/jobsFilter';
  import SessionsRail from './SessionsRail.svelte';
  import { resolveDefaultSession } from './defaultSession';
  import { attentionSessions } from './sessionModel';
  import { roomBusy } from './roomModel';

  let {
    focusedSessionId,
    onOpenChat,
  }: {
    /** The focused chat surface's raw sessionId param (null for the default chat). */
    focusedSessionId: string | null;
    onOpenChat: (sessionId: string) => void;
  } = $props();

  // Superseded sessions are loaded (view-source targets) but not listed - the
  // live successor is the row that matters here.
  const rows = $derived(sessions.ordered.filter((r) => !r.superseded_by));
  const focused = $derived(focusedSurface(spaces.active.layout));
  const selectedRoomId = $derived(
    focused?.kind === 'room' ? (focused.params.roomId ?? null) : null,
  );
  // Highlight the row the focused chat surface shows: its explicit param, else the
  // same default the surface itself resolves to.
  const selectedId = $derived(
    selectedRoomId ? null : resolveDefaultSession(rows, focusedSessionId),
  );

  const jobCounts = $derived(jobTallyBySession(jobs.jobs));
  const attn = $derived(new Set(attentionSessions(rows).map((r) => r.id)));
  const busyRooms = $derived(
    new Set(rooms.list.filter((r) => roomBusy(r, sessions.ordered)).map((r) => r.id)),
  );

  $effect(() => {
    void sessions.load();
    void agents.load();
    loadRooms();
  });
  $effect(() => {
    void agents.resolve(sessions.rows.map((r) => r.agent_id));
  });

  // A failure shows as the rail's rooms error.
  function loadRooms() {
    rooms.load().catch(() => {});
  }

  async function newSession() {
    const id = await sessions.newSession();
    await sessions.load();
    onOpenChat(id);
  }

  async function openAgent(agentId: string) {
    try {
      const id = await agents.openMain(agentId);
      await sessions.load();
      select(id);
    } catch (err) {
      toasts.push('err', 'Could not open main chat', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // The surface marks read only on a selection change, which re-picking the chat
  // already on screen never makes.
  function select(id: string) {
    void sessions.markViewed(id);
    onOpenChat(id);
  }

  function serverSearch(q: string) {
    void sessions.search(q);
  }
</script>

<SessionsRail
  {rows}
  agents={agents.list}
  agentsError={agents.error}
  onRetryAgents={() => void agents.load()}
  rooms={rooms.list}
  roomsError={rooms.error}
  onRetryRooms={loadRooms}
  {busyRooms}
  {selectedRoomId}
  onOpenRoom={onOpenChat}
  {selectedId}
  {attn}
  {jobCounts}
  loading={sessions.loading}
  onSelect={select}
  onNew={newSession}
  onOpenAgent={openAgent}
  onServerSearch={serverSearch}
  onOpenSchedule={(id) => navigate('schedules', { id })}
/>
