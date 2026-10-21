<script lang="ts">
  // Chats context rail: loads the session list and renders the
  // filterable SessionsRail. A row click opens (or focuses) that session as a chat
  // surface in the focused pane via `onOpenChat`; the highlight tracks whichever
  // session the focused chat surface is currently showing (its explicit param, or
  // the resolved default when the paramless default chat is focused).
  import { sessions } from '$lib/stores/sessions.svelte';
  import { agents } from '$lib/stores/agents.svelte';
  import { navigate } from '$lib/router.svelte';
  import { jobs } from '$lib/stores/jobs.svelte';
  import { jobTallyBySession } from '$lib/stores/jobsFilter';
  import SessionsRail from './SessionsRail.svelte';
  import { resolveDefaultSession } from './defaultSession';
  import { attentionSessions } from './sessionModel';

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
  // Highlight the row the focused chat surface shows: its explicit param, else the
  // same default the surface itself resolves to.
  const selectedId = $derived(resolveDefaultSession(rows, focusedSessionId));

  const jobCounts = $derived(jobTallyBySession(jobs.jobs));
  const attn = $derived(new Set(attentionSessions(rows).map((r) => r.id)));

  $effect(() => {
    void sessions.load();
    void agents.load();
  });
  $effect(() => {
    void agents.resolve(sessions.rows.map((r) => r.agent_id));
  });

  async function newSession() {
    const id = await sessions.newSession();
    await sessions.load();
    onOpenChat(id);
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
  {selectedId}
  {attn}
  {jobCounts}
  loading={sessions.loading}
  onSelect={select}
  onNew={newSession}
  onServerSearch={serverSearch}
  onOpenSchedule={(id) => navigate('schedules', { id })}
/>
