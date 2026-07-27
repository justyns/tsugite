<script lang="ts">
  // Chats context rail: loads the agent-scoped session list and renders the
  // filterable SessionsRail. A row click opens (or focuses) that session as a chat
  // surface in the focused pane via `onOpenChat`; the highlight tracks whichever
  // session the focused chat surface is currently showing (its explicit param, or
  // the resolved default when the paramless default chat is focused).
  import { sessions } from '$lib/stores/sessions.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import SessionsRail from './SessionsRail.svelte';
  import { resolveDefaultSession } from './defaultSession';

  let {
    focusedSessionId,
    onOpenChat,
  }: {
    /** The focused chat surface's raw sessionId param (null for the default chat). */
    focusedSessionId: string | null;
    onOpenChat: (sessionId: string, agent?: string) => void;
  } = $props();

  // The rail is agent-scoped; the picker swaps which agent's sessions it lists
  // and rides along on every open so cross-agent sessions resolve correctly.
  let agentPick = $state('');
  const agentNames = $derived(agentsMeta.agents.map((a) => a.name));
  const agent = $derived(agentPick || agentNames[0] || '');
  // Superseded sessions are loaded (view-source targets) but not listed - the
  // live successor is the row that matters here.
  const rows = $derived(sessions.ordered.filter((r) => !r.superseded_by));
  // Highlight the row the focused chat surface shows: its explicit param, else the
  // same default the surface itself resolves to.
  const selectedId = $derived(resolveDefaultSession(rows, focusedSessionId));

  const attn = $derived.by(() => {
    const set = new Set<string>();
    for (const r of rows) {
      const text = String(r.progress?.status_text ?? '').toLowerCase();
      if (text.includes('awaiting') || text.includes('question') || text.includes('input on'))
        set.add(r.id);
    }
    return set;
  });

  $effect(() => {
    if (agentsMeta.agents.length === 0) void agentsMeta.load();
  });
  $effect(() => {
    if (agent) void sessions.load(agent);
  });

  async function newSession() {
    const id = await sessions.newSession(agent);
    await sessions.load(agent);
    onOpenChat(id, agent);
  }

  function serverSearch(q: string) {
    void sessions.search(q);
  }
</script>

<SessionsRail
  {rows}
  {agent}
  agents={agentNames}
  {selectedId}
  {attn}
  loading={sessions.loading}
  onSelect={(id) => onOpenChat(id, agent)}
  onNew={newSession}
  onAgentChange={(a) => (agentPick = a)}
  onServerSearch={serverSearch}
/>
