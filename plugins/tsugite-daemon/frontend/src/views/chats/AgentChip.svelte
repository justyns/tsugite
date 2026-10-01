<script lang="ts">
  import Select from '$lib/components/inputs/Select.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { TESTID } from '$lib/testids';
  import { agentLabel, chatAgents } from '../agents/launcher';

  let { sessionId }: { sessionId: string | null } = $props();

  let agentFile = $state<string | null>(null);

  const pool = $derived(chatAgents(agentsMeta.agentFiles));
  const labels = $derived(pool.map((f) => agentLabel(f, agentsMeta.agentFiles)));
  const current = $derived.by(() => {
    if (!agentFile) return pool.find((f) => f.default);
    return pool.find((f) => f.path === agentFile);
  });

  $effect(() => {
    if (agentsMeta.agentFiles.length === 0)
      agentsMeta.loadAgentFiles().catch((err) => {
        toasts.push('err', 'Could not load agents', {
          body: err instanceof Error ? err.message : String(err),
        });
      });
  });

  // settingsRev advances on a settings broadcast from another tab.
  $effect(() => {
    const id = sessionId;
    void (id ? sessions.settingsRev[id] : 0);
    agentFile = null;
    if (!id) return;
    sessions
      .getSettings(id)
      .then((s) => {
        if (sessionId === id) agentFile = s.agent_file;
      })
      .catch((err) => {
        toasts.push('err', 'Could not load agent', {
          body: err instanceof Error ? err.message : String(err),
        });
      });
  });

  async function choose(label: string) {
    const picked = pool[labels.indexOf(label)];
    if (!sessionId || !picked || picked === current) return;
    const prev = agentFile;
    agentFile = picked.default ? null : picked.path;
    try {
      await sessions.patchSettings(sessionId, { agent_file: agentFile });
      toasts.push('ok', `Agent: ${picked.name}`);
    } catch (err) {
      agentFile = prev;
      toasts.push('err', 'Could not change agent', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }
</script>

{#if sessionId && pool.length > 1 && current}
  <span data-testid={TESTID.chatAgentSelect}>
    <Select
      options={labels}
      value={agentLabel(current, agentsMeta.agentFiles)}
      ariaLabel="Chat agent"
      onchange={choose}
    />
  </span>
{/if}
