<script lang="ts">
  // Header model + effort pair: the model chip/popover plus a reasoning-effort
  // seg sized to what the session's resolved model actually supports
  // (GET /api/agents/{agent}/effort-levels?session_id=). A model with no
  // declared effort levels gets no seg at all, and the resolved model string
  // also names the picker's "default" chip. Effort is the persisted per-session
  // setting (GET/PATCH /api/sessions/{id}/settings), not a per-message override.
  import Seg from '$lib/components/inputs/Seg.svelte';
  import { api } from '$lib/api/client';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { TESTID } from '$lib/testids';
  import ModelPicker from './ModelPicker.svelte';

  let { sessionId, agent }: { sessionId: string | null; agent: string } = $props();

  // Seg labels stay compact; everything not listed here shows verbatim.
  const SHORT: Record<string, string> = { minimal: 'min', medium: 'med' };
  const display = (level: string) => SHORT[level] ?? level;

  let levels = $state<string[] | null>(null);
  let resolvedModel = $state<string | null>(null);
  // Bumped after a model change so the capability list refetches.
  let modelRev = $state(0);

  // The seg's bound value is a display label; `persisted` mirrors the server's
  // level word so the loader's write doesn't look like a user edit.
  let segValue = $state('');
  let persisted = $state('');

  let levelsKey = '';
  $effect(() => {
    const id = sessionId;
    // settingsRev advances on a cross-tab settings broadcast, so a model change
    // elsewhere refetches the capability list (levels are model-dependent).
    const rev = id ? (sessions.settingsRev[id] ?? 0) : 0;
    const key = `${id ?? ''}#${modelRev}#${rev}`;
    if (key === levelsKey) return;
    levelsKey = key;
    levels = null;
    resolvedModel = null;
    if (!id) return;
    api
      .get<{ model: string; supported_effort_levels: string[] | null }>(
        `/api/agents/${encodeURIComponent(agent)}/effort-levels?session_id=${encodeURIComponent(id)}`,
      )
      .then((res) => {
        if (levelsKey !== key) return;
        resolvedModel = res.model;
        levels = res.supported_effort_levels;
      })
      .catch(() => {});
  });

  let settingsFor: string | null = null;
  let settingsRev = -1;
  $effect(() => {
    const id = sessionId;
    const rev = id ? (sessions.settingsRev[id] ?? 0) : 0;
    const idChanged = id !== settingsFor;
    if (!idChanged && rev === settingsRev) return;
    settingsFor = id;
    settingsRev = rev;
    if (idChanged) {
      segValue = ''; // a session switch, not a live update: clear before refetch
      persisted = '';
    }
    if (!id) return;
    sessions
      .getSettings(id)
      .then((s) => {
        if (settingsFor !== id) return;
        persisted = s.reasoning_effort ?? '';
        segValue = persisted ? display(persisted) : '';
      })
      .catch(() => {});
  });

  const options = $derived((levels ?? []).map(display));
  // With no persisted choice, the seg rests on medium when the model offers it,
  // else its middle option - display-only until the user actually picks one.
  const shownValue = $derived.by(() => {
    if (segValue && options.includes(segValue)) return segValue;
    if (levels?.includes('medium')) return display('medium');
    return options[Math.floor((options.length - 1) / 2)] ?? '';
  });

  function onPick(label: string) {
    const level = (levels ?? []).find((l) => display(l) === label);
    const id = sessionId;
    if (!level || !id || level === persisted) return;
    const prev = persisted;
    persisted = level;
    segValue = label;
    void sessions
      .patchSettings(id, { reasoning_effort: level })
      .then(() => toasts.push('ok', `Reasoning effort → ${level}`))
      .catch((err) => {
        persisted = prev;
        segValue = prev ? display(prev) : '';
        toasts.push('err', 'Could not update effort', {
          body: err instanceof Error ? err.message : String(err),
        });
      });
  }
</script>

<ModelPicker {sessionId} {agent} {resolvedModel} onChanged={() => (modelRev += 1)} />
{#if sessionId && options.length > 0}
  <span class="effort" data-testid={TESTID.chatEffortSeg}>
    <Seg {options} value={shownValue} ariaLabel="Reasoning effort" onchange={onPick} />
  </span>
{/if}

<style>
  .effort {
    display: inline-flex;
    flex: none;
  }
</style>
