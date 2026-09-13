<script lang="ts">
  // Reasoning-effort control for one session: a seg of the levels the session's
  // resolved model declares (GET /api/chat/effort-levels?session_id=), bound to
  // the persisted per-session setting (GET/PATCH /api/sessions/{id}/settings),
  // not a per-message override. A model that declares no levels renders nothing.
  import Seg from '$lib/components/inputs/Seg.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { fetchEffortLevels } from './effortLevels';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { TESTID } from '$lib/testids';

  let {
    sessionId,
    modelRev = 0,
  }: {
    sessionId: string | null;
    /** Bumped after a model change. Levels are model-dependent. */
    modelRev?: number;
  } = $props();

  // Seg labels stay compact; everything not listed here shows verbatim.
  const SHORT: Record<string, string> = { minimal: 'min', medium: 'med' };
  const display = (level: string) => SHORT[level] ?? level;

  let levels = $state<string[] | null>(null);

  // The seg's bound value is a display label; `persisted` mirrors the server's
  // level word so the loader's write doesn't look like a user edit.
  let segValue = $state('');
  let persisted = $state('');

  let levelsKey = '';
  $effect(() => {
    const id = sessionId;
    // settingsRev advances on a cross-tab settings broadcast, including a model change.
    const rev = id ? (sessions.settingsRev[id] ?? 0) : 0;
    const key = `${id ?? ''}#${modelRev}#${rev}`;
    if (key === levelsKey) return;
    levelsKey = key;
    levels = null;
    if (!id) return;
    fetchEffortLevels(id)
      .then((res) => {
        if (levelsKey !== key) return;
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
      segValue = '';
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
