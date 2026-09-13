<script lang="ts">
  // Header model + effort pair: the model chip/popover plus the reasoning-effort
  // seg beside it. A phone header has no room for the seg. The session menu
  // renders it there, with `showEffort` cleared here.
  // GET /api/chat/effort-levels?session_id= resolves the model behind the chip's
  // "default" label.
  import { sessions } from '$lib/stores/sessions.svelte';
  import { fetchEffortLevels } from './effortLevels';
  import ModelPicker from './ModelPicker.svelte';
  import EffortSeg from './EffortSeg.svelte';

  let {
    sessionId,
    showEffort = true,
  }: {
    sessionId: string | null;
    /** Render the effort seg beside the chip. */
    showEffort?: boolean;
  } = $props();

  let resolvedModel = $state<string | null>(null);
  // Bumped after a model change so the seg refetches its levels.
  let modelRev = $state(0);

  let modelKey = '';
  $effect(() => {
    const id = sessionId;
    const rev = id ? (sessions.settingsRev[id] ?? 0) : 0;
    const key = `${id ?? ''}#${modelRev}#${rev}`;
    if (key === modelKey) return;
    modelKey = key;
    resolvedModel = null;
    if (!id) return;
    fetchEffortLevels(id)
      .then((res) => {
        if (modelKey !== key) return;
        resolvedModel = res.model;
      })
      .catch(() => {});
  });
</script>

<ModelPicker {sessionId} {resolvedModel} onChanged={() => (modelRev += 1)} />
{#if showEffort}
  <EffortSeg {sessionId} {modelRev} />
{/if}
