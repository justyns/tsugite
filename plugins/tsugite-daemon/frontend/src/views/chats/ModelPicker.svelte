<script lang="ts">
  // Conversation-header model control: a compact chip showing the session's
  // current model (GET /api/sessions/{id}/settings) that opens a filterable,
  // keyboard-navigable popover of GET /api/models. The popover groups models
  // under provider headers and shows each model's context window, price, and
  // vision/reasoning badges. Selecting a model PATCHes the session settings and
  // toasts the outcome. The models list (~80 entries) is fetched once and cached
  // at module scope so reopening / switching sessions is instant.
  import { tick, untrack } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { pwmIgnore } from '$lib/components/inputs/pwmIgnore';
  import { api } from '$lib/api/client';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { TESTID } from '$lib/testids';
  import { clipBoundaryLeft } from '$lib/dom';
  import { modelPickerRequest } from './modelPickerSignal.svelte';
  import {
    groupModelsByProvider,
    formatContext,
    formatPrice,
    type PickerModel,
  } from './modelGrouping';

  type Model = PickerModel;

  let {
    sessionId,
    agent,
    resolvedModel = null,
    onChanged,
  }: {
    sessionId: string | null;
    agent: string;
    /** The effective model when no per-session override is set (the agent/global
     *  default), so the chip can name what "default" actually runs. */
    resolvedModel?: string | null;
    /** Fired after a model change is persisted (the effort control refetches). */
    onChanged?: () => void;
  } = $props();

  // Module-level cache: the model registry is process-global, not per session.
  let modelCache: Model[] | null = null;
  let modelCachePromise: Promise<Model[]> | null = null;
  function loadModels(): Promise<Model[]> {
    if (modelCache) return Promise.resolve(modelCache);
    if (!modelCachePromise) {
      modelCachePromise = api
        .get<{ models: Model[] }>('/api/models')
        .then((r) => (modelCache = r.models))
        .catch((err) => {
          modelCachePromise = null;
          throw err;
        });
    }
    return modelCachePromise;
  }

  let current = $state<string | null>(null);
  let models = $state<Model[]>([]);
  let open = $state(false);
  let query = $state('');
  let selected = $state(0);
  let loading = $state(false);
  let root = $state<HTMLElement>();
  let inputEl = $state<HTMLInputElement>();
  let listEl = $state<HTMLElement>();
  let popEl = $state<HTMLElement>();
  // The popover is right-anchored to the chip; when that would cross the
  // nearest scroll/clip ancestor's left edge (the pane body cuts it under
  // the sessions rail - short chip labels pull it furthest left), it flips
  // to left-anchored.
  let alignLeft = $state(false);
  const listId = `chat-model-ls-${Math.random().toString(36).slice(2, 8)}`;

  // Track which session `current` reflects so a stale fetch can't clobber a newer
  // one. settingsRev advances on a cross-tab settings broadcast (/model, the
  // settings PATCH), so the chip refetches and updates live without a reopen.
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
      current = null; // a session switch, not a live update: clear before refetch
      open = false;
    }
    if (!id) return;
    sessions
      .getSettings(id)
      .then((s) => {
        if (settingsFor === id) current = s.model;
      })
      .catch(() => {});
  });

  // Short chip label: the model id without its provider prefix (tooltip keeps the
  // full id). Null = no override, so the agent/global default is in force - named
  // when the caller resolved it.
  const shortId = (id: string) => id.split(':').slice(-1)[0];
  const label = $derived(
    current
      ? shortId(current)
      : resolvedModel
        ? `default · ${shortId(resolvedModel)}`
        : 'default model',
  );

  const filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    if (!q) return models;
    return models.filter((m) => m.id.toLowerCase().includes(q));
  });

  // Grouped for rendering (provider headers), and flattened back into the exact
  // option order the keyboard navigation and `selected` index walk - the flat
  // order is the concatenation of the groups, so it crosses group boundaries
  // seamlessly and never falls out of sync with what's on screen.
  const groups = $derived(groupModelsByProvider(filtered));
  const options = $derived(groups.flatMap((g) => g.models));
  const provLabel = (m: Model) => m.provider || 'other';

  const optionId = (i: number) => `chat-model-opt-idx-${i}`;

  function toggle() {
    if (open) {
      open = false;
      return;
    }
    void openPicker();
  }

  async function openPicker() {
    open = true;
    query = '';
    selected = 0;
    alignLeft = false;
    if (models.length === 0) {
      loading = true;
      try {
        models = await loadModels();
      } catch (err) {
        toasts.push('err', 'Could not load models', {
          body: err instanceof Error ? err.message : String(err),
        });
        open = false;
        return;
      } finally {
        loading = false;
      }
    }
    // Land the selection on the current model so it's visible on open.
    const at = options.findIndex((m) => m.id === current);
    selected = at >= 0 ? at : 0;
    await tick();
    if (popEl && popEl.getBoundingClientRect().left < clipBoundaryLeft(popEl) + 8) alignLeft = true;
    inputEl?.focus();
    scrollSelectedIntoView();
  }

  // A `/model` slash pick (palette or the inline `/` menu) opens this header's
  // picker for the matching session instead of prefilling a text field. Reading
  // `.pending` subscribes; the open call is untracked so its own state reads
  // don't re-trigger this effect.
  $effect(() => {
    const req = modelPickerRequest.pending;
    if (!req || req.sessionId !== sessionId) return;
    modelPickerRequest.consume(sessionId ?? '');
    untrack(() => void openPicker());
  });

  function scrollSelectedIntoView() {
    listEl?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' });
  }

  function move(delta: number) {
    const max = options.length - 1;
    if (max < 0) return;
    selected = Math.min(Math.max(selected + delta, 0), max);
    scrollSelectedIntoView();
  }

  async function choose(model: Model) {
    open = false;
    if (!sessionId || model.id === current) return;
    const prev = current;
    current = model.id; // optimistic
    try {
      const res = await sessions.patchSettings(sessionId, { model: model.id });
      current = res.model;
      toasts.push('ok', `Model → ${shortId(model.id)}`);
      onChanged?.();
    } catch (err) {
      current = prev;
      toasts.push('err', 'Could not change model', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      move(1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      move(-1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const m = options[selected];
      if (m) void choose(m);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      open = false;
      root?.querySelector<HTMLElement>('button')?.focus();
    }
  }

  $effect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root && !root.contains(e.target as Node)) open = false;
    };
    window.addEventListener('mousedown', onDown);
    return () => window.removeEventListener('mousedown', onDown);
  });
</script>

{#if sessionId}
  <div class="model-anchor" bind:this={root}>
    <button
      type="button"
      class="model-chip"
      title={current
        ? `Model: ${current}`
        : resolvedModel
          ? `Using the agent default model: ${resolvedModel}`
          : 'Using the agent default model'}
      aria-haspopup="listbox"
      aria-expanded={open}
      data-testid={TESTID.chatModelTrigger}
      onclick={toggle}
    >
      <Icon name="sparkle" size={11} />
      <span class="lbl">{label}</span>
      <Icon name="chev-r" size={9} />
    </button>

    {#if open}
      <div
        class="model-pop"
        data-align={alignLeft ? 'left' : 'right'}
        data-testid={TESTID.chatModelPopover}
        bind:this={popEl}
      >
        <div class="mp-in">
          <Icon name="search" size={14} />
          <!-- svelte-ignore a11y_autofocus -->
          <input
            bind:this={inputEl}
            bind:value={query}
            type="search"
            placeholder="filter models…"
            {...pwmIgnore}
            spellcheck="false"
            role="combobox"
            aria-label="Filter models"
            aria-autocomplete="list"
            aria-controls={listId}
            aria-expanded={options.length > 0}
            aria-activedescendant={options.length > 0 ? optionId(selected) : undefined}
            data-testid={TESTID.chatModelSearch}
            oninput={() => (selected = 0)}
            onkeydown={onKeydown}
          />
        </div>
        <div class="mp-ls" id={listId} role="listbox" aria-label="Models" bind:this={listEl}>
          {#if loading}
            <div class="mp-empty">loading models…</div>
          {:else if options.length === 0}
            <div class="mp-empty">no models match “{query.trim()}”</div>
          {:else}
            {#each options as m, i (m.id)}
              {@const ctx = formatContext(m.context_window)}
              {@const price = formatPrice(m.input_cost_per_million, m.output_cost_per_million)}
              {#if i === 0 || provLabel(options[i - 1]!) !== provLabel(m)}
                <div class="mp-group">{provLabel(m)}</div>
              {/if}
              <button
                type="button"
                role="option"
                class="mp-it"
                class:is-sel={i === selected}
                class:is-current={m.id === current}
                id={optionId(i)}
                aria-selected={i === selected}
                data-testid={TESTID.chatModelOption(m.id)}
                onmousemove={() => (selected = i)}
                onclick={() => void choose(m)}
              >
                <span class="mp-main">
                  <span class="mp-id" title={m.id}>{shortId(m.id)}</span>
                  {#if m.supports_vision}
                    <span class="mp-badge" title="vision input" aria-label="vision input">
                      <Icon name="camera" size={10} />
                    </span>
                  {/if}
                  {#if m.supports_reasoning}
                    <span class="mp-badge" title="reasoning" aria-label="reasoning">
                      <Icon name="sparkle" size={10} />
                    </span>
                  {/if}
                  <span class="mp-spacer"></span>
                  {#if m.id === current}<Icon name="check" size={11} color="var(--acc)" />{/if}
                </span>
                {#if ctx || price}
                  <span class="mp-meta">
                    {#if ctx}<span class="mp-ctx">{ctx}</span>{/if}
                    {#if price}<span class="mp-price">{price}</span>{/if}
                  </span>
                {/if}
              </button>
            {/each}
          {/if}
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .model-anchor {
    position: relative;
    display: inline-flex;
    flex: none;
  }
  .model-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    max-width: 30ch;
    padding: 0 6px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    cursor: pointer;
  }
  .model-chip:hover {
    color: var(--acc);
    border-color: var(--bd1);
  }
  .model-chip .lbl {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .model-pop {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    z-index: 60;
    width: min(420px, 90vw);
    display: flex;
    flex-direction: column;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    overflow: hidden;
  }
  /* Chip near the viewport's left edge: right-anchoring would clip the popover
     off-screen, so it flips to hang rightward from the chip instead. */
  .model-pop[data-align='left'] {
    right: auto;
    left: 0;
  }
  .mp-in {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
  }
  .mp-in :global(.ic) {
    color: var(--tx3);
    flex: none;
  }
  .mp-in input {
    flex: 1;
    min-width: 0;
    background: none;
    border: 0;
    outline: none;
    color: var(--tx0);
    font: 400 var(--fs-sm) var(--font-ui);
  }
  .mp-in input::placeholder {
    color: var(--tx3);
  }
  /* type=search opts out of Chromium's password manager but pulls in the UA
     clear button; drop it so the field looks the same as before. */
  .mp-in input::-webkit-search-cancel-button,
  .mp-in input::-webkit-search-decoration {
    -webkit-appearance: none;
    appearance: none;
  }
  .mp-ls {
    overflow-y: auto;
    max-height: min(420px, 60vh);
    padding: 4px;
  }
  /* Provider header rows: sticky so the group stays named while its models
     scroll under it. Opaque popover background hides rows passing beneath. */
  .mp-group {
    position: sticky;
    top: 0;
    z-index: 1;
    padding: 7px 10px 3px;
    background: var(--bg3);
    color: var(--tx3);
    font: 600 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.04em;
    text-transform: lowercase;
  }
  .mp-it {
    display: flex;
    flex-direction: column;
    gap: 2px;
    width: 100%;
    padding: 6px 10px;
    border: 0;
    border-radius: var(--r-sm);
    background: transparent;
    color: var(--tx1);
    font: 500 var(--fs-xs) var(--font-mono);
    text-align: left;
    cursor: pointer;
  }
  .mp-main {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .mp-id {
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mp-spacer {
    flex: 1;
    min-width: 0;
  }
  .mp-badge {
    display: inline-flex;
    flex: none;
  }
  .mp-badge :global(.ic) {
    color: var(--tx3);
  }
  .mp-main :global(.ic) {
    flex: none;
  }
  .mp-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--tx3);
    font: 400 var(--fs-2xs) var(--font-ui);
  }
  .mp-price {
    font-family: var(--font-mono);
  }
  .mp-it.is-current {
    color: var(--tx0);
  }
  .mp-it.is-sel {
    background: var(--bg4);
    color: var(--tx0);
  }
  .mp-it.is-sel .mp-badge :global(.ic),
  .mp-it.is-sel .mp-meta {
    color: var(--tx2);
  }
  .mp-empty {
    padding: 16px 12px;
    text-align: center;
    color: var(--tx3);
    font: 400 var(--fs-sm) var(--font-ui);
  }
</style>
