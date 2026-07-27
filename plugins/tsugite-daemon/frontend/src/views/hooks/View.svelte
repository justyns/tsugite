<script lang="ts">
  // Hooks: per-agent lifecycle hook rules - the daemon fires each agent's
  // workspace `.tsugite/hooks.yaml` (post_tool, pre_message, pre_compact, ...).
  // The loader reads that file fresh on every firing, so a save here applies on
  // the agent's next turn with no restart. The editor is the raw YAML (the
  // schema is too expressive for a form - jinja match expressions, env maps);
  // the parsed rule cards above it are the server's read-back of the same file.
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import { api } from '$lib/api/client';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';

  interface HookRuleSummary {
    name: string | null;
    type: 'shell' | 'agent' | 'python';
    run: string | null;
    agent: string | null;
    tools: string[];
    match: string | null;
    wait: boolean;
    capture_as: string | null;
    only_interactive: boolean;
  }
  interface HooksPayload {
    path: string;
    exists: boolean;
    raw: string;
    phases: Record<string, HookRuleSummary[]> | null;
    error: string | null;
  }

  let agentPick = $state('');
  const agents = $derived(agentsMeta.agents.map((a) => a.name));
  const agent = $derived(agentPick || agents[0] || '');

  let loading = $state(false);
  let loadError = $state<string | null>(null);
  let payload = $state<HooksPayload | null>(null);
  let draft = $state('');
  let saving = $state(false);
  let saveError = $state<string | null>(null);
  const dirty = $derived(payload != null && draft !== payload.raw);

  const TEMPLATE = `hooks:
  post_tool:
    - name: example
      tools: [write_file]
      run: ["echo", "{{ path }}"]
`;

  $effect(() => {
    if (agentsMeta.agents.length === 0) void agentsMeta.load();
  });

  let loadedFor: string | null = null;
  $effect(() => {
    if (agent && agent !== loadedFor) {
      loadedFor = agent;
      void load(agent);
    }
  });

  async function load(name: string) {
    loading = true;
    loadError = null;
    saveError = null;
    try {
      const res = await api.get<HooksPayload>(`/api/agents/${encodeURIComponent(name)}/hooks`);
      payload = res;
      draft = res.raw;
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
      payload = null;
    } finally {
      loading = false;
    }
  }

  async function save() {
    if (!agent) return;
    saving = true;
    saveError = null;
    try {
      const res = await api.put<HooksPayload>(`/api/agents/${encodeURIComponent(agent)}/hooks`, {
        raw: draft,
      });
      payload = res;
      draft = res.raw;
      toasts.push('ok', 'Hooks saved', { body: 'applies on the next agent turn' });
    } catch (err) {
      saveError = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  function startFromTemplate() {
    draft = TEMPLATE;
  }

  const phaseEntries = $derived(payload?.phases ? Object.entries(payload.phases) : []);
  const ruleCount = $derived(phaseEntries.reduce((n, [, rules]) => n + rules.length, 0));
</script>

<div class="hooks-view">
  <header class="vw-hd">
    <Icon name="fork" size={14} />
    <h2>hooks</h2>
    {#if agents.length > 1}
      <Select
        options={agents}
        value={agent}
        ariaLabel="Hooks agent"
        onchange={(a) => (agentPick = a)}
      />
    {/if}
    <span class="hint">
      {payload?.path ?? ''}{ruleCount ? ` · ${ruleCount} rule${ruleCount === 1 ? '' : 's'}` : ''}
    </span>
    <div class="grow"></div>
    <Button variant="pri" size="sm" disabled={!dirty || saving} loading={saving} onclick={save}>
      {#snippet icon()}<Icon name="check" />{/snippet}
      Save
    </Button>
  </header>

  <div class="vw-body">
    {#if loading && !payload}
      <div class="pane-pad"><PaneState kind="loading" lines={4} /></div>
    {:else if loadError}
      <div class="pane-pad">
        <PaneState kind="error" title="Couldn't load hooks">
          {#snippet icon()}<Icon name="alert" />{/snippet}
          {#snippet hint()}{loadError}{/snippet}
          {#snippet actions()}
            <Button size="sm" onclick={() => load(agent)}>
              {#snippet icon()}<Icon name="retry" />{/snippet}
              Retry
            </Button>
          {/snippet}
        </PaneState>
      </div>
    {:else if payload}
      {#if payload.error}
        <div class="parse-err" role="alert">
          <Icon name="alert" size={13} />
          <span>{payload.error}</span>
        </div>
      {:else if phaseEntries.length > 0}
        <div class="phases">
          {#each phaseEntries as [phase, rules] (phase)}
            <section class="phase">
              <h3>{phase}</h3>
              {#each rules as rule, i (i)}
                <div class="rule">
                  <span class="t-type" data-k={rule.type}>{rule.type}</span>
                  {#if rule.name}<b class="nm">{rule.name}</b>{/if}
                  <code class="cmd">{rule.run ?? (rule.agent ? `agent: ${rule.agent}` : '')}</code>
                  <span class="meta">
                    {#if rule.tools.length}<span class="chip">tools: {rule.tools.join(', ')}</span
                      >{/if}
                    {#if rule.match}<span class="chip" title={rule.match}>match</span>{/if}
                    {#if rule.capture_as}<span class="chip">→ {rule.capture_as}</span>{/if}
                    {#if rule.wait}<span class="chip">wait</span>{/if}
                    {#if rule.only_interactive}<span class="chip">interactive-only</span>{/if}
                  </span>
                </div>
              {/each}
            </section>
          {/each}
        </div>
      {:else}
        <div class="empty-note">
          <p>
            No hooks configured for <b>{agent}</b>. Hooks run shell commands or agents on lifecycle
            events (after tool calls, before messages, around compaction, …).
          </p>
          <Button size="sm" onclick={startFromTemplate}>
            {#snippet icon()}<Icon name="plus" />{/snippet}
            Start from a template
          </Button>
        </div>
      {/if}

      {#if saveError}
        <div class="parse-err" role="alert">
          <Icon name="alert" size={13} />
          <span>{saveError}</span>
        </div>
      {/if}

      <div class="editor">
        <div class="ed-hd">
          <span>{payload.exists ? '.tsugite/hooks.yaml' : '.tsugite/hooks.yaml (new file)'}</span>
          <span class="ed-note">validated on save · applies on the next agent turn</span>
        </div>
        <textarea
          bind:value={draft}
          spellcheck="false"
          aria-label="Hooks YAML"
          placeholder={'hooks:\n  post_tool:\n    - run: ["echo", "hi"]'}></textarea>
      </div>
    {/if}
  </div>
</div>

<style>
  .hooks-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    background: var(--bg0);
  }
  .vw-hd {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 14px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
  }
  .vw-hd h2 {
    margin: 0;
    font: 600 var(--fs-lg) var(--font-ui);
    color: var(--tx0);
  }
  .vw-hd :global(.ic) {
    color: var(--tx3);
  }
  .hint {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .vw-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 14px;
  }
  .pane-pad {
    padding: 10px;
  }
  .parse-err {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 11px;
    border: 1px solid color-mix(in oklab, var(--st-err) 32%, transparent);
    background: color-mix(in oklab, var(--st-err) 10%, transparent);
    border-radius: var(--r-md);
    color: var(--st-err);
    font: 500 var(--fs-sm) var(--font-mono);
    flex: none;
  }
  .phases {
    display: grid;
    gap: 10px;
    flex: none;
  }
  .phase {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    background: var(--bg1);
    padding: 8px 10px;
  }
  .phase h3 {
    margin: 0 0 6px;
    font: 600 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .rule {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    min-width: 0;
    flex-wrap: wrap;
    font: 400 var(--fs-sm) var(--font-ui);
  }
  .rule .nm {
    color: var(--tx0);
    font-weight: 600;
  }
  .rule .cmd {
    font: 400 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    background: var(--bg0);
    border: 1px solid var(--bd0);
    border-radius: var(--r-sm);
    padding: 1px 6px;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 52ch;
  }
  .rule .meta {
    display: inline-flex;
    gap: 5px;
    flex-wrap: wrap;
  }
  .chip {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx2);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 0 6px;
    line-height: 18px;
    white-space: nowrap;
  }
  /* t-type: same badge vocabulary as the session-type chip. */
  .t-type {
    display: inline-block;
    font: 600 var(--fs-2xs) / 1.5 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0 4px;
    border-radius: 3px;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 15%, transparent);
    flex: none;
  }
  .t-type[data-k='shell'] {
    --c: var(--st-ok);
  }
  .t-type[data-k='agent'] {
    --c: var(--acc);
  }
  .t-type[data-k='python'] {
    --c: var(--st-warn);
  }
  .empty-note {
    display: grid;
    gap: 10px;
    justify-items: start;
    border: 1px dashed var(--bd1);
    border-radius: var(--r-md);
    padding: 14px;
    color: var(--tx2);
    font: 400 var(--fs-sm) / 1.5 var(--font-ui);
    flex: none;
  }
  .empty-note p {
    margin: 0;
    max-width: 64ch;
  }
  .editor {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow: hidden;
    flex: 1;
    min-height: 220px;
  }
  .ed-hd {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 5px 10px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg1);
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx2);
    flex: none;
  }
  .ed-note {
    color: var(--tx3);
  }
  .editor textarea {
    flex: 1;
    min-height: 200px;
    resize: none;
    border: 0;
    outline: none;
    background: var(--bg0);
    color: var(--tx1);
    font: 400 var(--fs-sm) / 1.6 var(--font-mono);
    padding: 10px 12px;
    tab-size: 2;
  }
</style>
