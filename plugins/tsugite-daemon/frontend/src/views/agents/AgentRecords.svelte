<script lang="ts">
  import { onMount } from 'svelte';
  import { rowKeydown } from '$lib/actions/rowKeydown';
  import { TESTID } from '$lib/testids';
  import { agents, AGENT_COLORS, validateAgentId, type Agent } from '$lib/stores/agents.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { navigate } from '$lib/router.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Badge from '$lib/components/buttons/Badge.svelte';
  import Field from '$lib/components/inputs/Field.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import AgentDot from '$lib/components/rows/AgentDot.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';

  const DEFAULT_EFFORT = 'default';
  const NO_COLOR = 'none';
  const AGENT_TEMPLATE = "(default agent's)";

  let selectedId = $state<string | null>(null);
  let creating = $state(false);
  let id = $state('');
  let name = $state('');
  let template = $state('');
  let model = $state('');
  let effort = $state(DEFAULT_EFFORT);
  let color = $state(NO_COLOR);
  let identity = $state('');
  let saving = $state(false);
  let formError = $state<string | null>(null);

  const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

  const current = $derived(agents.list.find((a) => a.id === selectedId) ?? null);
  const idError = $derived(id === '' ? null : validateAgentId(id));
  const templateNames = $derived(agentsMeta.agentFiles.map((f) => f.name));
  const templateOptions = $derived(
    creating
      ? [AGENT_TEMPLATE, ...templateNames]
      : [...new Set([template, ...templateNames].filter((t) => t !== ''))],
  );
  const effortOptions = $derived([...new Set([DEFAULT_EFFORT, 'low', 'medium', 'high', effort])]);
  const templateLocked = $derived(current?.is_default ?? false);

  const nullable = (v: string, none: string) => (v === none || v.trim() === '' ? null : v.trim());

  const dirty = $derived(
    current !== null &&
      (name !== current.name ||
        (!templateLocked && template !== current.template) ||
        nullable(model, '') !== current.model ||
        nullable(effort, DEFAULT_EFFORT) !== current.reasoning_effort ||
        nullable(color, NO_COLOR) !== current.color),
  );
  const canSubmit = $derived(
    !saving && (creating ? id !== '' && idError === null : dirty && name.trim() !== ''),
  );

  onMount(() => {
    void agents.load();
    void agentsMeta.loadAgentFiles();
  });

  $effect(() => {
    const first = agents.list[0];
    if (selectedId === null && !creating && first) select(first);
  });

  function fill(a: Agent): void {
    id = a.id;
    name = a.name;
    template = a.template;
    model = a.model ?? '';
    effort = a.reasoning_effort ?? DEFAULT_EFFORT;
    color = a.color ?? NO_COLOR;
  }

  function select(a: Agent): void {
    selectedId = a.id;
    creating = false;
    formError = null;
    fill(a);
  }

  function startCreate(): void {
    selectedId = null;
    creating = true;
    formError = null;
    id = '';
    name = '';
    template = AGENT_TEMPLATE;
    model = '';
    effort = DEFAULT_EFFORT;
    color = NO_COLOR;
    identity = '';
  }

  async function submit(): Promise<void> {
    if (!canSubmit) return;
    saving = true;
    formError = null;
    try {
      if (creating) {
        const created = await agents.create({
          id,
          ...(name.trim() ? { name: name.trim() } : {}),
          ...(template !== AGENT_TEMPLATE ? { template } : {}),
          model: nullable(model, ''),
          reasoning_effort: nullable(effort, DEFAULT_EFFORT),
          color: nullable(color, NO_COLOR),
          ...(identity.trim() ? { identity: identity.trim() } : {}),
        });
        select(created);
        toasts.push('ok', 'Agent created', { body: created.id });
      } else if (current) {
        const target = current;
        const updated = await agents.update(target.id, {
          name: name.trim(),
          ...(templateLocked ? {} : { template }),
          model: nullable(model, ''),
          reasoning_effort: nullable(effort, DEFAULT_EFFORT),
          color: nullable(color, NO_COLOR),
        });
        if (selectedId === target.id) fill(updated);
        toasts.push('ok', 'Saved', { body: `${updated.id} updated` });
      }
    } catch (e) {
      formError = msg(e);
    } finally {
      saving = false;
    }
  }

  async function openMain(agentId: string): Promise<void> {
    try {
      const sessionId = await agents.openMain(agentId);
      await sessions.load();
      navigate('chats', { sessionId });
    } catch (e) {
      toasts.push('err', 'Could not open main chat', { body: msg(e) });
    }
  }
</script>

<div class="agents-shell">
  <div class="agents-list">
    <div class="sidebar-hd">
      <strong class="hd-title">Agents</strong>
      <span class="cnt">{agents.list.length}</span>
      <span class="grow"></span>
      <Button size="sm" data-testid={TESTID.agentRecordNew} onclick={startCreate}>New</Button>
    </div>

    {#if agents.error && agents.list.length === 0}
      <div class="pad">
        <PaneState kind="error" title="Couldn't load agents">
          {#snippet hint()}<span>{agents.error}</span>{/snippet}
          {#snippet actions()}
            <Button size="sm" onclick={() => void agents.load()}>Retry</Button>
          {/snippet}
        </PaneState>
      </div>
    {:else}
      <div class="roster" role="listbox" aria-label="Agents" tabindex="-1">
        {#each agents.list as a, i (a.id)}
          {@const isSel = a.id === selectedId}
          <button
            type="button"
            class="ag-row"
            class:is-selected={isSel}
            role="option"
            aria-selected={isSel}
            tabindex={isSel || (creating && i === 0) ? 0 : -1}
            data-rowidx={i}
            data-testid={TESTID.agentRecordRow(a.id)}
            onclick={() => select(a)}
            onkeydown={(e) => rowKeydown(e, i, agents.list, select)}
          >
            <AgentDot color={a.color} />
            <span class="ag-main">
              <span class="ag-name">{a.name}</span>
              <span class="ag-sub">{a.id}{a.model ? ` · ${a.model}` : ''}</span>
            </span>
            {#if a.is_default}<Badge label="default agent">default</Badge>{/if}
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="agents-ed">
    {#if selectedId === null && !creating}
      <div class="pad">
        <PaneState kind="empty" title="Select an agent">
          {#snippet hint()}<span>Pick an agent to edit it, or create a new one.</span>{/snippet}
        </PaneState>
      </div>
    {:else}
      <div class="wk-toolbar">
        <span class="wk-crumb">agents / <b>{creating ? 'new agent' : (current?.id ?? '')}</b></span>
        <span class="grow"></span>
        {#if current}
          <Button
            size="sm"
            data-testid={TESTID.agentOpenMain}
            onclick={() => void openMain(current.id)}
          >
            Open main chat
          </Button>
        {/if}
        <Button
          size="sm"
          variant="pri"
          data-testid={TESTID.agentRecordSave}
          disabled={!canSubmit}
          loading={saving}
          onclick={() => void submit()}
        >
          {creating ? 'Create' : 'Save'}
        </Button>
      </div>

      <form
        class="rec-form"
        data-testid={TESTID.agentRecordForm}
        onsubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        {#if formError}
          <div class="form-err" role="alert" data-testid={TESTID.agentRecordError}>
            {formError}
          </div>
        {/if}

        <Field
          id="agent-id"
          label="id"
          error={creating && idError ? idError : undefined}
          hint={creating ? 'Fixed once created.' : undefined}
        >
          {#snippet children(describedBy)}
            <Input
              id="agent-id"
              bind:value={id}
              disabled={!creating}
              invalid={creating && idError !== null}
              ariaDescribedby={describedBy}
              mono
            />
          {/snippet}
        </Field>

        <Field id="agent-name" label="name">
          {#snippet children(describedBy)}
            <Input
              id="agent-name"
              bind:value={name}
              placeholder={creating ? id : undefined}
              ariaDescribedby={describedBy}
            />
          {/snippet}
        </Field>

        <Field
          id="agent-template"
          label="template"
          hint={templateLocked
            ? "The default agent's template is set in the daemon config."
            : undefined}
        >
          {#snippet children(describedBy)}
            {#if templateLocked}
              <Input
                id="agent-template"
                value={template}
                disabled
                mono
                ariaDescribedby={describedBy}
              />
            {:else}
              <Select id="agent-template" options={templateOptions} bind:value={template} />
            {/if}
          {/snippet}
        </Field>

        <Field id="agent-model" label="model" hint="Empty uses the daemon default.">
          {#snippet children(describedBy)}
            <Input
              id="agent-model"
              bind:value={model}
              placeholder="provider:model"
              ariaDescribedby={describedBy}
              mono
            />
          {/snippet}
        </Field>

        <div class="t-field">
          <span class="fl">reasoning effort</span>
          <Seg options={effortOptions} bind:value={effort} ariaLabel="Reasoning effort" />
        </div>

        <Field id="agent-color" label="color">
          {#snippet children()}
            <span class="color-pick">
              <Select id="agent-color" options={[NO_COLOR, ...AGENT_COLORS]} bind:value={color} />
              <AgentDot color={color === NO_COLOR ? null : color} />
            </span>
          {/snippet}
        </Field>

        {#if creating}
          <Field
            id="agent-identity"
            label="identity"
            hint="Optional. Written into the new workspace's IDENTITY.md."
          >
            {#snippet children(describedBy)}
              <textarea
                id="agent-identity"
                class="identity"
                rows="4"
                bind:value={identity}
                aria-describedby={describedBy}></textarea>
            {/snippet}
          </Field>
        {:else if current}
          <div class="t-field">
            <span class="fl">workspace</span>
            <code class="ws">{current.workspace}</code>
          </div>
        {/if}
      </form>
    {/if}
  </div>
</div>

<style>
  .ag-sub {
    font-family: var(--font-mono);
  }
  .rec-form {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: grid;
    align-content: start;
    gap: 14px;
    padding: 14px 16px;
    max-width: 560px;
  }
  .t-field {
    display: grid;
    gap: 5px;
  }
  .fl {
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  .color-pick {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  .ws {
    font: 400 var(--fs-xs) var(--font-mono);
    color: var(--tx2);
    overflow-wrap: anywhere;
  }
  .identity {
    width: 100%;
    resize: vertical;
    font: 400 var(--fs-md) var(--font-ui);
    padding: 6px 9px;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    color: var(--tx0);
  }
  .identity:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
  .form-err {
    font: 400 var(--fs-xs) var(--font-ui);
    color: var(--st-err);
    background: color-mix(in oklab, var(--st-err) 10%, transparent);
    border: 1px solid color-mix(in oklab, var(--st-err) 40%, transparent);
    border-radius: var(--r-md);
    padding: 7px 10px;
  }
</style>
