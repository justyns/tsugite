<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import Modal from '$lib/components/overlays/Modal.svelte';
  import Field from '$lib/components/inputs/Field.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import ThemeSeg from './ThemeSeg.svelte';
  import ProvidersSettings, { type ProviderStatus } from './ProvidersSettings.svelte';
  import { messageOf } from '$lib/errors';
  import { api } from '$lib/api/client';
  import { auth } from '$lib/stores/auth.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import type { RuntimeInfo } from '$lib/stores/agentsMeta.svelte';
  import { agents, type Agent } from '$lib/stores/agents.svelte';
  import { welcomeDismissed } from '$lib/stores/welcomeDismissed.svelte';
  import { TESTID } from '$lib/testids';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  type Model = { id: string; provider?: string | null };
  type Setup = { complete: boolean; missing: string[]; fresh: boolean; workspace: string };

  const AGENT_COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'teal', 'pink'];

  let current = $state<RuntimeInfo | null>(null);
  let models = $state<Model[] | null>(null);
  let providers = $state<ProviderStatus[]>([]);
  let provider = $state('');
  let model = $state('');
  let userId = $state(auth.userId);
  let agent = $state<Agent | null>(null);
  let agentName = $state('');
  let agentColor = $state(AGENT_COLORS[0]);
  let agentModel = $state('');
  let about = $state('');
  let setup = $state<Setup | null>(null);
  let saving = $state(false);

  const loaded = $derived(current !== null && models !== null && agent !== null && setup !== null);
  const currentModel = $derived(current?.effective_model ?? null);
  const providerNames = $derived(providers.map((p) => p.name));
  const modelOptions = $derived.by(() => {
    const ids = (models ?? []).filter((m) => m.provider === provider).map((m) => m.id);
    if (currentModel?.startsWith(`${provider}:`) && !ids.includes(currentModel))
      ids.unshift(currentModel);
    return ids;
  });
  const modelHint = $derived.by(() => {
    if (currentModel && model !== currentModel)
      return `Replaces the current default, ${currentModel}.`;
    if (modelOptions.length || !provider) return 'Used by new chats.';
    return `${provider} lists no models.`;
  });

  async function load() {
    try {
      const [runtime, list, roster, status] = await Promise.all([
        current ?? api.get<RuntimeInfo>('/api/runtime'),
        api.get<{ models: Model[] }>('/api/models'),
        api.get<{ agents: Agent[] }>('/api/agents'),
        setup ?? api.get<Setup>('/api/setup'),
      ]);
      if (!open) return;
      current = runtime;
      models = list.models;
      userId = auth.userId;
      model = runtime.effective_model ?? '';
      provider = runtime.effective_model?.match(/^([^:]+):/)?.[1] ?? '';
      agent = roster.agents.find((a) => a.is_default) ?? null;
      agentName = agent?.name ?? '';
      agentColor = agent?.color ?? AGENT_COLORS[0];
      agentModel = agent?.model ?? '';
      setup = status;
    } catch (err) {
      toasts.push('err', 'Could not load the welcome', { body: messageOf(err) });
    }
  }

  // App mounts this only once the shell is authed. An incomplete workspace
  // opens it even if this browser dismissed it.
  onMount(async () => {
    try {
      const [runtime, status] = await Promise.all([
        api.get<RuntimeInfo>('/api/runtime'),
        api.get<Setup>('/api/setup'),
      ]);
      if (status.complete && (welcomeDismissed.enabled || runtime.effective_model)) return;
      current = runtime;
      setup = status;
      open = true;
    } catch (err) {
      toasts.push('err', 'Could not check for the welcome', { body: messageOf(err) });
    }
  });

  $effect(() => {
    if (open) untrack(load);
  });

  $effect(() => {
    if (provider) return;
    provider =
      (providers.find((p) => p.binary?.resolved || p.key?.set) ?? providers[0])?.name ?? '';
  });

  $effect(() => {
    if (!modelOptions.includes(model)) model = modelOptions[0] ?? '';
  });

  function dismiss() {
    welcomeDismissed.set(true);
    open = false;
    current = null;
    models = null;
    agent = null;
    setup = null;
  }

  async function finish() {
    const a = agent;
    if (!a) return;
    const name = agentName.trim() || a.name;
    const agentModelId = agentModel.trim() || null;
    const changes: Partial<Agent> = {};
    if (name !== a.name) changes.name = name;
    if (agentColor !== a.color) changes.color = agentColor;
    if (agentModelId !== a.model) changes.model = agentModelId;
    saving = true;
    try {
      if (model && model !== currentModel) await api.patch('/api/runtime/config', { model });
      if (Object.keys(changes).length) {
        await api.patch(`/api/agents/${encodeURIComponent(a.id)}`, changes);
        void agents.load();
      }
      if (setup?.fresh)
        await api.post('/api/setup', {
          user_name: userId.trim(),
          agent_name: name,
          about: about.trim(),
        });
      if (userId.trim()) auth.setUserId(userId.trim());
      dismiss();
    } catch (err) {
      toasts.push('err', 'Could not save the welcome', { body: messageOf(err) });
    } finally {
      saving = false;
    }
  }
</script>

<Modal {open} onclose={dismiss} title="Welcome to tsugite">
  <div class="welcome" data-testid={TESTID.welcome}>
    <p>Pick the model new chats use, name your agent, and set your user ID and a theme.</p>

    <Field id="welcome-provider" label="Provider">
      {#snippet children()}
        <Select id="welcome-provider" options={providerNames} bind:value={provider} />
      {/snippet}
    </Field>
    {#if open}<ProvidersSettings only={provider} bind:providers />{/if}
    <Field id="welcome-model" label="Default model" hint={modelHint}>
      {#snippet children()}
        <Select id="welcome-model" options={modelOptions} bind:value={model} />
      {/snippet}
    </Field>
    <section>
      <h4>Agent</h4>
      {#if setup}
        <p class="hint">
          {setup.fresh
            ? `Finish creates ${setup.workspace} from the workspace templates.`
            : `${setup.workspace} already has files. Finish leaves them alone.`}
        </p>
      {/if}
      <Field id="welcome-agent-name" label="Agent name">
        {#snippet children()}
          <Input bind:value={agentName} id="welcome-agent-name" />
        {/snippet}
      </Field>
      <Field id="welcome-agent-color" label="Color">
        {#snippet children()}
          <Select id="welcome-agent-color" options={AGENT_COLORS} bind:value={agentColor} />
        {/snippet}
      </Field>
      <Field id="welcome-agent-model" label="Agent model" hint="Blank follows the default model.">
        {#snippet children(describedBy)}
          <Input
            bind:value={agentModel}
            id="welcome-agent-model"
            ariaDescribedby={describedBy}
            placeholder={model}
            mono
          />
        {/snippet}
      </Field>
      {#if setup?.fresh}
        <Field id="welcome-agent-about" label="About" hint="Goes into IDENTITY.md.">
          {#snippet children(describedBy)}
            <Input bind:value={about} id="welcome-agent-about" ariaDescribedby={describedBy} />
          {/snippet}
        </Field>
      {/if}
    </section>
    <Field id="welcome-userid" label="User ID" hint="Sent with every request as user_id.">
      {#snippet children(describedBy)}
        <Input bind:value={userId} id="welcome-userid" ariaDescribedby={describedBy} mono />
      {/snippet}
    </Field>
    <section>
      <h4>Theme</h4>
      <ThemeSeg />
    </section>
  </div>
  {#snippet footer()}
    <Button onclick={dismiss}>Skip</Button>
    <Button variant="pri" disabled={!loaded} loading={saving} onclick={finish}>Finish</Button>
  {/snippet}
</Modal>

<style>
  .welcome {
    display: grid;
    gap: 14px;
    max-height: 65dvh;
    overflow-y: auto;
  }
  .welcome p {
    margin: 0;
  }
  .hint {
    font: 400 var(--fs-2xs) var(--font-ui);
    color: var(--tx3);
    overflow-wrap: anywhere;
  }
  section {
    display: grid;
    gap: 5px;
  }
  .welcome :global(.t-selectw select) {
    flex: 1;
  }
  h4 {
    margin: 0;
    font: 600 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--tx3);
  }
</style>
