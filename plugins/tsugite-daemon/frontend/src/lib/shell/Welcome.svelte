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
  import { welcomeDismissed } from '$lib/stores/welcomeDismissed.svelte';
  import { TESTID } from '$lib/testids';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  type Model = { id: string; provider?: string | null };

  let current = $state<RuntimeInfo | null>(null);
  let models = $state<Model[] | null>(null);
  let providers = $state<ProviderStatus[]>([]);
  let provider = $state('');
  let model = $state('');
  let userId = $state(auth.userId);
  let saving = $state(false);

  const loaded = $derived(current !== null && models !== null);
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
      const [runtime, list] = await Promise.all([
        current ?? api.get<RuntimeInfo>('/api/runtime'),
        api.get<{ models: Model[] }>('/api/models'),
      ]);
      if (!open) return;
      current = runtime;
      models = list.models;
      userId = auth.userId;
      model = runtime.effective_model ?? '';
      provider = runtime.effective_model?.match(/^([^:]+):/)?.[1] ?? '';
    } catch (err) {
      toasts.push('err', 'Could not load the welcome', { body: messageOf(err) });
    }
  }

  // App mounts this only once the shell is authed.
  onMount(async () => {
    if (welcomeDismissed.enabled) return;
    try {
      const runtime = await api.get<RuntimeInfo>('/api/runtime');
      if (runtime.effective_model) return;
      current = runtime;
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
  }

  async function finish() {
    saving = true;
    try {
      if (model && model !== currentModel) await api.patch('/api/runtime/config', { model });
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
    <p>Pick the model new chats use, your user ID and a theme.</p>

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
