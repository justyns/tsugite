<script module lang="ts">
  export interface ProviderStatus {
    name: string;
    binary: { default: string; configured: string | null; resolved: string | null } | null;
    key: { env: string; secret: string; set: boolean; source: 'secret' | 'env' | null } | null;
    error?: string;
  }
</script>

<script lang="ts">
  import Field from '$lib/components/inputs/Field.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import { api } from '$lib/api/client';
  import { messageOf } from '$lib/errors';
  import { secrets } from '$lib/stores/secrets.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { TESTID } from '$lib/testids';

  let {
    only,
    providers = $bindable([]),
  }: {
    only?: string;
    providers?: ProviderStatus[];
  } = $props();

  type TestResult =
    { ok: true; model: string; reply: string } | { ok: false; model: string | null; error: string };

  let loadError = $state<string | null>(null);
  let binaryDrafts = $state<Record<string, string>>({});
  let keyDrafts = $state<Record<string, string>>({});
  let busy = $state<Record<string, boolean>>({});
  let results = $state<Record<string, TestResult>>({});

  function replace(status: ProviderStatus) {
    providers = providers.map((p) => (p.name === status.name ? status : p));
    binaryDrafts[status.name] = status.binary?.configured ?? '';
  }

  // Re-reads one row, keeping unsaved drafts in the others.
  async function refresh(name: string) {
    const res = await api.get<{ providers: ProviderStatus[] }>('/api/providers');
    const status = res.providers.find((p) => p.name === name);
    if (status) replace(status);
    keyDrafts[name] = '';
  }

  async function load() {
    try {
      const res = await api.get<{ providers: ProviderStatus[] }>('/api/providers');
      providers = res.providers;
      binaryDrafts = Object.fromEntries(
        res.providers.map((p) => [p.name, p.binary?.configured ?? '']),
      );
      keyDrafts = Object.fromEntries(res.providers.map((p) => [p.name, '']));
      loadError = null;
    } catch (err) {
      loadError = messageOf(err);
    }
  }

  $effect(() => {
    load();
  });

  async function saveBinary(p: ProviderStatus) {
    const binary = (binaryDrafts[p.name] ?? '').trim();
    if (binary === (p.binary?.configured ?? '')) return;
    try {
      replace(
        await api.put<ProviderStatus>(`/api/providers/${encodeURIComponent(p.name)}`, { binary }),
      );
    } catch (err) {
      toasts.push('err', 'Binary path not saved', { body: messageOf(err) });
    }
  }

  async function saveKey(p: ProviderStatus) {
    const value = (keyDrafts[p.name] ?? '').trim();
    if (!p.key || !value) return;
    try {
      await secrets.set(p.key.secret, value);
      await refresh(p.name);
    } catch (err) {
      toasts.push('err', 'Key not saved', { body: messageOf(err) });
    }
  }

  async function clearKey(p: ProviderStatus) {
    if (!p.key) return;
    try {
      await secrets.remove(p.key.secret);
      await refresh(p.name);
    } catch (err) {
      toasts.push('err', 'Key not cleared', { body: messageOf(err) });
    }
  }

  async function test(p: ProviderStatus) {
    busy[p.name] = true;
    try {
      results[p.name] = await api.post<TestResult>(
        `/api/providers/${encodeURIComponent(p.name)}/test`,
        {},
      );
    } catch (err) {
      results[p.name] = { ok: false, model: null, error: messageOf(err) };
    } finally {
      busy[p.name] = false;
    }
  }

  function status(p: ProviderStatus): { text: string; ok: boolean } {
    if (p.error) return { text: 'failed to load', ok: false };
    if (p.binary)
      return p.binary.resolved ? { text: 'found', ok: true } : { text: 'not found', ok: false };
    if (p.key) return p.key.set ? { text: 'key set', ok: true } : { text: 'no key', ok: false };
    return { text: 'ready', ok: true };
  }

  function keyHint(key: NonNullable<ProviderStatus['key']>): string {
    if (key.source === 'secret') return `Stored as secret ${key.secret}.`;
    if (key.source === 'env') return `Read from ${key.env}. A stored key takes precedence.`;
    return `Not set. Save a key here or set ${key.env}.`;
  }
</script>

{#if only === undefined}<p class="sub">Saved changes apply without restarting the daemon.</p>{/if}
{#if loadError}
  <p class="err" role="alert">{loadError}</p>
{/if}
<div class="provs">
  {#each only === undefined ? providers : providers.filter((p) => p.name === only) as p (p.name)}
    {@const st = status(p)}
    {@const result = results[p.name]}
    <div class="prov" data-testid={TESTID.settingsProvider(p.name)}>
      <div class="prov-head">
        <span class="nm">{p.name}</span>
        <span class="st" data-ok={st.ok}>{st.text}</span>
        <span class="grow"></span>
        {#if !p.error}
          <Button
            size="sm"
            loading={busy[p.name]}
            onclick={() => test(p)}
            aria-label={`Test ${p.name}`}
          >
            Test
          </Button>
        {/if}
      </div>
      {#if p.error}
        <p class="err">{p.error}</p>
      {/if}
      {#if p.binary}
        {@const id = `provider-binary-${p.name}`}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div onchange={() => saveBinary(p)}>
          <Field
            {id}
            label="Binary path"
            hint={p.binary.resolved
              ? `Runs ${p.binary.resolved}`
              : `Not found: ${p.binary.configured ?? `${p.binary.default} on PATH`}`}
          >
            {#snippet children(describedBy)}
              <Input
                bind:value={binaryDrafts[p.name]}
                {id}
                ariaDescribedby={describedBy}
                placeholder={`${p.binary!.default} (search PATH)`}
                mono
              />
            {/snippet}
          </Field>
        </div>
      {/if}
      {#if p.key}
        {@const id = `provider-key-${p.name}`}
        <Field {id} label={p.key.set ? 'Replace API key' : 'API key'} hint={keyHint(p.key)}>
          {#snippet children(describedBy)}
            <div class="key-row">
              <Input
                type="password"
                bind:value={keyDrafts[p.name]}
                {id}
                ariaDescribedby={describedBy}
                placeholder={p.key!.set ? '••••••••' : 'paste a key'}
                mono
              />
              <Button
                size="sm"
                variant="pri"
                disabled={!keyDrafts[p.name]?.trim()}
                onclick={() => saveKey(p)}
              >
                Save
              </Button>
              {#if p.key!.source === 'secret'}
                <Button size="sm" variant="danger" onclick={() => clearKey(p)}>Clear</Button>
              {/if}
            </div>
          {/snippet}
        </Field>
      {/if}
      {#if result}
        <p class="result" data-ok={result.ok} role="status">
          {#if result.ok}
            OK{result.model ? ` · ${result.model}` : ''}
          {:else}
            {result.model ? `${result.model}: ` : ''}{result.error}
          {/if}
        </p>
      {/if}
    </div>
  {/each}
</div>

<style>
  .sub {
    margin: 0 0 8px;
    font: 400 var(--fs-xs) / 1.45 var(--font-ui);
    color: var(--tx3);
  }
  .provs {
    display: grid;
    gap: 8px;
  }
  .prov {
    display: grid;
    gap: 8px;
    padding: 9px 10px;
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    background: var(--bg1);
    min-width: 0;
  }
  .prov-head {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .nm {
    font: 600 var(--fs-sm) var(--font-mono);
    color: var(--tx1);
  }
  .st {
    font: 500 var(--fs-xs) var(--font-ui);
    color: var(--st-ok);
  }
  .st[data-ok='false'] {
    color: var(--st-warn);
  }
  .grow {
    flex: 1;
  }
  .key-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .err,
  .result {
    margin: 0;
    font: 400 var(--fs-xs) / 1.45 var(--font-mono);
    overflow-wrap: anywhere;
  }
  .err,
  .result[data-ok='false'] {
    color: var(--st-err);
  }
  .result[data-ok='true'] {
    color: var(--st-ok);
  }
</style>
