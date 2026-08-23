<script lang="ts">
  // Webhooks: inbound HTTP triggers that drop a JSON envelope into an agent's
  // inbox/webhooks/ workspace folder. Real backend model (webhook_store.py)
  // is just {token, agent, source, created_at} - no enable/disable flag, no
  // signing secret, and no delivery-history endpoint (deliveries land as
  // files on disk with nothing to list them back over HTTP). This view
  // reflects that honestly rather than a fuller shape:
  //  - no enable switch: there is no field to toggle. Delete is the only
  //    real lifecycle action a webhook has.
  //  - "secret field (write-only display)" maps onto the real `token`, since
  //    that's the actual bearer credential embedded in the delivery URL.
  //  - "last delivery" is scoped honestly to *this session*: test-fire makes
  //    a real POST to the real public endpoint, and the row/log reflect that
  //    real round trip - not a fabricated history the backend can't provide.
  import { onMount } from 'svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import Modal from '$lib/components/overlays/Modal.svelte';
  import Field from '$lib/components/inputs/Field.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';
  import { webhooks, type Webhook } from '$lib/stores/webhooks.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { api, type ApiError } from '$lib/api/client';
  import WebhookRow, { type TestResult } from './WebhookRow.svelte';
  import { buildTestPayload, deliveryPath, isValidSource } from './logic';
  import { formatAgo } from '$lib/relativeTime';

  const TICK_MS = 30_000;
  const MAX_LOG_LINES = 20;

  // Unique per component instance - the mux can dock more than one Webhooks
  // tab at once, and hardcoded ids would collide across instances.
  const uid = $props.id();
  const agentFieldId = `${uid}-agent`;
  const sourceFieldId = `${uid}-source`;
  const createFormId = `${uid}-create-form`;
  const headingId = `${uid}-h`;

  interface LogLine extends TestResult {
    id: number;
    token: string;
    source: string;
    agent: string;
  }

  let now = $state(Date.now());
  let showCreate = $state(false);
  let createAgent = $state('');
  let createSource = $state('');
  let createError = $state<string | null>(null);
  let creating = $state(false);
  let pendingDelete = $state<Webhook | null>(null);
  let deleting = $state(false);
  let testLog = $state<LogLine[]>([]);
  let lastTestByToken = $state<Record<string, TestResult>>({});
  let logIdSeq = 0;

  onMount(() => {
    void webhooks.load();
    void agentsMeta.load();
  });

  $effect(() => {
    const id = setInterval(() => (now = Date.now()), TICK_MS);
    return () => clearInterval(id);
  });

  const agentOptions = $derived(agentsMeta.agents.map((a) => a.name));
  const sorted = $derived(
    [...webhooks.list].sort((a, b) => b.created_at.localeCompare(a.created_at)),
  );
  const failingCount = $derived(
    sorted.filter((w) => {
      const t = lastTestByToken[w.token];
      return t != null && !t.ok;
    }).length,
  );
  const countLabel = $derived(
    `${sorted.length} webhook${sorted.length === 1 ? '' : 's'}` +
      (failingCount > 0 ? ` · ${failingCount} failing this session` : ''),
  );

  function openCreate(): void {
    createAgent = agentOptions[0] ?? '';
    createSource = '';
    createError = null;
    showCreate = true;
  }

  async function submitCreate(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    const source = createSource.trim();
    if (!createAgent) {
      createError = 'Select an agent.';
      return;
    }
    if (!isValidSource(source)) {
      createError = 'Source must be 1-64 chars: letters, digits, dot, underscore, or dash.';
      return;
    }
    creating = true;
    createError = null;
    try {
      await webhooks.create({ agent: createAgent, source });
      showCreate = false;
      toasts.push('ok', `Webhook created for ${createAgent}`, { body: `source: ${source}` });
    } catch (err) {
      createError = err instanceof Error ? err.message : String(err);
    } finally {
      creating = false;
    }
  }

  async function doDelete(): Promise<void> {
    const target = pendingDelete;
    if (!target) return;
    deleting = true;
    try {
      await webhooks.remove(target.token);
      toasts.push('ok', 'Webhook deleted', { body: target.source });
      pendingDelete = null;
    } catch (err) {
      toasts.push('err', 'Delete failed', {
        body: err instanceof Error ? err.message : String(err),
      });
    } finally {
      deleting = false;
    }
  }

  function recordTest(webhook: Webhook, result: TestResult): void {
    lastTestByToken = { ...lastTestByToken, [webhook.token]: result };
    const line: LogLine = {
      ...result,
      id: ++logIdSeq,
      token: webhook.token,
      source: webhook.source,
      agent: webhook.agent,
    };
    testLog = [line, ...testLog].slice(0, MAX_LOG_LINES);
  }

  async function testFire(webhook: Webhook): Promise<void> {
    const payload = buildTestPayload(webhook.source);
    try {
      const res = await api.post<{ status: string; file: string }>(
        deliveryPath(webhook.token),
        payload,
      );
      recordTest(webhook, {
        ok: true,
        status: 202,
        at: Date.now(),
        detail: `saved as ${res.file}`,
      });
      toasts.push('ok', `Test delivery sent to ${webhook.source}`, {
        body: `saved as ${res.file}`,
      });
    } catch (err) {
      const apiErr = err as ApiError;
      const detail = apiErr.message || 'request failed';
      recordTest(webhook, { ok: false, status: apiErr.status ?? 0, at: Date.now(), detail });
      toasts.push('err', `Test delivery failed for ${webhook.source}`, { body: detail });
    }
  }
</script>

<section data-testid={TESTID.view('webhooks')} aria-labelledby={headingId}>
  <div class="head">
    <Icon name="hook" />
    <h2 id={headingId}>Webhooks</h2>
    {#if !webhooks.loading && sorted.length > 0}
      <span class="count mono">{countLabel}</span>
    {/if}
    <Button
      size="sm"
      iconOnly
      variant="ghost"
      aria-label="Refresh webhooks"
      loading={webhooks.loading}
      onclick={() => webhooks.load()}
    >
      {#snippet icon()}<Icon name="retry" />{/snippet}
    </Button>
    <div class="grow"></div>
    <Button
      variant="pri"
      size="sm"
      onclick={openCreate}
      disabled={agentOptions.length === 0}
      title={agentOptions.length === 0 ? 'No agents configured on this daemon' : undefined}
      data-testid={TESTID.webhooksNew}
    >
      {#snippet icon()}<Icon name="plus" />{/snippet}
      New webhook
    </Button>
  </div>

  {#if webhooks.loading && sorted.length === 0}
    <PaneState kind="loading" lines={3} />
  {:else if webhooks.error && sorted.length === 0}
    <PaneState kind="error" title="Couldn't load webhooks">
      {#snippet icon()}<Icon name="alert" />{/snippet}
      {#snippet hint()}{webhooks.error}{/snippet}
      {#snippet actions()}
        <Button size="sm" onclick={() => webhooks.load()}>
          {#snippet icon()}<Icon name="retry" />{/snippet}
          Retry
        </Button>
      {/snippet}
    </PaneState>
  {:else if sorted.length === 0}
    <!-- No separate CTA here - the header's "New webhook" button (above) is
         the one action, not duplicated per-state. -->
    <PaneState kind="empty" title="No webhooks yet">
      {#snippet icon()}<Icon name="hook" />{/snippet}
      {#snippet hint()}Create one above to receive inbound events into an agent's inbox.{/snippet}
    </PaneState>
  {:else}
    <div class="tablewrap">
      <table class="t-table" aria-label="Configured webhooks" data-testid={TESTID.webhooksTable}>
        <thead>
          <tr>
            <th scope="col">webhook</th>
            <th scope="col">delivery URL</th>
            <th scope="col">last test</th>
            <th scope="col">created</th>
            <th scope="col"><span class="vh">actions</span></th>
          </tr>
        </thead>
        <tbody>
          {#each sorted as webhook (webhook.token)}
            <WebhookRow
              {webhook}
              lastTest={lastTestByToken[webhook.token] ?? null}
              {now}
              onTest={testFire}
              onDelete={(w) => (pendingDelete = w)}
            />
          {/each}
        </tbody>
      </table>
    </div>

    <div class="testlog">
      <h3>Test log <span class="scope">this session</span></h3>
      {#if testLog.length === 0}
        <p class="empty-hint">Fire a test to see the real delivery result here.</p>
      {:else}
        <div
          class="t-log"
          role="log"
          aria-label="Test delivery results this session"
          data-testid={TESTID.webhooksLog}
        >
          {#each testLog as line (line.id)}
            <div class="ln" class:lvl-e={!line.ok}>
              <span class="ts_">{formatAgo(new Date(line.at).toISOString(), now, 'bare')}</span>
              {line.source} ({line.agent}) ← {line.status || 'error'}
              {line.detail}
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</section>

<Modal open={showCreate} onclose={() => (showCreate = false)} title="New webhook">
  <form id={createFormId} onsubmit={submitCreate} data-testid={TESTID.webhooksCreateForm}>
    <div class="mform">
      <Field id={agentFieldId} label="agent">
        {#snippet children()}
          <Select id={agentFieldId} bind:value={createAgent} options={agentOptions} />
        {/snippet}
      </Field>
      <Field
        id={sourceFieldId}
        label="source"
        hint="A short slug identifying who's calling this (e.g. github-events). Letters, digits, dot, underscore, dash."
        error={createError ?? undefined}
      >
        {#snippet children(describedBy)}
          <Input
            id={sourceFieldId}
            bind:value={createSource}
            placeholder="github-events"
            mono
            ariaDescribedby={describedBy}
          />
        {/snippet}
      </Field>
    </div>
  </form>
  {#snippet footer()}
    <Button onclick={() => (showCreate = false)}>Cancel</Button>
    <Button variant="pri" type="submit" form={createFormId} loading={creating}>
      {#snippet icon()}<Icon name="check" />{/snippet}
      Create
    </Button>
  {/snippet}
</Modal>

<Modal
  open={pendingDelete !== null}
  onclose={() => (pendingDelete = null)}
  title="Delete webhook?"
  tone="danger"
>
  {#if pendingDelete}
    <p data-testid={TESTID.webhooksDeleteConfirm}>
      Delete <code>{pendingDelete.source}</code> for agent <code>{pendingDelete.agent}</code>? Any
      service still posting to this URL will start getting 404s. This can't be undone.
    </p>
  {/if}
  {#snippet footer()}
    <Button onclick={() => (pendingDelete = null)}>Cancel</Button>
    <Button variant="danger" loading={deleting} onclick={doDelete}>
      {#snippet icon()}<Icon name="x" />{/snippet}
      Delete webhook
    </Button>
  {/snippet}
</Modal>

<style>
  section {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 14px 16px 26px;
    display: grid;
    gap: 16px;
    align-content: start;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .head h2 {
    margin: 0;
    font: 600 var(--fs-lg) var(--font-ui);
  }
  .count {
    font-size: var(--fs-2xs);
    color: var(--tx3);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .vh {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }
  .tablewrap {
    overflow-x: auto;
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
  }
  .t-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--fs-sm);
  }
  .t-table th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--bg1);
    text-align: left;
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
    padding: 7px 10px;
    border-bottom: 1px solid var(--bd1);
    white-space: nowrap;
  }
  .testlog h3 {
    margin: 0 0 6px;
    font: 600 var(--fs-xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
    display: flex;
    align-items: baseline;
    gap: 6px;
  }
  .testlog .scope {
    text-transform: none;
    font-weight: 400;
    color: var(--tx3);
    letter-spacing: normal;
  }
  .empty-hint {
    margin: 0;
    font-size: var(--fs-xs);
    color: var(--tx3);
  }
  .t-log {
    background: var(--bg0);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    font: 400 var(--fs-xs) / 1.65 var(--font-mono);
    padding: 8px 10px;
    max-height: 170px;
    overflow: auto;
    color: var(--tx2);
  }
  .t-log .ln {
    white-space: pre-wrap;
    word-break: break-word;
  }
  .t-log .ts_ {
    color: var(--tx3);
    opacity: 0.7;
  }
  .t-log .lvl-e {
    color: var(--st-err);
  }
  .mform {
    display: grid;
    gap: 11px;
    text-align: left;
  }
</style>
