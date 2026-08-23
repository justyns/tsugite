<script lang="ts" module>
  export interface TestResult {
    ok: boolean;
    /** HTTP status, or 0 for a network-level failure (no response at all). */
    status: number;
    at: number;
    detail: string;
  }
</script>

<script lang="ts">
  // One configured webhook. Hand-rolled <tr> (not the shared Table.svelte)
  // because every cell here needs its own local interactive state (reveal
  // toggle, in-flight test-fire) - the same reason SecTable.svelte hand-rolls
  // its own <table> instead of forcing the generic data-driven component.
  import type { Webhook } from '$lib/stores/webhooks.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Chip from '$lib/components/buttons/Chip.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { deliveryUrl } from './logic';
  import { formatAgo } from '$lib/relativeTime';

  let {
    webhook,
    lastTest = null,
    now,
    onTest,
    onDelete,
  }: {
    webhook: Webhook;
    lastTest?: TestResult | null;
    /** Ticks periodically in the parent so relative-time labels stay fresh. */
    now: number;
    onTest: (webhook: Webhook) => Promise<void>;
    onDelete: (webhook: Webhook) => void;
  } = $props();

  let revealed = $state(false);
  let testing = $state(false);

  const fullUrl = $derived(deliveryUrl(webhook.token, window.location.origin));

  async function copyUrl(): Promise<void> {
    try {
      await navigator.clipboard.writeText(fullUrl);
      toasts.push('ok', 'Delivery URL copied');
    } catch {
      toasts.push('err', 'Could not copy', { body: 'Clipboard access was denied by the browser.' });
    }
  }

  async function fire(): Promise<void> {
    testing = true;
    try {
      await onTest(webhook);
    } finally {
      testing = false;
    }
  }
</script>

<tr data-testid={TESTID.webhookRow(webhook.token)}>
  <td>
    <div class="src">{webhook.source}</div>
    <Chip>{webhook.agent}</Chip>
  </td>
  <td>
    <div class="wh-url">
      {#if revealed}
        <code class="url-val">{fullUrl}</code>
      {:else}
        <span class="url-mask" aria-label="Delivery URL hidden">••••••••••••••••</span>
      {/if}
      <Button
        size="sm"
        aria-pressed={revealed}
        onclick={() => (revealed = !revealed)}
        data-testid={TESTID.webhookReveal(webhook.token)}
      >
        {revealed ? 'hide' : 'show'}
      </Button>
      <Button
        size="sm"
        iconOnly
        variant="ghost"
        aria-label="Copy delivery URL for {webhook.source}"
        data-testid={TESTID.webhookCopy(webhook.token)}
        onclick={copyUrl}
      >
        {#snippet icon()}<Icon name="copy" />{/snippet}
      </Button>
    </div>
  </td>
  <td>
    {#if lastTest}
      <span
        class="test-result"
        class:is-err={!lastTest.ok}
        title={lastTest.detail}
        aria-label="{lastTest.ok ? 'Test succeeded' : 'Test failed'}: {lastTest.detail}"
      >
        <Icon name={lastTest.ok ? 'check' : 'x'} size={11} />
        {lastTest.status || 'error'}
        <span class="t-sub">{formatAgo(new Date(lastTest.at).toISOString(), now, 'bare')}</span>
      </span>
    {:else}
      <span class="test-result is-idle">
        <Icon name="ring" size={11} />
        not tested this session
      </span>
    {/if}
  </td>
  <td class="c3 mono">{formatAgo(webhook.created_at, now, 'bare')}</td>
  <td>
    <div class="acts">
      <Button
        size="sm"
        loading={testing}
        onclick={fire}
        data-testid={TESTID.webhookTest(webhook.token)}
      >
        {#snippet icon()}<Icon name="send" />{/snippet}
        test fire
      </Button>
      <Button
        size="sm"
        iconOnly
        variant="ghost"
        aria-label="Delete webhook {webhook.source}"
        data-testid={TESTID.webhookDelete(webhook.token)}
        onclick={() => onDelete(webhook)}
      >
        {#snippet icon()}<Icon name="x" />{/snippet}
      </Button>
    </div>
  </td>
</tr>

<style>
  td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--bd0);
    vertical-align: middle;
  }
  tr:hover td {
    background: color-mix(in oklab, var(--bg3) 45%, transparent);
  }
  .c3 {
    color: var(--tx3);
  }
  .mono {
    font-family: var(--font-mono);
  }
  .src {
    font: 600 var(--fs-md) var(--font-ui);
    color: var(--tx0);
    margin-bottom: 3px;
  }
  .wh-url {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .url-val {
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
  }
  .url-mask {
    font: 600 var(--fs-md) / 1 var(--font-mono);
    color: var(--tx3);
    letter-spacing: 0.18em;
  }
  .test-result {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--st-ok);
  }
  .test-result.is-err {
    color: var(--st-err);
  }
  .test-result.is-idle {
    color: var(--tx3);
  }
  .test-result .t-sub {
    color: var(--tx3);
    font-size: var(--fs-2xs);
  }
  .acts {
    display: flex;
    gap: 6px;
    justify-content: flex-end;
  }
</style>
