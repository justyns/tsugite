<script lang="ts">
  // Debug overlay: what the daemon stores about a session, for diagnosing missing
  // or stale workstream links and arbitrary agent-authored keys. Repeats the
  // header's link chips, which narrow screens hide - this is the only way to
  // reach them from a phone.
  import { onMount } from 'svelte';
  import RawOverlay from './RawOverlay.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { TESTID } from '$lib/testids';
  import type { MetaLink } from './metaLinks';

  let {
    sessionId,
    metadata,
    links,
    onClose,
  }: {
    sessionId: string;
    metadata: Record<string, unknown>;
    links: MetaLink[];
    onClose: () => void;
  } = $props();

  // Content, not metadata: each has its own surface and would swamp the overlay.
  // `deferred_deliveries` is whole cards with message bodies; session_detail
  // narrows `pending_deliveries` to ids but leaves this one raw.
  const CONTENT_FIELDS = ['metadata', 'prompt', 'result', 'deferred_deliveries'];

  let copied = $state(false);
  let detail = $state<Record<string, unknown> | null>(null);

  // A fresh read, not the row the sidebar cached: the point of the overlay is
  // what the daemon holds right now.
  onMount(() => {
    sessions.getDetail(sessionId).then(
      (d) => (detail = d),
      // The metadata block still renders from the session row.
      () => {},
    );
  });

  const sessionText = $derived(
    detail
      ? JSON.stringify(
          Object.fromEntries(Object.entries(detail).filter(([k]) => !CONTENT_FIELDS.includes(k))),
          null,
          2,
        )
      : '',
  );
  const metaText = $derived(
    JSON.stringify((detail?.metadata as Record<string, unknown> | undefined) ?? metadata, null, 2),
  );
  const text = $derived(sessionText ? `${sessionText}\n\n${metaText}` : metaText);

  async function copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
      setTimeout(() => {
        copied = false;
      }, 1200);
    } catch {
      // Clipboard unavailable (insecure context / denied): the raw JSON still reads.
    }
  }
</script>

<RawOverlay title="raw session metadata" width="760px" testid={TESTID.chatRawMetadata} {onClose}>
  {#snippet actions()}
    <button type="button" class="raw-copy" onclick={copy}>{copied ? 'copied' : 'copy'}</button>
  {/snippet}

  {#if links.length > 0}
    <div class="meta-links">
      {#each links as link (link.key)}
        <a href={link.href} target="_blank" rel="noreferrer">
          <Icon name="link" size={11} />{link.label}<Icon name="out" size={9} />
        </a>
      {/each}
    </div>
  {/if}
  {#if sessionText}
    <h3>session</h3>
    <pre>{sessionText}</pre>
    <h3>metadata</h3>
  {/if}
  <pre>{metaText}</pre>
</RawOverlay>

<style>
  .meta-links {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .meta-links a {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: var(--r-md);
    background: var(--bg1);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    text-decoration: none;
  }
  .meta-links a:hover {
    color: var(--acc);
  }
  h3 {
    margin: 12px 0 4px;
    font: 600 var(--fs-xs) / 1.4 var(--font-mono);
    color: var(--tx3);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  h3:first-of-type {
    margin-top: 0;
  }
  pre {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font: 400 var(--fs-sm) / 1.55 var(--font-mono);
    color: var(--tx1);
  }
</style>
