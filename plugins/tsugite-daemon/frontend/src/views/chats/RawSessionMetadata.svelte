<script lang="ts">
  // Debug overlay: raw session.metadata exactly as stored by the daemon, for
  // diagnosing missing or stale workstream links and arbitrary agent-authored
  // keys. Repeats the header's link chips, which narrow screens hide - this is
  // the only way to reach them from a phone.
  import RawOverlay from './RawOverlay.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';
  import type { MetaLink } from './metaLinks';

  let {
    metadata,
    links,
    onClose,
  }: {
    metadata: Record<string, unknown>;
    links: MetaLink[];
    onClose: () => void;
  } = $props();

  let copied = $state(false);
  const text = $derived(JSON.stringify(metadata, null, 2));

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
  <pre>{text}</pre>
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
  pre {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font: 400 var(--fs-sm) / 1.55 var(--font-mono);
    color: var(--tx1);
  }
</style>
