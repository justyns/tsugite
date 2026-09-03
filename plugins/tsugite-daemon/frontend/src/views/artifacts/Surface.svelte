<script lang="ts">
  // The document `open_artifact` asked the UI to show beside the chat, as an
  // ordinary mux surface. `params.id` is the daemon's artifact slot; the rest
  // lives in the artifacts store under that id. Watching the slot's `rev` rather
  // than its params is what reloads the pane on a re-open of the same path.
  //
  // Content comes from a workspace file read back through the authenticated API,
  // or from inline text the agent generated. Either way the agent chose it, so
  // rendered markdown and HTML alike go through the sandboxed preview frame the
  // file browser uses: raw HTML in the content never reaches the app document.
  import { untrack } from 'svelte';
  import { TESTID } from '$lib/testids';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import HtmlPreview from '$lib/components/media/HtmlPreview.svelte';
  import { markdownDoc } from '$lib/media/markdownDoc';
  import { artifacts, type AgentArtifact } from '$lib/stores/artifacts.svelte';
  import { spaces } from '$lib/stores/spaces.svelte';
  import { files } from '$lib/stores/files.svelte';
  import { renderMarkdown } from '../files/wiki';

  let {
    params,
    setTitle,
  }: { params?: Record<string, string>; setTitle?: (title: string) => void } = $props();

  const id = $derived(params?.id ?? '');
  const artifact = $derived(id ? artifacts.get(id) : undefined);

  let body = $state<string | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(false);
  let mode = $state<string>('rendered');

  const renderable = $derived(artifact != null && artifact.contentType !== 'text');
  const html = $derived(artifact?.contentType === 'html' && mode === 'rendered');
  const markdown = $derived(artifact?.contentType === 'markdown' && mode === 'rendered');
  const rendered = $derived(
    markdown && body != null ? markdownDoc(renderMarkdown(body, () => null)) : '',
  );
  const framed = $derived(html || markdown);
  const frameHtml = $derived(html ? (body ?? '') : rendered);

  // Re-run per (slot, revision): a second open of the same path still reloads,
  // and a replaced slot swaps content without remounting the tab.
  $effect(() => {
    const current = artifact;
    void current?.rev;
    untrack(() => {
      if (!current) return;
      mode = current.mode;
      setTitle?.(current.title);
      void load(current);
    });
  });

  async function load(doc: AgentArtifact) {
    error = null;
    if (doc.path === null) {
      body = doc.content;
      return;
    }
    loading = true;
    try {
      const file = await files.read(doc.path, doc.sessionId);
      body = file.content ?? '';
    } catch (err) {
      body = null;
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  function dismiss() {
    if (!id) return;
    // Drop the record first, so a stale pane can never outlive it.
    artifacts.close(id);
    spaces.closeSurface({ kind: 'artifact', params: { id } });
  }
</script>

<section class="art-shell" data-testid={TESTID.artifactPane} aria-label="Agent artifact">
  <header class="art-bar">
    {#if artifact?.openedByAgent}
      <span class="art-by" data-testid={TESTID.artifactAgentBadge}>
        <Icon name="agent" />
        <span>opened by the agent</span>
      </span>
    {/if}
    <span class="art-title" title={artifact?.path}>{artifact?.title ?? 'Artifact'}</span>
    <div class="grow"></div>
    {#if renderable}
      <span data-testid={TESTID.artifactModeSeg}>
        <Seg options={['rendered', 'source']} bind:value={mode} ariaLabel="Artifact view" />
      </span>
    {/if}
    <button
      type="button"
      class="art-x"
      aria-label="Close artifact"
      data-testid={TESTID.artifactClose}
      onclick={dismiss}
    >
      <Icon name="x" />
    </button>
  </header>

  <div class="art-body" class:is-frame={framed}>
    {#if !artifact}
      <!-- The slot is gone: an ephemeral artifact does not survive a reload, and
           there is nothing to re-read without a path. -->
      <PaneState kind="empty" title="This artifact is no longer open">
        {#snippet icon()}<Icon name="file" />{/snippet}
        {#snippet hint()}
          <span>Ask the agent to open it again.</span>
        {/snippet}
      </PaneState>
    {:else if error}
      <PaneState kind="error" title="Could not read this artifact">
        {#snippet hint()}<span class="mono">{error}</span>{/snippet}
        {#snippet actions()}
          <Button size="sm" onclick={() => load(artifact)}>Retry</Button>
        {/snippet}
      </PaneState>
    {:else if loading && body === null}
      <PaneState kind="loading" lines={7} />
    {:else if framed}
      <HtmlPreview
        html={frameHtml}
        docPath={artifact.path ?? ''}
        sessionId={artifact.sessionId}
        title={`Rendered ${artifact.title}`}
        testid={TESTID.artifactHtmlFrame}
      />
    {:else}
      <pre class="art-raw">{body ?? ''}</pre>
    {/if}
  </div>
</section>

<style>
  .art-shell {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
    min-height: 0;
    background: var(--bg1);
  }
  .art-bar {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    padding: 7px 10px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg2);
    flex: none;
  }
  .grow {
    flex: 1;
  }
  /* State is never signaled by color alone: the badge carries an icon + words. */
  .art-by {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    flex: none;
    padding: 2px 7px;
    border: 1px solid var(--bd1);
    border-radius: var(--r-full);
    background: var(--bg3);
    color: var(--tx2);
    font: 500 var(--fs-xs) / 1.4 var(--font-ui);
  }
  .art-title {
    color: var(--tx0);
    font: 600 var(--fs-sm) / 1.3 var(--font-ui);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .art-x {
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border: 1px solid transparent;
    border-radius: var(--r-md);
    background: transparent;
    color: var(--tx2);
    cursor: pointer;
  }
  .art-x:hover {
    background: var(--bg4);
    color: var(--tx0);
  }
  .art-x:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .art-body {
    flex: 1;
    min-height: 0;
    overflow: auto;
    padding: 16px 20px 30px;
  }
  /* The sandboxed frame is full-bleed and scrolls inside itself. */
  .art-body.is-frame {
    display: flex;
    padding: 0;
    overflow: hidden;
  }
  .art-raw {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font: var(--fs-sm) / 1.55 var(--font-mono);
    color: var(--tx1);
  }
  .mono {
    font-family: var(--font-mono);
  }
</style>
