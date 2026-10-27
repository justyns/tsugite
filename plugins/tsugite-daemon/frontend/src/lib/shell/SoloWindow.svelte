<script lang="ts">
  // The chat surface alone, for a window of its own: no nav rail, top bar, context
  // rail or mux tabs. An agent artifact docks beside the conversation as a second pane.
  import { sessions } from '$lib/stores/sessions.svelte';
  import { artifacts, artifactSurfaceParams } from '$lib/stores/artifacts.svelte';
  import type { ApiError } from '$lib/api/client';
  import { buildHash, type RouteParams } from '$lib/router.svelte';
  import { TESTID } from '$lib/testids';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import ChatSurface from '../../views/chats/Surface.svelte';
  import ArtifactSurface from '../../views/artifacts/Surface.svelte';
  import { SOLO_VIEW, resolveSolo, type SoloProbe } from './solo';

  let { params }: { params: RouteParams } = $props();

  let probe = $state<SoloProbe>('pending');
  let listLoaded = $state(false);
  let listFailed = $state(false);
  let attempt = $state(0);
  $effect(() => {
    const id = params.sessionId;
    void attempt;
    probe = 'pending';
    listFailed = false;
    let stale = false;
    void (async () => {
      await sessions.load();
      if (stale) return;
      if (sessions.error) {
        listFailed = true;
        return;
      }
      listLoaded = true;
      if (!id || sessions.rows.some((r) => r.id === id)) return;
      try {
        const detail = await sessions.getDetail(id);
        if (stale) return;
        const successor = detail.superseded_by;
        probe =
          typeof successor === 'string' && successor
            ? { kind: 'superseded', successorId: successor }
            : 'found';
      } catch (err) {
        if (!stale) probe = (err as ApiError).status === 404 ? 'missing' : 'error';
      }
    })();
    return () => (stale = true);
  });

  const resolution = $derived(resolveSolo(params, sessions.rows, probe, listLoaded));
  const artifact = $derived(
    resolution.state === 'ready' ? artifacts.agentForSession(resolution.sessionId) : undefined,
  );
  // The visible pane at 700px wide or less.
  let narrowPane = $state('chat');
</script>

<main class="solo" data-testid={TESTID.solo} aria-label="Chat">
  {#if resolution.state === 'ready' && !listFailed}
    {#if artifact}
      <div class="solo-switch">
        <Seg options={['chat', 'artifact']} bind:value={narrowPane} ariaLabel="Pane" />
      </div>
    {/if}
    <div class="solo-pane" class:narrow-hidden={artifact && narrowPane !== 'chat'}>
      <ChatSurface params={{ sessionId: resolution.sessionId }} solo dense />
    </div>
    {#if artifact}
      <div
        class="solo-pane"
        class:narrow-hidden={narrowPane !== 'artifact'}
        data-testid={TESTID.soloArtifact}
      >
        <ArtifactSurface params={artifactSurfaceParams(artifact)} />
      </div>
    {/if}
  {:else}
    <div class="solo-note" role="status" data-testid={TESTID.soloNote}>
      {#if listFailed}
        <p>Couldn't load the session list.</p>
        <button type="button" class="retry" onclick={() => attempt++}>Retry</button>
      {:else if resolution.state === 'superseded'}
        <p>This session was compacted into a successor.</p>
        <a href={buildHash(SOLO_VIEW, { sessionId: resolution.successorId })}>Open the successor</a>
      {:else if resolution.state === 'not-found'}
        <p>No session <code>{resolution.sessionId}</code> exists.</p>
      {:else if resolution.state === 'error'}
        <p>Couldn't load session <code>{resolution.sessionId}</code>.</p>
        <button type="button" class="retry" onclick={() => attempt++}>Retry</button>
      {:else if resolution.state === 'alias-empty'}
        <p>No session holds the alias <code>{resolution.alias}</code> yet.</p>
        <p class="hint">This window opens it as soon as one claims the alias.</p>
      {:else if resolution.state === 'loading'}
        <p>Loading session…</p>
      {:else}
        <p>Add <code>sessionId=…</code> or <code>alias=…</code> to the URL hash.</p>
      {/if}
    </div>
  {/if}
</main>

<style>
  .solo {
    display: flex;
    height: 100vh;
    height: 100dvh;
    min-width: 0;
    background: var(--bg0);
    overflow: hidden;
  }
  .solo-pane {
    flex: 1 1 0;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .solo-pane + .solo-pane {
    border-left: 1px solid var(--bd1);
  }
  .solo-switch {
    display: none;
  }
  @media (max-width: 700px) {
    .solo {
      flex-direction: column;
    }
    .solo-switch {
      display: flex;
      justify-content: center;
      padding: 6px;
      border-bottom: 1px solid var(--bd1);
    }
    .narrow-hidden {
      display: none;
    }
    .solo-pane + .solo-pane {
      border-left: 0;
    }
  }
  .solo-note {
    margin: auto;
    padding: 24px;
    text-align: center;
    color: var(--tx2);
    font-size: var(--fs-sm);
  }
  .retry {
    background: none;
    border: 0;
    padding: 0;
    color: var(--acc);
    font: inherit;
    text-decoration: underline;
    cursor: pointer;
  }
  .hint {
    color: var(--tx3);
  }
</style>
