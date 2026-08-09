<script lang="ts">
  // The one component every plugin UI surface renders through: the plugin's page
  // in a pane-filling iframe, plus the host end of the postMessage bridge (init →
  // ready handshake, theme pushes, plugin-set tab title).
  //
  // The frame is same-origin under /api/plugins/<name>/, which is what lets the
  // plugin page reach its own routes - and equally what makes the sandbox
  // defence in depth rather than a boundary. A plugin surface is trusted code
  // the operator installed; a hostile adapter already has Python-level access to
  // the daemon. The entry page loads as a browser navigation, which carries no
  // bearer header, so it has to come from the plugin's public routes.
  import { pluginsMeta } from '$lib/stores/pluginsMeta.svelte';
  import { theme } from '$lib/stores/theme.svelte';
  import {
    READY_TIMEOUT_MS,
    initMessage,
    parsePluginMessage,
    readThemeTokens,
    surfaceSrc,
    themeMessage,
    type ThemePayload,
  } from '$lib/plugins/bridge';
  import Button from '$lib/components/buttons/Button.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import { TESTID } from '$lib/testids';
  import type { SurfaceProps } from '../../../views/surfaces';

  let { params = {}, kind = '', setTitle }: SurfaceProps = $props();

  const surface = $derived(pluginsMeta.byKind(kind));
  const src = $derived(surface ? surfaceSrc(surface, params) : '');

  let frame = $state<HTMLIFrameElement | null>(null);
  let phase = $state<'loading' | 'ready' | 'stalled'>('loading');
  /** Bumped by Reload to tear the frame down and start the handshake over. */
  let attempt = $state(0);
  /** Theme the frame has already been told about, so reaching `ready` doesn't
   *  re-push the one `init` just carried. */
  let sentTheme = theme.current;

  function payload(): ThemePayload {
    return { name: theme.current, tokens: readThemeTokens(document.documentElement) };
  }

  function onLoad(): void {
    sentTheme = theme.current;
    frame?.contentWindow?.postMessage(initMessage(kind, params, payload()), location.origin);
  }

  function reload(): void {
    phase = 'loading';
    attempt += 1;
  }

  $effect(() => {
    function onMessage(event: MessageEvent): void {
      if (!frame || event.source !== frame.contentWindow) return;
      const message = parsePluginMessage(event.data);
      if (!message) return;
      if (message.type === 'tsugite:ready') phase = 'ready';
      else if (message.type === 'tsugite:title') setTitle?.(message.title);
    }
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  });

  // Give up waiting per attempt: reload() flips phase back, which restarts this.
  $effect(() => {
    if (phase !== 'loading') return;
    const timer = setTimeout(() => (phase = 'stalled'), READY_TIMEOUT_MS);
    return () => clearTimeout(timer);
  });

  // A theme switch re-skins a live surface; before the handshake the init
  // message carries the current tokens anyway.
  $effect(() => {
    const next = theme.current;
    if (next === sentTheme) return;
    sentTheme = next;
    frame?.contentWindow?.postMessage(themeMessage(payload()), location.origin);
  });
</script>

<div class="plugin-surface" data-testid={TESTID.pluginSurface} data-phase={phase}>
  {#if surface}
    {#if phase !== 'ready'}
      <div class="ps-overlay">
        {#if phase === 'stalled'}
          <PaneState kind="error" title="{surface.label} did not finish loading.">
            {#snippet hint()}<span class="mono">{surface.entry}</span>{/snippet}
            {#snippet actions()}
              <Button size="sm" data-testid={TESTID.pluginSurfaceReload} onclick={reload}>
                Reload
              </Button>
            {/snippet}
          </PaneState>
        {:else}
          <PaneState kind="loading" lines={3} />
        {/if}
      </div>
    {/if}
    {#key attempt}
      <iframe
        bind:this={frame}
        {src}
        title={surface.label}
        sandbox="allow-scripts allow-forms allow-same-origin"
        onload={onLoad}
      ></iframe>
    {/key}
  {:else if pluginsMeta.loaded}
    <!-- The tab outlived its plugin. Keep its slot so the user can see what they
         lost and close it deliberately, rather than the pane going blank. -->
    <div class="ps-overlay" data-testid={TESTID.pluginSurfaceMissing}>
      <PaneState kind="error" title="This tab's plugin isn't installed on this daemon.">
        {#snippet hint()}<span class="mono">{kind}</span>{/snippet}
      </PaneState>
    </div>
  {:else}
    <div class="ps-overlay"><PaneState kind="loading" lines={3} /></div>
  {/if}
</div>

<style>
  /* Fills the pane (unlike the MCP app view's capped card); the overlay covers
     the frame until the handshake lands so a half-painted page never shows. */
  .plugin-surface {
    position: relative;
    display: grid;
    min-width: 0;
    min-height: 0;
    height: 100%;
    background: var(--bg1);
  }
  .plugin-surface iframe {
    width: 100%;
    height: 100%;
    border: 0;
    display: block;
  }
  .ps-overlay {
    position: absolute;
    inset: 0;
    z-index: 1;
    display: grid;
    align-content: center;
    padding: var(--sp-4);
    background: var(--bg1);
  }
  .mono {
    font-family: var(--font-mono);
  }
</style>
