<script lang="ts">
  // The one component every plugin UI surface renders through: the plugin's page
  // in a pane-filling iframe, plus the host end of the postMessage bridge.
  // Threat model and the public-assets rule: docs/plugin-adapters.md.
  import { auth } from '$lib/stores/auth.svelte';
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
  import Icon from '$lib/components/icon/Icon.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import { TESTID } from '$lib/testids';
  import type { SurfaceProps } from '../../../views/surfaces';

  let { params = {}, kind = '', setTitle }: SurfaceProps = $props();

  const surface = $derived(pluginsMeta.byKind(kind));
  const src = $derived(surface ? surfaceSrc(surface, params) : '');

  let frame = $state<HTMLIFrameElement | null>(null);
  let phase = $state<'loading' | 'ready' | 'stalled'>('loading');
  let attempt = $state(0);
  /** Theme the frame has already been told about, so reaching `ready` doesn't
   *  re-push the one `init` just carried. */
  let sentTheme = theme.current;

  /** A surface the registry has settled on as absent is gone for good; before it
   *  settles the tab's plugin may still arrive. */
  const status = $derived(surface ? phase : pluginsMeta.loaded ? 'missing' : 'loading');

  function payload(): ThemePayload {
    return { name: theme.current, tokens: readThemeTokens(document.documentElement) };
  }

  function onLoad(): void {
    sentTheme = theme.current;
    frame?.contentWindow?.postMessage(
      initMessage(kind, params, payload(), auth.token),
      location.origin,
    );
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

  // Armed only once the frame exists, so a slow registry load doesn't spend the
  // plugin's whole handshake budget before it has been given a chance.
  $effect(() => {
    if (status !== 'loading' || !surface) return;
    const timer = setTimeout(() => (phase = 'stalled'), READY_TIMEOUT_MS);
    return () => clearTimeout(timer);
  });

  // Before the handshake the init message carries the current tokens anyway.
  $effect(() => {
    const next = theme.current;
    if (next === sentTheme) return;
    sentTheme = next;
    frame?.contentWindow?.postMessage(themeMessage(payload()), location.origin);
  });
</script>

<div class="plugin-surface" data-testid={TESTID.pluginSurface} data-phase={status}>
  {#if status !== 'ready'}
    <div class="ps-overlay">
      {#if status === 'stalled'}
        <PaneState kind="error" title="Couldn't load {surface?.label}">
          {#snippet icon()}<Icon name="alert" />{/snippet}
          {#snippet hint()}<span class="mono">{surface?.entry}</span>{/snippet}
          {#snippet actions()}
            <Button size="sm" data-testid={TESTID.pluginSurfaceReload} onclick={reload}>
              {#snippet icon()}<Icon name="retry" />{/snippet}
              Reload
            </Button>
          {/snippet}
        </PaneState>
      {:else if status === 'missing'}
        <div data-testid={TESTID.pluginSurfaceMissing}>
          <PaneState kind="error" title="This tab's plugin isn't installed">
            {#snippet icon()}<Icon name="alert" />{/snippet}
            {#snippet hint()}<span class="mono">{kind}</span>{/snippet}
          </PaneState>
        </div>
      {:else}
        <PaneState kind="loading" lines={3} />
      {/if}
    </div>
  {/if}
  {#if surface}
    {#key attempt}
      <iframe
        bind:this={frame}
        {src}
        title={surface.label}
        sandbox="allow-scripts allow-forms allow-same-origin"
        onload={onLoad}
      ></iframe>
    {/key}
  {/if}
</div>

<style>
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
  /* Covers the frame until the handshake lands, so a half-painted page never shows. */
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
