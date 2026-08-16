<script lang="ts">
  // App shell - top bar, nav rail, keystrip, the shared workspace, the command
  // palette, the settings drawer, and the first-run token gate. The shell owns the
  // SSE connection lifecycle + the conn indicator.
  //
  // View-host contract: chats/terminals/files are WORKSPACE views - a nav click
  // swaps the context rail (session list / pty list / file tree) while the shared
  // multiplexer keeps its docked surfaces. Selecting a rail item opens/focuses that
  // item as a surface tab in the focused pane. Every other view is a FULL view: it
  // replaces the workspace region entirely, and switching back restores the mux
  // exactly as it was (its layout is never touched while a full view shows).
  import { untrack } from 'svelte';
  import { TESTID } from '$lib/testids';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { auth } from '$lib/stores/auth.svelte';
  import { conn } from '$lib/stores/conn.svelte';
  import { theme } from '$lib/stores/theme.svelte';
  import { spaces } from '$lib/stores/spaces.svelte';
  import { shellView } from '$lib/stores/shellView.svelte';
  import { router, initRouter, navigate } from '$lib/router.svelte';
  import { connectEvents, type SSEEvent } from '$lib/api/sse';
  import { routeShellEvent, type ShellEventSink } from '$lib/api/events';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { pluginsMeta } from '$lib/stores/pluginsMeta.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { jobs } from '$lib/stores/jobs.svelte';
  import { schedules } from '$lib/stores/schedules.svelte';
  import { terminals } from '$lib/stores/terminals.svelte';
  import { files } from '$lib/stores/files.svelte';
  import { usage } from '$lib/stores/usage.svelte';
  import { formatTokensCompact, formatUsd } from './views/usage/format';
  import { isEditableTarget } from '$lib/dom';
  import { allViews, dockedSurface, viewById } from './views';
  import { surfaceComponent } from './views/surfaces';
  import { workspacePhoneScreen } from '$lib/shell/phoneNav';
  import { focusedSurface } from '$lib/shell/shellNav';
  import { resolveShellShortcut } from '$lib/shell/keymap';
  import {
    buildPaletteItems,
    buildSessionItems,
    commandPaletteAction,
    runPaletteHref,
    type CommandLike,
    type PaletteContext,
  } from '$lib/shell/palette-sources';
  import { api } from '$lib/api/client';
  import { isFinishedSession, formatWhen, sessionTopic } from './views/chats/sessionModel';
  import { neighborSession } from './views/chats/chatNav';
  import { chatRouteParams } from './views/chats/chatLink';
  import { resolveDefaultSession } from './views/chats/defaultSession';
  import { composerPrefill } from './views/chats/composerPrefill.svelte';
  import { modelPickerRequest } from './views/chats/modelPickerSignal.svelte';
  import TopBar from '$lib/shell/TopBar.svelte';
  import NavRail from '$lib/shell/NavRail.svelte';
  import ContextRail from '$lib/shell/ContextRail.svelte';
  import TokenPane from '$lib/shell/TokenPane.svelte';
  import SettingsDrawer from '$lib/shell/SettingsDrawer.svelte';
  import Mux from '$lib/shell/mux/Mux.svelte';
  import Palette from '$lib/components/palette/Palette.svelte';
  import HelpOverlay from '$lib/components/overlays/HelpOverlay.svelte';
  import Toasts from '$lib/components/feedback/Toasts.svelte';

  initRouter();

  let settingsOpen = $state(false);
  let paletteOpen = $state(false);
  let helpOpen = $state(false);

  const activeView = $derived(viewById(shellView.activeViewId));
  // The view registry owns which region an id gets, so a plugin surface can ask
  // for the workspace the same way the built-in chats/terminals/files views do.
  const mode = $derived(activeView.mode);
  const railCollapsed = $derived(shellView.isRailCollapsed());

  // Narrow (phone) shell. Every workspace view (chats/terminals/files) drills down:
  // the rail/list and one item's content are separate full screens driven by the
  // hash (see phoneScreen), not a rail drawer toggled over the content.
  let narrow = $state(false);
  $effect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const m = window.matchMedia('(max-width: 640px)');
    const sync = () => (narrow = m.matches);
    sync();
    m.addEventListener('change', sync);
    return () => m.removeEventListener('change', sync);
  });

  // Phone drilldown screen for the active workspace view: 'list' (no content param
  // in the hash) or 'content' (a rail pick / deep link set it); null on desktop,
  // where the rail and content share the grid untouched.
  const phoneScreen = $derived(
    workspacePhoneScreen({
      narrow,
      view: router.view,
      workspaceView: shellView.workspaceView,
      params: router.params,
    }),
  );

  // The surface with focus in the mux, so the context rail can highlight the row
  // it belongs to and read its params.
  const focused = $derived(focusedSurface(spaces.active.layout));
  const focusedSessionId = $derived(
    focused?.kind === 'chat' ? (focused.params.sessionId ?? null) : null,
  );
  const focusedTerminalId = $derived(
    focused?.kind === 'terminal' ? (focused.params.terminalId ?? null) : null,
  );
  const focusedFilePath = $derived(focused?.kind === 'file' ? (focused.params.path ?? null) : null);

  // Nav activation drives the hash router; the effect below reflects the change
  // back into the shell store (and opens any deep-linked surface).
  function openView(id: string): void {
    navigate(id);
  }

  // --- opening surfaces into the shared mux (from context-rail selection) ---
  // A rail click reuses the existing surface of that kind (retargets it in
  // place); new tabs come from dragging a rail item into the mux instead.
  function openSurface(ref: Parameters<typeof spaces.openReusing>[0]): void {
    spaces.openReusing(ref);
  }
  function openChat(sessionId: string, agent?: string): void {
    const title = sessions.ordered.find((r) => r.id === sessionId)?.title ?? 'Chat';
    openSurface({ kind: 'chat', params: chatRouteParams(sessionId, agent), title });
  }
  function openTerminal(terminalId: string): void {
    const title = terminals.list.find((t) => t.id === terminalId)?.cmd ?? 'Terminal';
    openSurface({ kind: 'terminal', params: { terminalId }, title });
  }
  // Files open VSCode-style: a single-click previews into one reusable ephemeral
  // tab (the next click replaces it); double-clicking a file pins it (see
  // pinFilePreview). Chats/terminals keep the retarget-in-place openReusing path.
  function openFile(agent: string, path: string): void {
    spaces.openPreview({
      kind: 'file',
      params: { agent, path },
      title: path.split('/').pop() ?? path,
    });
  }

  // Rail selections route through the hash (not straight into the mux) so each
  // pick is a history entry - browser back/forward walks conversations,
  // terminals, and files again. The router effect below does the actual open.
  function selectChat(sessionId: string, agent?: string): void {
    navigate('chats', chatRouteParams(sessionId, agent));
  }
  function selectTerminal(terminalId: string): void {
    navigate('terminals', { terminalId });
  }
  function selectFile(agent: string, path: string): void {
    navigate('files', { ...(agent ? { agent } : {}), path });
  }
  // Double-clicking a file pins its preview (the single-clicks already opened it
  // as the focused pane's ephemeral tab). Not a navigation, so it hits the store
  // directly rather than the hash - it mutates an open surface, not history.
  function pinFilePreview(): void {
    spaces.pinPreviewInFocusedPane();
  }

  // Deep links - a shared #view URL, a PWA shortcut, browser back/forward - open
  // any surface the route names. Nav clicks route here too (they set the hash).
  // untrack keeps the store mutations out of the dep set.
  $effect(() => {
    const id = router.view;
    const params = router.params;
    untrack(() => {
      if (id === 'chats' && params.sessionId) openChat(params.sessionId, params.agent);
      else if (id === 'terminals' && params.terminalId) openTerminal(params.terminalId);
      else if (id === 'files' && params.path) openFile(params.agent ?? '', params.path);
    });
  });

  // Selecting the route's view is kept apart from opening its surfaces, because
  // plugin views join the registry after boot: this has to re-run on that arrival,
  // and re-running the block above would re-open a surface under the user.
  $effect(() => {
    const id = router.view;
    const known = allViews().some((v) => v.id === id);
    untrack(() => {
      if (id && known) shellView.activate(id);
    });
  });

  // A plugin surface that declared workspace mode docks into the mux like a rail
  // pick instead of taking the region. Kept out of the deep-link effect above
  // because a surface's mode arrives with the registry, after boot.
  $effect(() => {
    const surface = dockedSurface(router.view);
    const params = router.params;
    if (!surface) return;
    untrack(() => openSurface({ kind: surface.kind, params, title: surface.label }));
  });

  // Plugin metadata at shell scope: plugin-contributed UI surfaces seed the
  // surface registry (a persisted plugin tab needs its entry to render) and the
  // nav rail, both of which live outside the plugins view.
  $effect(() => {
    if (auth.gated) return;
    void pluginsMeta.load();
  });

  // --- command palette ---
  // Slash commands for the palette, fetched once at shell scope (best-effort - the
  // commands group is just omitted if it fails). The composer fetches its own copy
  // for the inline `/` menu; this list feeds the palette rows and the pick action
  // (commandPaletteAction: open the model picker / run / prefill).
  let commands = $state<CommandLike[]>([]);
  $effect(() => {
    if (auth.gated) return;
    api
      .get<{ commands: CommandLike[] }>('/api/commands')
      .then((res) => (commands = res.commands))
      .catch(() => (commands = []));
  });
  const paletteItems = $derived(
    buildPaletteItems({
      views: allViews().map((v) => ({ id: v.id, label: v.label, icon: v.icon })),
      surfaces: pluginsMeta.surfaces,
      themes: theme.list,
      currentTheme: theme.current,
      spaces: spaces.spaces.map((s) => ({ id: s.id, name: s.name })),
      activeSpaceId: spaces.activeSpaceId,
      commands,
    }),
  );
  // Sessions are a query-only palette pool: mapped from the already-loaded rail
  // rows (read-only), live-first, so ⌘K can jump to any chat by title/topic.
  const sessionItems = $derived(
    buildSessionItems(
      sessions.ordered.map((r) => ({
        id: r.id,
        title: r.title?.trim() || sessionTopic(r) || 'untitled chat',
        ended: isFinishedSession(r),
        when: formatWhen(r.last_active ?? r.created_at),
        topic: sessionTopic(r),
        superseded: !!r.superseded_by,
      })),
    ),
  );
  // Start a fresh chat and navigate to it, mirroring the chats rail's + button but
  // at shell scope so ⌘/Ctrl+Shift+O and the palette reach it from any view. The
  // roster is loaded lazily by whichever rail/view is mounted (chats is the default,
  // so it is populated in practice); before it lands, fall back to the chats view
  // where the rail's + button lives rather than force a load here.
  async function newChat(): Promise<string | null> {
    const agent = agentsMeta.agents[0]?.name;
    if (!agent) {
      openView('chats');
      return null;
    }
    const id = await sessions.newSession(agent);
    await sessions.load(agent);
    selectChat(id, agent);
    return id;
  }

  // A ⌘K command pick, decided by the command's arg hint (commandPaletteAction):
  // a `/model` opens that chat's header model picker; anything else runs now (every
  // param auto-injected, e.g. /status) or prefills `/name ` for the user to complete
  // - both routing through the target chat's composer (one path with the inline `/`
  // menu) via the prefill store, which survives the selectChat navigation. Target
  // the open chat if we're on one, else the most-recent live session, else a fresh
  // chat.
  async function runCommand(name: string): Promise<void> {
    const cmd = commands.find((c) => c.name === name);
    if (!cmd) return; // list still loading or unknown - no-op rather than guess
    const action = commandPaletteAction(cmd);
    const rows = sessions.ordered.filter((r) => !r.superseded_by);
    let targetId =
      router.view === 'chats' && router.params.sessionId
        ? router.params.sessionId
        : resolveDefaultSession(rows);
    if (targetId) {
      selectChat(targetId);
    } else {
      targetId = await newChat(); // already navigates (with the agent hint)
      if (!targetId) return;
    }
    if (action.kind === 'model-picker') modelPickerRequest.request(targetId);
    else composerPrefill.request(targetId, action.text, action.kind === 'run');
  }

  const paletteCtx: PaletteContext = {
    openView,
    openSurface: (kind) => openSurface({ kind, title: pluginsMeta.byKind(kind)?.label }),
    setTheme: (t) => theme.set(t),
    setSpace: (id) => spaces.setActive(id),
    openSettings: () => (settingsOpen = true),
    openSession: (id) => selectChat(id),
    newChat: () => void newChat(),
    showHelp: () => (helpOpen = true),
    runCommand: (name) => void runCommand(name),
  };

  // Alt+↑/↓ steps the chats rail. Operates on the same ordered+filtered list the
  // rail shows and moves via the hash, so it's just a keyboard-driven rail click.
  // Gated to the chats view, and never switches out from under an open overlay.
  function stepChat(dir: 1 | -1): void {
    if (paletteOpen || settingsOpen || helpOpen || router.view !== 'chats') return;
    const ids = sessions.ordered.filter((r) => !r.superseded_by).map((r) => r.id);
    const next = neighborSession(ids, router.params.sessionId ?? null, dir);
    if (next) selectChat(next);
  }

  function openPalette(): void {
    paletteOpen = true;
  }

  // SSE lifecycle. `connectEvents` already flips the auth gate on a 401 and
  // retries with replay; the shell needs only the connection status (for the conn
  // chips) and a hard reload when the daemon's epoch changes.
  $effect(() => {
    if (auth.gated) return;
    const handle = connectEvents(onShellEvent, (connected) => conn.setConnected(connected));
    const onVisible = () => {
      if (document.visibilityState === 'visible') handle.kick();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      handle.close();
    };
  });

  const shellEventSink: ShellEventSink = {
    onReconnect: () => location.reload(),
    onSessionEvent: (data) => {
      sessions.applySessionEvent(data);
      // Carries the agent's file writes too, which open file tabs follow.
      files.applySessionEvent(data);
    },
    onSessionUpdate: (data) => sessions.applySessionUpdate(data),
    onCompactionStarted: (data) => sessions.applyCompaction(data, true),
    onCompactionFinished: (data) => sessions.applyCompaction(data, false),
    onJobUpdate: (data) => jobs.applyJobUpdate(data),
    onScheduleUpdate: (data) => schedules.applyScheduleUpdate(data),
    onTerminalState: (data) => terminals.applyTerminalState(data),
  };

  // The shell holds the origin's one event stream for everyone on it, and the
  // shell's own handlers and the open plugin surfaces are independent consumers:
  // a surface hears every type its descriptor declared, whether or not a shell
  // handler acts on it too. Handing a frame to only one of the two would make a
  // plugin's reachable set depend on the shell's routing table.
  function onShellEvent(event: SSEEvent) {
    routeShellEvent(event, shellEventSink);
    pluginsMeta.applyPluginEvent(event);
  }

  // Keystrip "today" cost/tokens: a real since-UTC-midnight fetch, kicked off once
  // at boot so the rail reads real numbers from the start.
  $effect(() => {
    if (auth.gated) return;
    usage.loadToday();
  });
  const keystripCost = $derived(usage.today ? formatUsd(usage.today.total_cost) : undefined);
  const keystripTokens = $derived(
    usage.today ? formatTokensCompact(usage.today.total_tokens) : undefined,
  );

  function onKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      if (paletteOpen) paletteOpen = false;
      else if (settingsOpen) settingsOpen = false;
      else if (helpOpen) helpOpen = false;
      return;
    }
    const action = resolveShellShortcut({
      key: event.key,
      metaKey: event.metaKey,
      ctrlKey: event.ctrlKey,
      shiftKey: event.shiftKey,
      altKey: event.altKey,
      typing: isEditableTarget(event.target),
    });
    if (!action) return;
    event.preventDefault();
    if (action === 'toggle-palette') paletteOpen = !paletteOpen;
    else if (action === 'open-palette') paletteOpen = true;
    else if (action === 'open-settings') settingsOpen = true;
    else if (action === 'new-chat') void newChat();
    else if (action === 'show-help') helpOpen = !helpOpen;
    else if (action === 'next-chat') stepChat(1);
    else if (action === 'prev-chat') stepChat(-1);
    // Tab-switch is harmless over an overlay (the pane's under it), so it stays
    // unguarded, unlike stepChat.
    else if (action === 'next-tab') spaces.cycleTab(1);
    else if (action === 'prev-tab') spaces.cycleTab(-1);
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#if auth.gated}
  <TokenPane />
{:else}
  <div class="app">
    <TopBar
      onOpenPalette={openPalette}
      onOpenSettings={() => (settingsOpen = true)}
      cost={keystripCost}
      tokens={keystripTokens}
    />
    <div class="app-shell">
      <NavRail
        views={allViews()}
        activeId={shellView.activeViewId}
        collapsed={shellView.navCollapsed}
        onActivate={openView}
        onToggleCollapsed={() => shellView.toggleNav()}
        onOpenSettings={() => (settingsOpen = true)}
        {keystripCost}
        {keystripTokens}
      />
      <main class="app-main" id="app-main" data-testid={TESTID.viewHost} aria-label="Workspace">
        <!-- Workspace region: always mounted so terminals + docked surfaces survive
             a trip through a full view; hidden (not unmounted) while a full view shows. -->
        <section
          class="app-view work-view"
          data-testid={TESTID.view(shellView.workspaceView)}
          hidden={mode === 'full'}
        >
          <div
            class="work-shell"
            class:rail-collapsed={railCollapsed && !narrow}
            class:phone-list={phoneScreen === 'list'}
            class:phone-content={phoneScreen === 'content'}
          >
            {#if railCollapsed && !narrow}
              <button
                type="button"
                class="rail-expand"
                data-act="rail-collapse"
                aria-label="Show sidebar"
                title="Show sidebar"
                onclick={() => shellView.toggleRail()}
              >
                <Icon name="chev-r" />
              </button>
            {:else}
              <ContextRail
                view={shellView.workspaceView}
                onCollapse={() => shellView.toggleRail()}
                {focusedSessionId}
                {focusedTerminalId}
                {focusedFilePath}
                onOpenChat={selectChat}
                onOpenTerminal={selectTerminal}
                onOpenFile={selectFile}
                onPinFile={pinFilePreview}
              />
            {/if}
            <div class="work-main">
              <Mux
                layout={spaces.active.layout}
                onDock={(paneId, ref) => spaces.dock(paneId, ref)}
                onSplit={(paneId, dir, ref, position) => spaces.split(paneId, dir, ref, position)}
                onCloseTab={(paneId, tabId) => spaces.closeTab(paneId, tabId)}
                onCloseOtherTabs={(paneId, tabId) => spaces.closeOtherTabs(paneId, tabId)}
                onCloseAllTabs={(paneId) => spaces.closeAllTabs(paneId)}
                onSelectTab={(paneId, tabId) => spaces.selectTab(paneId, tabId)}
                onPinTab={(paneId, tabId) => spaces.pinTab(paneId, tabId)}
                onFocusPane={(paneId) => spaces.focusPane(paneId)}
                onResize={(splitId, dividerIndex, delta) =>
                  spaces.resize(splitId, dividerIndex, delta)}
                onNewTab={(paneId) => {
                  spaces.focusPane(paneId);
                  openPalette();
                }}
              >
                {#snippet content(tab, focusPane)}
                  {@const Surface = surfaceComponent(tab.kind)}
                  <!-- Key by the tab's identity so two same-kind surfaces (e.g. two
                       chats on different sessions) mount distinct instances. -->
                  {#key tab.id}
                    {#if Surface}
                      <Surface
                        params={tab.params}
                        kind={tab.kind}
                        setTitle={(title) => spaces.retitleTab(tab.id, title)}
                        {focusPane}
                      />
                    {/if}
                  {/key}
                {/snippet}
              </Mux>
            </div>
          </div>
        </section>

        {#if mode === 'full' && activeView.load}
          <section class="app-view" data-full-view={activeView.id}>
            {#await activeView.load() then mod}
              {@const Full = mod.default}
              <Full />
            {:catch}
              <p class="view-load-error">Failed to load this view. Try again.</p>
            {/await}
          </section>
        {/if}
      </main>
    </div>
    <SettingsDrawer open={settingsOpen} onclose={() => (settingsOpen = false)} />
  </div>
{/if}

<Palette
  bind:open={paletteOpen}
  items={paletteItems}
  {sessionItems}
  onSelect={(item) => runPaletteHref(item.href, paletteCtx)}
/>
<HelpOverlay bind:open={helpOpen} />
<Toasts />
<div data-testid={TESTID.appReady} hidden></div>

<style>
  /* .app / .app-shell / .app-main. `.app` is the
     positioned frame the settings drawer slides over. */
  .app {
    position: relative;
    display: flex;
    flex-direction: column;
    /* dvh, not vh: on phones the nav rail is the last flex child (a bottom bar),
       and 100vh extends behind the browser toolbar, pushing it out of reach - so a
       full view like Jobs became impossible to leave. */
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
  }
  .app-shell {
    flex: 1;
    display: flex;
    min-height: 0;
  }
  .app-main {
    flex: 1;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
  }
  /* Each screen fills the main region. The workspace screen stays mounted while a
     full view shows, so an explicit [hidden] rule (higher specificity than the
     flex default) actually hides it. */
  .app-view {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }
  .work-view[hidden] {
    display: none;
  }
  /* work-shell: context rail + mux. Collapsing the
     rail drops to a single column; the expand strip then rides the work-main edge. */
  .work-shell {
    --w-work: 262px;
    flex: 1;
    display: grid;
    grid-template-columns: clamp(200px, var(--w-work), 30%) minmax(0, 1fr);
    min-height: 0;
    position: relative;
    background: var(--bg0);
  }
  .work-shell.rail-collapsed {
    grid-template-columns: minmax(0, 1fr);
  }
  .work-main {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    position: relative;
  }
  .work-shell.rail-collapsed .work-main {
    margin-left: 20px;
  }
  /* Thin expand strip shown while the rail is collapsed. */
  .rail-expand {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 20px;
    z-index: 25;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg1);
    border: 0;
    border-right: 1px solid var(--bd1);
    color: var(--tx3);
    cursor: pointer;
    padding: 0;
  }
  .rail-expand:hover {
    color: var(--acc);
    background: var(--bg2);
  }
  .rail-expand :global(.ic) {
    width: 13px;
    height: 13px;
  }
  /* Narrow: the nav rail becomes a bottom bar under the view, and every workspace
     view drills down. The list screen's rail fills the
     surface (overlaying the mux); the content screen drops the rail entirely so the
     content is full-bleed and the header's back control is the way back. */
  @media (max-width: 640px) {
    .app-shell {
      flex-direction: column;
    }
    .work-shell {
      grid-template-columns: minmax(0, 1fr);
    }
    .work-shell.phone-list > :global(.work-rail) {
      position: absolute;
      inset: 0;
      z-index: 20;
    }
    /* The content surface sits behind the list overlay; hide it from view + focus
       (visibility, not display, so a docked terminal's canvas keeps its size). */
    .work-shell.phone-list .work-main {
      visibility: hidden;
    }
    .work-shell.phone-content > :global(.work-rail) {
      display: none;
    }
  }
</style>
