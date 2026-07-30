<script lang="ts">
  // Chat surface: the conversation (replay + live stream) plus the composer, for
  // one session docked as a mux tab. The session list lives in the shared context
  // rail (SessionsRail); this surface just renders whichever session it's pointed
  // at. `params.sessionId` binds it to a specific session (a rail click or deep
  // link); with none, it resolves the default (primary/pinned/newest) like the old
  // single-surface view, so the default-docked chat still lands on a real thread.
  import { untrack } from 'svelte';
  import { conn } from '$lib/stores/conn.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { shellView } from '$lib/stores/shellView.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';
  import type { SessionRow } from '$lib/stores/sessions.svelte';
  import Conversation from './Conversation.svelte';
  import ChatComposer from './ChatComposer.svelte';
  import { ConversationController, type SendOpts } from './conversation.svelte';
  import { resolveDefaultSession } from './defaultSession';
  import { resolveChatAgent, isJobArtifact } from './chatAgent';
  import { goBackToWorkspaceList, isPhoneWidth } from '$lib/shell/phoneNav';
  import { hasFiles, extractFiles } from './dropFiles';

  let { params }: { params?: Record<string, string> } = $props();

  const ctrl = new ConversationController();
  // svelte-ignore state_referenced_locally -- seeds from the initial param; the effect below follows later changes.
  let selectedId = $state<string | null>(params?.sessionId ?? null);

  // Resolve the selected session's own agent + metadata from its record. A deep
  // link can arrive with a stale/wrong `agent` param (the jobs board passed the
  // job's worker-agent name, which is a builtin agent file, never a chat adapter),
  // so the session's TRUE agent - not the param - must drive the rail/effort/
  // composer calls or they 404. null until the fetch lands or the id is unknown.
  let sessionInfo = $state<{
    agent: string | null;
    metadata: Record<string, unknown>;
    contextLimit: number | null;
    cumulativeTokens: number | null;
  } | null>(null);
  // Fetch the session's identity + resolved context limit + cumulative tokens.
  // Sequence-guarded so a stale in-flight fetch never overwrites a newer
  // selection, and re-runnable: a settling turn refreshes the resolved limit and
  // token count the meter's fallback reads (see the busy-settle effect below).
  let infoSeq = 0;
  function loadSessionInfo(id: string) {
    const seq = ++infoSeq;
    void sessions
      .getInfo(id)
      .then((info) => {
        if (seq === infoSeq && id === selectedId) sessionInfo = info;
      })
      .catch(() => {});
  }
  $effect(() => {
    const id = selectedId;
    sessionInfo = null;
    infoSeq++; // invalidate any fetch still in flight for the previous selection
    if (id) loadSessionInfo(id);
  });

  const agent = $derived(
    resolveChatAgent({
      sessionAgent: sessionInfo?.agent ?? null,
      paramAgent: params?.agent,
      fallbackAgent: agentsMeta.agents[0]?.name,
    }),
  );
  // A background job artifact (worker/verifier session) is inspect-only: the
  // conversation continues in the parent chat, so drop the composer rather than
  // let a turn be injected into the transcript.
  const jobArtifact = $derived(isJobArtifact(sessionInfo?.metadata));
  // A session whose true agent is no longer in the live roster (removed from
  // config) can only fail a send, so the composer is gated off. Guarded on a
  // resolved sessionInfo AND a loaded roster so it doesn't flash mid-load; a deep
  // link to an unknown session (agent null) falls through to the backend's
  // session-owner routing rather than tripping this.
  const agentMissing = $derived(
    sessionInfo?.agent != null &&
      agentsMeta.agents.length > 0 &&
      !agentsMeta.agents.some((a) => a.name === sessionInfo!.agent),
  );
  const canCompose = $derived(!jobArtifact && !agentMissing);

  const rows = $derived(sessions.ordered);
  const selectedRow = $derived<SessionRow | null>(rows.find((r) => r.id === selectedId) ?? null);

  // Ensure the roster is loaded so `agent` resolves, then (re)load its sessions.
  $effect(() => {
    if (agentsMeta.agents.length === 0) void agentsMeta.load();
  });
  $effect(() => {
    if (agent) void sessions.load(agent);
  });

  // A rail click retargets this tab in place (spaces.openReusing rewrites the
  // tab's params; the instance survives) - follow the pointed-at session.
  const paramSessionId = $derived(params?.sessionId);
  $effect(() => {
    const id = paramSessionId;
    if (!id) return;
    untrack(() => {
      if (id !== selectedId) selectSession(id);
    });
  });

  // Seed / recover the selection: keep an explicit params.sessionId (or any live
  // selection) when it's still valid, otherwise fall back to the resolved default.
  // A background list reload never yanks a live selection, and an explicitly
  // pointed-at session is never yanked either - the shared sessions store holds
  // ONE agent's rows, so another surface/rail loading a different agent must not
  // steal this tab's cross-agent selection.
  $effect(() => {
    if (rows.length === 0) return;
    const current = selectedId;
    if (current && (current === paramSessionId || rows.some((r) => r.id === current))) return;
    const next = resolveDefaultSession(rows, params?.sessionId);
    if (next) untrack(() => selectSession(next));
  });

  // Keep the controller pointed at the selected session.
  $effect(() => {
    void ctrl.open(agent, selectedId);
  });

  // Feed the session's live server-busy truth into the controller so the working
  // indicator survives a switch-away/back (the local per-chat stream doesn't).
  $effect(() => {
    ctrl.serverBusy = selectedRow?.busy ?? false;
  });

  // Resume/reconnect catch-up. The per-chat send stream doesn't survive a mobile
  // background, and turn-end frames are withheld from the global broadcast, so a
  // reconnect (foreground kicks the SSE; a network blip drops it) leaves the open
  // conversation stale. On the transition back to `live`, reload its events in
  // place - non-destructive, and a no-op while a live local stream is delivering.
  let prevConn = conn.status;
  $effect(() => {
    const status = conn.status;
    const reconnected = status === 'live' && (prevConn === 'reconnecting' || prevConn === 'lost');
    prevConn = status;
    if (reconnected) untrack(() => void ctrl.resync());
  });

  // A turn that settles while this session is on screen (server busy true -> false)
  // won't push its reply down the withheld broadcast; reload in place so the answer
  // lands without a manual reopen. The first effect run just seeds prevBusy.
  let prevBusy = false;
  $effect(() => {
    const busy = selectedRow?.busy ?? false;
    const settled = prevBusy && !busy;
    prevBusy = busy;
    if (settled)
      untrack(() => {
        void ctrl.resync();
        // The first turn resolves the provider window and moves the token count;
        // refresh the meter's fallback so it isn't stale until a session switch.
        if (selectedId) loadSessionInfo(selectedId);
      });
  });

  // Route the open session's cross-session broadcast frames into the controller
  // so a turn started elsewhere (another tab, a schedule, /model from Discord)
  // grows the timeline live. Bound to the selected session; rebinds on switch,
  // and the controller ignores frames while THIS surface streams the turn.
  $effect(() => {
    const id = selectedId;
    if (!id) return;
    return sessions.bindConversation(id, (data) => ctrl.ingestBroadcast(data));
  });

  $effect(() => () => ctrl.closeStream());

  function selectSession(id: string) {
    selectedId = id;
    const row = rows.find((r) => r.id === id);
    if (row?.unread) void sessions.markViewed(id);
  }

  async function afterMutation() {
    await sessions.load(agent);
  }

  // Lifecycle actions must land visibly: a toast on success, a toast on failure
  // (an unawaited rejection would otherwise vanish and the menu looks like a no-op).
  async function lifecycle(action: () => Promise<void>, okMsg: string, errMsg: string) {
    try {
      await action();
      toasts.push('ok', okMsg);
      await afterMutation();
    } catch (err) {
      toasts.push('err', errMsg, {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  async function onSend(text: string, opts: SendOpts) {
    const id = await ctrl.send(text, opts);
    if (id && id !== selectedId) {
      selectedId = id;
      await sessions.load(agent);
    }
  }

  async function copySessionId() {
    if (!selectedId) return;
    try {
      await navigator.clipboard?.writeText(selectedId);
      toasts.push('ok', 'Session id copied', { body: selectedId });
    } catch {
      toasts.push('err', 'Could not copy session id');
    }
  }

  // OS file drag/drop onto the whole surface (conversation + composer), funneled
  // into the composer's attach pipeline. Only file drags are handled: internal
  // rail-to-mux drags carry a private MIME (never `Files`), so they pass straight
  // through to the mux, and a read-only artifact (no composer) is left inert.
  let composer = $state<{ attachFiles: (files: File[]) => void; focus: () => void }>();
  let surfaceEl = $state<HTMLDivElement>();
  let fileDragActive = $state(false);

  // Auto-focus the composer on navigation: landing on this surface with a session,
  // or the selected session changing, lets the user type immediately. Keyed on the
  // resolved session id so it fires on navigation ALONE - a stream frame, a
  // busy-settle resync, or a viewport resize never re-runs it against the same id,
  // so focus is never stolen while the user is reading or typing elsewhere. Gated
  // on a loaded sessionInfo so canCompose reflects the true metadata (a read-only
  // job artifact briefly looks composable before its info lands - don't grab focus
  // for it). Skipped at phone width, where it would pop the on-screen keyboard over
  // the just-opened conversation.
  let autofocusedId: string | null = null;
  $effect(() => {
    const id = selectedId;
    if (!id || sessionInfo == null || !canCompose || !composer) return;
    if (autofocusedId === id) return;
    autofocusedId = id;
    if (isPhoneWidth()) return;
    untrack(() => composer?.focus());
  });

  function onDragOver(e: DragEvent) {
    if (!canCompose || !hasFiles(e.dataTransfer)) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
    fileDragActive = true;
  }

  function onDragLeave(e: DragEvent) {
    // Ignore leaves into descendants; only clear when the pointer exits the surface.
    if (e.relatedTarget && surfaceEl?.contains(e.relatedTarget as Node)) return;
    fileDragActive = false;
  }

  function onDrop(e: DragEvent) {
    fileDragActive = false;
    if (!canCompose || !hasFiles(e.dataTransfer)) return;
    e.preventDefault();
    const files = extractFiles(e.dataTransfer);
    if (files.length) composer?.attachFiles(files);
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="chat-surface"
  bind:this={surfaceEl}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  ondrop={onDrop}
>
  <Conversation
    {ctrl}
    row={selectedRow}
    fallbackContext={sessionInfo?.contextLimit
      ? { tokens: sessionInfo.cumulativeTokens ?? 0, limit: sessionInfo.contextLimit }
      : null}
    railCollapsed={shellView.isRailCollapsed('chats')}
    onToggleRail={() => shellView.toggleRail('chats')}
    onBack={() => goBackToWorkspaceList('chats')}
    onRenameCommit={(title) => selectedId && void sessions.rename(selectedId, title)}
    onTopicCommit={(topic) => selectedId && void sessions.setTopic(selectedId, topic)}
    onComplete={() =>
      selectedId &&
      void lifecycle(
        () => sessions.complete(selectedId!),
        'Session marked complete',
        'Could not mark complete',
      )}
    onCancel={() =>
      selectedId &&
      void lifecycle(() => sessions.cancel(selectedId!), 'Run cancelled', 'Could not cancel')}
    onRestart={() =>
      selectedId &&
      void lifecycle(() => sessions.restart(selectedId!), 'Session restarted', 'Could not restart')}
    onPin={() => selectedId && void sessions.pin(selectedId)}
    onUnpin={() => selectedId && void sessions.unpin(selectedId)}
    onSetPrimary={() => selectedId && void sessions.setPrimary(selectedId).then(afterMutation)}
    onCopyId={() => void copySessionId()}
    onOpenSession={selectSession}
    onRetry={(text) => void onSend(text, { uploadedFiles: [] })}
  />
  {#if canCompose}
    <ChatComposer
      bind:this={composer}
      {agent}
      sessionId={selectedId}
      streaming={ctrl.streaming}
      busy={selectedRow?.busy ?? false}
      queuedMessages={ctrl.queued.map((q) => q.text)}
      restoreFailed={ctrl.sendFailed}
      {onSend}
      onStop={() => void ctrl.stop()}
      onQueue={(text, opts) => ctrl.queue(text, opts)}
      onUnqueue={(i) => ctrl.unqueue(i)}
      onCommandResult={(command, output, ok, action) => ctrl.pushEcho(command, output, ok, action)}
    />
  {:else if jobArtifact}
    <div class="ro-note" data-testid={TESTID.chatReadonly}>
      <Icon name="lock" size={13} />
      <span>Read-only: a job worker or verifier transcript. Reply in the parent chat.</span>
    </div>
  {:else}
    <div class="ro-note" data-testid={TESTID.chatReadonly}>
      <Icon name="lock" size={13} />
      <span
        >This chat's agent '{sessionInfo?.agent}' is no longer configured, so it can't take new
        messages.</span
      >
    </div>
  {/if}
  {#if fileDragActive}
    <div class="chat-drop" aria-hidden="true">
      <div class="chat-drop-card">
        <Icon name="files" size={15} />
        <span>Drop files to attach</span>
      </div>
    </div>
  {/if}
</div>

<style>
  .chat-surface {
    flex: 1;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    background: var(--bg0);
    position: relative;
  }
  /* File-drag affordance, matching the mux drop-zone skin (dashed accent border
     + translucent accent wash + a pill label). Non-interactive so the drop lands
     on the surface underneath. */
  .chat-drop {
    position: absolute;
    inset: 0;
    z-index: 40;
    display: grid;
    place-items: center;
    pointer-events: none;
    border: 2px dashed var(--acc);
    border-radius: var(--r-md);
    background: color-mix(in oklab, var(--acc) 12%, transparent);
  }
  .chat-drop-card {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    color: var(--tx0);
    font: 600 var(--fs-xs) var(--font-mono);
    padding: 6px 12px;
    border-radius: var(--r-full);
    box-shadow: var(--sh-2);
  }
  .ro-note {
    flex: none;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    color: var(--tx3);
    font-size: var(--fs-sm);
  }
</style>
