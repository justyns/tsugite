<script lang="ts">
  // The conversation pane: header (rail toggle, title w/ inline rename, type
  // badge, live pill, context meter, session menu), a compaction banner, the
  // turn timeline built from replay + live events, the ask_user prompt, and an
  // auto-follow "jump to live" affordance. Turns render through the chatturns
  // library components (Msg/Prose/Think/CodeBlock/ExecBlock); everything is fed
  // from the ConversationController's $derived timeline.
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Pill from '$lib/components/buttons/Pill.svelte';
  import type { PillState } from '$lib/components/buttons/pill-state';
  import Msg from '$lib/components/chatturns/Msg.svelte';
  import Prose from '$lib/components/chatturns/Prose.svelte';
  import Think from '$lib/components/chatturns/Think.svelte';
  import CodeBlock from '$lib/components/chatturns/CodeBlock.svelte';
  import ExecBlock from '$lib/components/chatturns/ExecBlock.svelte';
  import LocalEcho from '$lib/components/chatturns/LocalEcho.svelte';
  import Work from '$lib/components/feedback/Work.svelte';
  import Ask from '$lib/components/ask/Ask.svelte';
  import type { SessionRow } from '$lib/stores/sessions.svelte';
  import { TESTID } from '$lib/testids';
  import type { ConversationController } from './conversation.svelte';
  import { ScrollFollow } from './scrollFollow.svelte';
  import { splitStreamFence, type Compaction, type DeliveryBlock } from './turns';
  import { formatAgo } from '$lib/relativeTime';
  import { buildHash } from '$lib/router.svelte';
  import { contextProvider } from '$lib/context/contextProviders';
  import type { IconName } from '$lib/components/icon/icons';
  import { sessionSourceType, isFinishedSession } from './sessionModel';
  import { formatTokens } from '$lib/components/chatturns/chatturns.util';
  import { hardLineBreaks } from '$lib/stores/hardLineBreaks.svelte';
  import SessionMenu from './SessionMenu.svelte';
  import ModelEffort from './ModelEffort.svelte';
  import JobTile from './JobTile.svelte';
  import DeliveryCard from './DeliveryCard.svelte';
  import Attachments from './Attachments.svelte';
  import PromptInspector, { type TokenBreakdown } from './PromptInspector.svelte';
  import RawMessages from './RawMessages.svelte';
  import RawSessionMetadata from './RawSessionMetadata.svelte';
  import { metaLinks } from './metaLinks';
  import PhoneBack from '$lib/shell/PhoneBack.svelte';

  let {
    ctrl,
    row,
    fallbackContext = null,
    railCollapsed,
    onToggleRail,
    onBack,
    onRenameCommit,
    onTopicCommit,
    onComplete,
    onCancel,
    onRestart,
    onPin,
    onUnpin,
    onSetPrimary,
    onCopyId,
    onOpenSession,
    onRetry,
    onDismissAttention,
  }: {
    ctrl: ConversationController;
    row: SessionRow | null;
    /** Durable context truth (session record) for freshly loaded conversations -
     *  session_info frames are live-only, so replay alone never sets timeline.context. */
    fallbackContext?: { tokens: number; limit: number } | null;
    railCollapsed: boolean;
    onToggleRail: () => void;
    /** Phone drilldown: leave the conversation and return to the sessions list. */
    onBack: () => void;
    onRenameCommit: (title: string) => void;
    onTopicCommit: (topic: string) => void;
    onComplete: () => void;
    onCancel: () => void;
    onRestart: () => void;
    onPin: () => void;
    onUnpin: () => void;
    onSetPrimary: () => void;
    onCopyId: () => void;
    onOpenSession: (id: string) => void;
    onRetry: (text: string) => void;
    onDismissAttention: (deliveryId?: string) => void;
  } = $props();

  let scrollEl = $state<HTMLElement>();
  const follow = new ScrollFollow();
  let editing = $state<'title' | 'topic' | null>(null);
  let editDraft = $state('');
  let rawOpen = $state(false);
  let rawMetadataOpen = $state(false);

  const timeline = $derived(ctrl.timeline);
  const title = $derived(row?.title ?? 'New chat');
  const topic = $derived(typeof row?.metadata?.topic === 'string' ? row.metadata.topic : '');
  const sourceType = $derived(row ? sessionSourceType(row) : 'chat');
  const canRestart = $derived(row?.status === 'failed' || row?.status === 'cancelled');
  const canComplete = $derived(row != null && !isFinishedSession(row));
  const canCancel = $derived(ctrl.streaming || (row?.busy ?? false));
  const jobCount = $derived(
    timeline.turns.reduce((n, t) => n + t.blocks.filter((b) => b.kind === 'job').length, 0),
  );
  // The chip is a shortcut to "the jobs this chat spawned", so it carries the
  // board's session filter rather than dropping the user on the whole board.
  const jobsHref = $derived(
    ctrl.sessionId ? buildHash('jobs', { q: `session:${ctrl.sessionId}` }) : '#jobs',
  );

  const links = $derived(metaLinks(row?.metadata));

  function isOutstanding(block: DeliveryBlock): boolean {
    return (
      block.needsAck &&
      !!block.deliveryId &&
      (row?.pending_deliveries ?? []).includes(block.deliveryId)
    );
  }

  // The most recent user prompt, re-sent by the retry affordance on the last turn.
  const lastUserText = $derived.by(() => {
    const t = timeline.turns.findLast((t) => t.role === 'user');
    const p = t?.blocks.find((b) => b.kind === 'prose');
    return p?.kind === 'prose' ? p.text : '';
  });

  // Header status pill. Failed/cancelled sessions map to the session-pill
  // vocabulary's `interrupted` (stopped, kept for the record) with an accurate
  // label; a compaction in flight shows `compacting`. That comes from the row's
  // authoritative flag (served from Session.compacting, kept live by the
  // compaction_started/finished broadcasts) - never from the progress label,
  // whose free-form hook_status text is neither reliable nor trustworthy.
  const pillState = $derived<PillState>(
    ctrl.streaming
      ? 'streaming'
      : row?.compacting
        ? 'compacting'
        : row?.status === 'failed' || row?.status === 'cancelled'
          ? 'interrupted'
          : row?.busy
            ? 'busy'
            : 'idle',
  );
  const pillLabel = $derived(
    pillState === 'interrupted'
      ? (row?.status ?? undefined)
      : pillState === 'idle' && row?.status === 'completed'
        ? 'completed'
        : undefined,
  );
  // Live session_info wins (fresh totals mid-turn); the session record backs it
  // on replay-only loads so the meter and inspector exist without a live turn.
  const ctxView = $derived(timeline.context ?? fallbackContext ?? null);
  const ctxPct = $derived(ctxView ? Math.round((ctxView.tokens / ctxView.limit) * 100) : 0);

  // Latest prompt_snapshot's per-category breakdown, read straight off the event
  // log: the reducer drops prompt_snapshot (not a timeline block), so the
  // context-meter inspector derives it here. `turn`/`at` ride along (the agent
  // now records these durably) so the popover can show how stale the breakdown
  // is. Null until a snapshot lands, which keeps the meter inert (no popover).

  const latestSnapshot = $derived.by<{
    breakdown: TokenBreakdown;
    turn: number | null;
    at: string | null;
  } | null>(() => {
    for (let i = ctrl.events.length - 1; i >= 0; i--) {
      const e = ctrl.events[i]!;
      if (e.type !== 'prompt_snapshot') continue;
      const b = e.token_breakdown;
      if (b && typeof b === 'object' && Array.isArray((b as { categories?: unknown }).categories)) {
        return {
          breakdown: b as TokenBreakdown,
          turn: typeof e.turn === 'number' ? e.turn : null,
          at: typeof e.timestamp === 'string' ? e.timestamp : null,
        };
      }
    }
    return null;
  });

  function compactLabel(c: Compaction): string {
    const counts =
      c.replacedCount != null && c.retainedCount != null
        ? ` · ${c.replacedCount} turns → ${c.retainedCount} kept`
        : '';
    const when = formatAgo(c.at);
    return `context compacted${when ? ` ${when}` : ''}${counts} · summary retained`;
  }
  let openSummary = $state<Record<string, boolean>>({});
  function toggleSummary(id: string) {
    openSummary = { ...openSummary, [id]: !openSummary[id] };
  }
  // Fold disclosures shut when the pane switches sessions (compaction ids can
  // repeat across sessions, so a stale open flag must not leak in).
  $effect(() => {
    void ctrl.sessionId;
    openSummary = {};
  });

  // While a sent turn is in flight and no block is showing its own spinner
  // (tool/code activity), the timeline gets an explicit "waiting on the model"
  // working line - a silent gap after Send reads as a hang.
  const waiting = $derived.by(() => {
    if (!ctrl.working) return false;
    const last = timeline.turns[timeline.turns.length - 1];
    if (!last || last.role !== 'ai') return true;
    if (last.stream) return false;
    return !last.blocks.some(
      (b) =>
        (b.kind === 'exec' && b.status === 'running') ||
        (b.kind === 'code' && b.status === 'running'),
    );
  });
  let waitStart = $state(Date.now());
  $effect(() => {
    if (waiting) waitStart = Date.now();
  });

  // A hook_status tick names what the wait actually is ("Running precommit...");
  // strip the message's own verb/ellipsis so the Work line reads "running
  // hook · precommit" instead of "waiting on the model".
  const liveStatus = $derived.by(() => {
    const last = timeline.turns[timeline.turns.length - 1];
    if (!last || last.role !== 'ai' || !last.streaming) return null;
    const msg = last.liveStatus?.trim();
    if (!msg) return null;
    const m = /^Running\s+(.+?)\.{0,3}$/i.exec(msg);
    return m ? `hook · ${m[1]}` : msg.replace(/\.{3}$/, '');
  });

  function clock(iso: string): string {
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return '';
    return new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  const MONTHS = [
    'jan',
    'feb',
    'mar',
    'apr',
    'may',
    'jun',
    'jul',
    'aug',
    'sep',
    'oct',
    'nov',
    'dec',
  ];
  function dayKey(iso: string): string {
    const d = new Date(Date.parse(iso));
    return Number.isNaN(d.getTime()) ? '' : `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
  }
  function dayLabel(iso: string): string {
    const d = new Date(Date.parse(iso));
    if (Number.isNaN(d.getTime())) return '';
    const today = new Date();
    const isToday = dayKey(iso) === `${today.getFullYear()}-${today.getMonth()}-${today.getDate()}`;
    const date = `${MONTHS[d.getMonth()]} ${d.getDate()}`;
    return isToday ? `today · ${date}` : date;
  }

  // Auto-follow: the ScrollFollow controller sticks to the tail while pinned and
  // unpins only on real user scroll gestures (never from scroll position), so the
  // follow's own catch-up scrolls can't fight the user's touch mid-stream.
  $effect(() => {
    if (!scrollEl) return;
    return follow.attach(scrollEl);
  });
  // Re-run when the timeline grows or a stream frame lands; sync() scrolls to the
  // tail only while pinned (a no-op when the user has scrolled away).
  $effect(() => {
    void timeline.turns.length;
    void ctrl.events.length;
    follow.sync();
  });
  // Re-pin when THIS surface starts a turn it sent (streaming flips false->true
  // only in ConversationController.send - a resync/switch-back never sets it) or
  // when the pane switches to another session (land at its latest). A background
  // resync that merely replaces events must NOT force-pin, so it isn't keyed here.
  let lastSession: string | null = null;
  let wasStreaming = false;
  $effect(() => {
    const id = ctrl.sessionId;
    const streaming = ctrl.streaming;
    const switched = id !== lastSession;
    const started = streaming && !wasStreaming;
    lastSession = id;
    wasStreaming = streaming;
    if (switched || started) follow.repin();
  });
  // A pending ask (approval / question) is a blocking prompt rendered at the tail:
  // when a new one appears - live mid-turn, or already open on a reload/resync -
  // re-pin so it's brought into view instead of sitting below the fold. Keyed off
  // the ask identity so it fires once per prompt, not on every reconcile, and only
  // for an unanswered prompt (the answered record needs no attention).
  let lastAskKey: string | null = null;
  $effect(() => {
    const ask = ctrl.ask;
    const key = ask && !ask.answered ? (ask.askId ?? ask.question) : null;
    if (key && key !== lastAskKey) follow.repin();
    lastAskKey = key;
  });

  function startEdit(which: 'title' | 'topic') {
    editing = which;
    editDraft = which === 'title' ? title : topic;
  }
  function commitEdit() {
    const value = editDraft.trim();
    if (editing === 'title') onRenameCommit(value || 'Untitled session');
    else if (editing === 'topic') onTopicCommit(value);
    editing = null;
  }
  function editKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitEdit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      editing = null;
    }
  }

  // Synthetic "user" turns are daemon injections, not the person: label the
  // gutter by what injected them instead of "you".
  const INJECTED_WHO: Record<string, string> = {
    scheduled_task: 'sched',
    background_task_complete: 'task',
    message_context: 'context',
    environment: 'context',
    client_context: 'context',
  };
  // A client-context item's glyph comes from its provider (a pin for location);
  // fall back to a neutral pin for an item whose provider is gone.
  const contextIcon = (key: string): IconName => contextProvider(key)?.icon ?? 'pin';
  // A short value (location coords) reads fine inline; a long or multiline one
  // (a fetched page, a terminal dump) folds into a collapsed panel instead.
  const isLongContext = (value: string): boolean => value.length > 100 || value.includes('\n');
  function turnWho(turn: (typeof timeline.turns)[number]): string {
    if (turn.role !== 'user') return 'tsugite';
    if (turn.synthetic) return INJECTED_WHO[turn.injected?.[0]?.tag ?? ''] ?? 'context';
    return 'you';
  }

  // Pair each turn with the day divider to show above it (only when the day
  // changes), computed once per timeline rather than mutating a cursor mid-render.
  const turnRows = $derived.by(() => {
    let cursor = '';
    return timeline.turns.map((turn, i) => {
      const key = dayKey(turn.at);
      const day = key && key !== cursor ? dayLabel(turn.at) : null;
      if (key) cursor = key;
      return { turn, i, day };
    });
  });
</script>

<section class="convo" data-testid={TESTID.chatConversation} aria-label="Conversation">
  <header class="convo-hd">
    <PhoneBack {onBack} label="Back to chats" />
    <Button
      variant="ghost"
      size="sm"
      iconOnly
      aria-label={railCollapsed ? 'Show sessions' : 'Hide sessions'}
      aria-pressed={!railCollapsed}
      data-testid={TESTID.chatRailToggle}
      onclick={onToggleRail}
    >
      {#snippet icon()}<Icon name="chat" />{/snippet}
    </Button>

    {#if editing === 'title'}
      <!-- svelte-ignore a11y_autofocus -->
      <input
        class="hd-edit"
        bind:value={editDraft}
        onkeydown={editKeydown}
        onblur={commitEdit}
        aria-label="Rename session"
        autofocus
      />
    {:else}
      <h2 title="Click to rename">
        <button type="button" class="title-btn" onclick={() => startEdit('title')}>{title}</button>
      </h2>
    {/if}

    <span class="t-type" data-k={sourceType}>{sourceType === 'research' ? 'res' : sourceType}</span>
    <Pill st={pillState} label={pillLabel} />
    <ModelEffort sessionId={ctrl.sessionId} />

    <div class="grow"></div>

    {#each links as link (link.key)}
      <a
        class="hd-chip"
        href={link.href}
        target="_blank"
        rel="noreferrer"
        title="{link.key}: {link.href}"
        data-testid={TESTID.chatMetaLink}
      >
        <Icon name="link" size={11} />{link.label}<Icon name="out" size={9} />
      </a>
    {/each}
    {#if jobCount > 0}
      <a class="hd-chip" href={jobsHref} title="{jobCount} job(s) spawned from this session">
        <Icon name="jobs" size={11} />{jobCount} job{jobCount === 1 ? '' : 's'}
      </a>
    {/if}
    {#if ctxView}
      <PromptInspector
        value={ctxView.tokens}
        max={ctxView.limit}
        label="Context {formatTokens(ctxView.tokens)} of {formatTokens(ctxView.limit)} tokens"
        displayText="{formatTokens(ctxView.tokens)}/{formatTokens(ctxView.limit)}"
        warn={ctxPct >= 80}
        breakdown={latestSnapshot?.breakdown ?? null}
        turn={latestSnapshot?.turn ?? null}
        at={latestSnapshot?.at ?? null}
        onViewRaw={() => (rawOpen = true)}
      />
    {/if}
    {#if row}
      <SessionMenu
        pinned={row.pinned}
        isPrimary={row.is_primary}
        {canRestart}
        {canComplete}
        {canCancel}
        onRename={() => startEdit('title')}
        onEditTopic={() => startEdit('topic')}
        {onPin}
        {onUnpin}
        {onSetPrimary}
        {onCopyId}
        onViewMetadata={() => (rawMetadataOpen = true)}
        {onComplete}
        {onCancel}
        {onRestart}
      />
    {/if}
    {#if editing === 'topic'}
      <!-- svelte-ignore a11y_autofocus -->
      <input
        class="convo-topic-edit"
        bind:value={editDraft}
        onkeydown={editKeydown}
        onblur={commitEdit}
        aria-label="Edit topic"
        placeholder="topic"
        autofocus
      />
    {:else if topic}
      <button type="button" class="convo-topic" onclick={() => startEdit('topic')}>{topic}</button>
    {/if}
  </header>

  {#each timeline.compactions as c (c.id)}
    {@const summary = c.summary?.trim() ? c.summary.trim() : null}
    {@const isOpen = openSummary[c.id] ?? false}
    <div class="t-compactbn cbn" role="status" data-testid={TESTID.chatCompactionBanner}>
      <div class="cbn-hd">
        <Icon name="compress" />
        {#if summary}
          <button
            type="button"
            class="cbn-disc"
            aria-expanded={isOpen}
            onclick={() => toggleSummary(c.id)}
          >
            <span class="chev" class:is-open={isOpen}><Icon name="chev-r" size={10} /></span>
            <span class="cbn-txt">{compactLabel(c)}</span>
          </button>
        {:else}
          <span class="cbn-txt">{compactLabel(c)}</span>
        {/if}
        {#if c.sourceId}<button
            type="button"
            class="cbn-link"
            onclick={() => onOpenSession(c.sourceId!)}>view source</button
          >{/if}
      </div>
      {#if summary && isOpen}
        <div class="cbn-summary">{summary}</div>
      {/if}
    </div>
  {/each}

  {#if row?.superseded_by}
    <div class="t-compactbn" role="status">
      <Icon name="fork" />
      <span>this session was compacted into a live successor</span>
      <button type="button" class="cbn-link" onclick={() => onOpenSession(row!.superseded_by!)}
        >continue in live session</button
      >
    </div>
  {/if}

  <div class="convo-scroll" bind:this={scrollEl}>
    {#if ctrl.loading && timeline.turns.length === 0}
      <p class="convo-empty">loading conversation…</p>
    {:else if ctrl.error && timeline.turns.length === 0}
      <p class="convo-empty is-err">{ctrl.error}</p>
    {:else if timeline.turns.length === 0}
      <div class="convo-empty" data-testid={TESTID.chatEmpty}>
        <Icon name="chat" size={22} />
        <p>No messages yet. Say something to start the conversation.</p>
      </div>
    {:else}
      <!-- chats-core: load-earlier affordance (tail-window pagination). Explicit
           click only; ScrollFollow.preserveAcross holds the reading position. -->
      {#if ctrl.hasEarlier}
        <button
          type="button"
          class="load-earlier"
          data-testid={TESTID.chatLoadEarlier}
          disabled={ctrl.loadingEarlier}
          onclick={() => follow.preserveAcross(() => ctrl.loadEarlier())}
        >
          {ctrl.loadingEarlier ? 'loading…' : 'load earlier messages'}
        </button>
      {/if}
      {#each turnRows as { turn, i, day } (turn.id)}
        {#if day}<div class="convo-day">{day}</div>{/if}
        {@const isLastAi = turn.role === 'ai' && i === turnRows.length - 1}
        <!-- A dead-end turn gets the prominent Retry: an errored turn (its error
             block), or the last turn of a session that ended failed/cancelled
             (a cancelled turn carries no error block, so lean on the row status). -->
        {@const failed = turn.blocks.some((b) => b.kind === 'error') || (isLastAi && canRestart)}
        <Msg
          role={turn.role}
          who={turnWho(turn)}
          at={clock(turn.at)}
          index={i + 1}
          streaming={turn.streaming}
          retryFailed={failed}
          onRetry={isLastAi && lastUserText ? () => onRetry(lastUserText) : undefined}
        >
          {#if turn.injected}
            <!-- Context injections (scheduled-task results, environment blocks)
                 fold into panels - never rendered as the person's own words.
                 client_context carries structured items: a short value shows as a
                 compact label:value row, a long/multiline one (a fetched page, a
                 terminal dump) folds into the same collapsed panel the other
                 injections use, so the gutter stays tidy. -->
            {#each turn.injected as inj, ii (ii)}
              {#if inj.items}
                <div class="ctx-inject">
                  {#each inj.items as item (item.key)}
                    {#if isLongContext(item.value)}
                      <div
                        class="ctx-panel"
                        class:is-untrusted={item.untrusted}
                        data-testid={TESTID.chatContextRow(item.key)}
                      >
                        {#if item.untrusted}<span class="ctx-untrusted">untrusted content</span
                          >{/if}
                        <CodeBlock
                          code={item.value}
                          lang="context"
                          filename={item.label}
                          collapsed
                        />
                      </div>
                    {:else}
                      <div
                        class="ctx-row"
                        class:is-untrusted={item.untrusted}
                        data-testid={TESTID.chatContextRow(item.key)}
                      >
                        <Icon name={contextIcon(item.key)} size={12} />
                        <span class="ctx-k">{item.label}</span>
                        <span class="ctx-v">{item.value}</span>
                        {#if item.untrusted}<span class="ctx-untrusted">untrusted</span>{/if}
                      </div>
                    {/if}
                  {/each}
                </div>
              {:else}
                <CodeBlock code={inj.body} lang={inj.tag} filename={inj.id} collapsed />
              {/if}
            {/each}
          {/if}
          {#each turn.blocks as block, bi (bi)}
            {#if block.kind === 'prose'}
              <Prose content={block.text} breaks={turn.role === 'user' && hardLineBreaks.enabled} />
            {:else if block.kind === 'think'}
              <Think content={block.content} tokens={block.tokens} label={block.label} />
            {:else if block.kind === 'exec'}
              <ExecBlock
                command={block.command}
                status={block.status}
                exitCode={block.exitCode}
                output={block.output}
                args={block.args}
                meta={block.meta}
                open={block.status === 'running' || block.status === 'error'}
              />
            {:else if block.kind === 'code'}
              <CodeBlock
                code={block.code}
                lang={block.lang}
                filename={block.filename}
                running={block.status === 'running'}
                collapsed={block.status !== 'running'}
                output={block.output}
                calls={block.calls}
                groups={block.groups}
                returnValue={block.returnValue}
                meta={block.meta}
              />
            {:else if block.kind === 'content'}
              <!-- Named content block (a fence-injected variable): its own panel
                   titled by name, not raw XML inside the prose. -->
              <CodeBlock code={block.text} lang="content" filename={block.name} collapsed />
            {:else if block.kind === 'result'}
              <CodeBlock code={JSON.stringify(block.data, null, 2)} lang="json" collapsed />
            {:else if block.kind === 'error'}
              <div class="t-turnerr" role="alert">
                <Icon name="alert" size={13} />
                <span>{block.message}</span>
              </div>
            {:else if block.kind === 'notice'}
              <!-- A calm informational line (session reset + continued from saved
                   history): muted, never an alert - distinct from the error block. -->
              <div class="t-turnnotice" role="status">
                <Icon name="retry" size={12} />
                <span>{block.message}</span>
              </div>
            {:else if block.kind === 'job'}
              <JobTile job={block.job} />
            {:else if block.kind === 'delivery'}
              <DeliveryCard
                {block}
                outstanding={isOutstanding(block)}
                onDismiss={() => onDismissAttention(block.deliveryId)}
              />
            {/if}
          {/each}
          {#if turn.attachments}
            <!-- Files/photos the person attached, rendered once under their words:
                 images as clickable thumbnails (lightbox), other files as chips. -->
            <Attachments attachments={turn.attachments} />
          {/if}
          {#if turn.stream}
            <!-- Raw token stream, rendered live; the reducer folds it into a
                 prose block (fences stripped) once the model turn settles. An
                 open code fence streams in the real code panel, pinned to its
                 newest lines, instead of an unbounded markdown pre. -->
            {@const streamed = splitStreamFence(turn.stream)}
            {#if streamed.text}<Prose content={streamed.text} />{/if}
            {#if streamed.code != null}
              <CodeBlock code={streamed.code} lang="python" streaming collapsible={false} />
            {/if}
          {/if}
          {#if turn.role === 'ai' && turn.meta?.cacheRead != null}
            <!-- Headline = the LAST step's cached-prefix size, which matches the
                 context meter's scale. The summed reads/writes across the turn's
                 steps live in the tooltip, so a 12-step turn's cross-step total
                 (~700k on a 60k window) no longer reads as the current context.
                 Shown only when a model_response reported cache (never a "0"). -->
            {@const rdTotal = turn.meta.cacheReadTotal ?? turn.meta.cacheRead}
            {@const wrTotal = turn.meta.cacheWriteTotal}
            {@const steps = turn.meta.cacheSteps ?? 1}
            <div
              class="turn-meta mono"
              title={`${formatTokens(rdTotal)} cache reads${
                wrTotal != null ? ` / ${formatTokens(wrTotal)} writes` : ''
              } across ${steps} step${steps === 1 ? '' : 's'}`}
            >
              {formatTokens(turn.meta.cacheRead)} cached
            </div>
          {/if}
        </Msg>
      {/each}

      {#if waiting}
        <!-- No Stop here: the composer's Stop is the single cancel control. -->
        <div class="work-wrap">
          <Work
            verb={liveStatus ? 'running' : 'waiting'}
            operation={liveStatus ?? 'on the model'}
            startedAt={waitStart}
          />
        </div>
      {/if}

      {#if ctrl.ask}
        <div class="ask-wrap">
          <Ask
            question={ctrl.ask.question}
            questionType={ctrl.ask.questionType}
            options={ctrl.ask.options}
            heading={ctrl.ask.questionType === 'approval' ? 'Approval required' : 'Question'}
            alert={ctrl.ask.questionType === 'yes_no' || ctrl.ask.questionType === 'approval'}
            resolution={ctrl.ask.answered
              ? { tone: 'approved', text: `answered · ${ctrl.ask.answer}` }
              : null}
            onAnswer={(value) => ctrl.answerAsk(value)}
          />
        </div>
      {/if}
    {/if}

    {#if ctrl.localEcho.length > 0}
      <!-- Ephemeral slash-command echoes (local-only: not persisted, not sent to
           the model). Rendered at the tail as the most-recent activity and cleared
           when the controller reopens a session. Its own channel, outside the
           event log / timeline, so a resync never purges or doubles it. -->
      <div class="echo-wrap" data-testid={TESTID.chatLocalEcho}>
        {#each ctrl.localEcho as echo (echo.id)}
          <LocalEcho
            command={echo.command}
            output={echo.output}
            ok={echo.ok}
            action={echo.action}
          />
        {/each}
      </div>
    {/if}
  </div>

  {#if !follow.pinned}
    <button type="button" class="jumplive" onclick={() => follow.repin()}>
      <Icon name="down" size={11} />following paused — jump to live
    </button>
  {/if}

  {#if rawOpen && ctrl.sessionId}
    <RawMessages sessionId={ctrl.sessionId} onClose={() => (rawOpen = false)} />
  {/if}
  {#if rawMetadataOpen && row}
    <RawSessionMetadata metadata={row.metadata} {links} onClose={() => (rawMetadataOpen = false)} />
  {/if}
</section>

<style>
  .convo {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    flex: 1;
    background: var(--bg0);
    position: relative;
  }
  .convo-hd {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 14px;
    border-bottom: 1px solid var(--bd0);
    flex: none;
    min-width: 0;
    flex-wrap: wrap;
  }
  .convo-hd h2 {
    margin: 0;
    min-width: 0;
  }
  .title-btn {
    background: none;
    border: 0;
    padding: 1px 3px;
    border-radius: 3px;
    cursor: text;
    color: var(--tx0);
    font: 600 var(--fs-lg) / 1.2 var(--font-ui);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 42ch;
  }
  .title-btn:hover {
    background: var(--bg2);
  }
  .hd-edit {
    flex: 1;
    min-width: 12ch;
    height: 26px;
    background: var(--bg1);
    border: 1px solid var(--acc);
    border-radius: var(--r-sm);
    padding: 0 8px;
    color: var(--tx0);
    font: 600 var(--fs-md) var(--font-ui);
  }
  .hd-edit:focus {
    outline: none;
  }
  .t-type {
    display: inline-block;
    font: 600 var(--fs-2xs) / 1.5 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0 4px;
    border-radius: 3px;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 15%, transparent);
    flex: none;
  }
  .t-type[data-k='code'] {
    --c: var(--acc);
  }
  .t-type[data-k='ops'] {
    --c: var(--st-warn);
  }
  .t-type[data-k='research'] {
    --c: var(--st-queue);
  }
  .t-type[data-k='chat'] {
    /* tx2, not st-mute: --c renders as small text on a tinted chip (see
       SessionRow's matching rule). */
    --c: var(--tx2);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .hd-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    text-decoration: none;
    white-space: nowrap;
  }
  .hd-chip:hover {
    color: var(--acc);
  }
  .convo-topic {
    flex-basis: 100%;
    min-width: 0;
    text-align: left;
    background: none;
    border: 0;
    font: 400 var(--fs-xs) / 1.4 var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: text;
    padding: 1px 3px;
    border-radius: 3px;
  }
  .convo-topic::before {
    content: 'topic: ';
    opacity: 0.65;
  }
  .convo-topic:hover {
    background: var(--bg2);
    color: var(--tx2);
  }
  .convo-topic-edit {
    flex-basis: 100%;
    height: 24px;
    background: var(--bg1);
    border: 1px solid var(--acc);
    border-radius: var(--r-sm);
    padding: 0 6px;
    color: var(--tx1);
    font: 400 var(--fs-xs) var(--font-mono);
  }
  .convo-topic-edit:focus {
    outline: none;
  }
  /* t-compactbn - compaction / supersession banner. */
  .t-compactbn {
    display: flex;
    align-items: center;
    gap: 8px;
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx2);
    background: color-mix(in oklab, var(--st-warn) 8%, transparent);
    border-block: 1px solid color-mix(in oklab, var(--st-warn) 22%, transparent);
    padding: 5px 16px;
    flex: none;
  }
  .cbn-link {
    margin-left: auto;
    background: none;
    border: 0;
    padding: 0;
    color: var(--acc);
    font: inherit;
    cursor: pointer;
    text-decoration: none;
  }
  .cbn-link:hover {
    text-decoration: underline;
  }
  /* Compaction banner with an expandable summary: the header stays the row the
     base .t-compactbn draws; .cbn stacks the revealed summary beneath it. */
  .cbn {
    flex-direction: column;
    align-items: stretch;
    gap: 4px;
  }
  .cbn-hd {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .cbn-disc {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    background: none;
    border: 0;
    padding: 0;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .cbn-disc:hover {
    color: var(--tx1);
  }
  .cbn-txt {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cbn .chev {
    display: inline-flex;
    flex: none;
    transition: rotate var(--t-2) var(--ease);
  }
  .cbn .chev.is-open {
    rotate: 90deg;
  }
  .cbn-summary {
    max-height: 220px;
    overflow-y: auto;
    padding: 2px 0 4px 26px;
    color: var(--tx3);
    font: 400 var(--fs-xs) / 1.55 var(--font-ui);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  @media (prefers-reduced-motion: reduce) {
    .cbn .chev {
      transition: none;
    }
  }
  .convo-scroll {
    flex: 1;
    overflow-y: auto;
    /* The pane itself never scrolls sideways: genuinely wide blocks
       (code, tables) scroll inside their own overflow-x container; everything else
       wraps. Without this, overflow-y:auto coerces overflow-x to auto too. */
    overflow-x: hidden;
    overscroll-behavior: contain;
    scroll-padding-bottom: 40px;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .convo-day {
    text-align: center;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    padding: 12px 0 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  /* chats-core: load-earlier affordance, muted like the day divider. */
  .load-earlier {
    align-self: center;
    margin: 10px 0 4px;
    background: none;
    border: 0;
    padding: 4px 10px;
    border-radius: var(--r-full);
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.04em;
    cursor: pointer;
  }
  .load-earlier:hover:not(:disabled) {
    color: var(--acc);
    background: var(--bg2);
  }
  .load-earlier:disabled {
    cursor: default;
    opacity: 0.6;
  }
  .turn-meta {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    opacity: 0.85;
    cursor: default;
    align-self: start;
  }
  /* Client-context gutter: attached location/... shown as muted label:value rows
     above the user's message, distinct from the person's own words. */
  .ctx-inject {
    display: flex;
    flex-direction: column;
    gap: 3px;
    align-self: start;
    max-width: 100%;
    margin-bottom: 2px;
  }
  /* A long context item folds into a collapsed panel; keep it inside the gutter
     so a wide value never pushes the conversation sideways. */
  .ctx-panel {
    max-width: 100%;
    min-width: 0;
  }
  /* Untrusted external content (a fetched page): a visible tag so the reader
     knows the model was told to treat it as data, not instructions. */
  .ctx-untrusted {
    align-self: flex-start;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--st-warn);
    background: color-mix(in oklab, var(--st-warn) 14%, transparent);
    padding: 2px 6px;
    border-radius: 4px;
    flex: none;
  }
  .ctx-row {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx3);
  }
  .ctx-row .ctx-k {
    color: var(--tx2);
  }
  .ctx-row .ctx-k::after {
    content: ':';
  }
  .ctx-row .ctx-v {
    color: var(--tx1);
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .convo-empty {
    margin: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 40px 20px;
    text-align: center;
    color: var(--tx3);
    font: 400 var(--fs-sm) var(--font-ui);
  }
  .convo-empty.is-err {
    color: var(--st-err);
  }
  .t-turnerr {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 11px;
    border: 1px solid color-mix(in oklab, var(--st-err) 32%, transparent);
    background: color-mix(in oklab, var(--st-err) 10%, transparent);
    border-radius: var(--r-md);
    color: var(--st-err);
    font: 500 var(--fs-sm) var(--font-mono);
  }
  /* A calm session-reset notice: neutral surface + muted text, hugging its
     content so it reads as a quiet annotation, not the full-width red error bar. */
  .t-turnnotice {
    justify-self: start;
    max-width: 100%;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 10px;
    border: 1px solid var(--bd0);
    background: var(--bg1);
    border-radius: var(--r-md);
    color: var(--tx2);
    font: 400 var(--fs-xs) / 1.5 var(--font-mono);
  }
  .t-turnnotice :global(svg) {
    flex: none;
    color: var(--tx3);
  }
  .ask-wrap {
    padding: 2px 18px 10px;
  }
  .work-wrap {
    padding: 4px 18px 10px;
  }
  /* Ephemeral slash-command echoes, stacked at the conversation tail. Left-inset
     to sit under the turns without impersonating one. */
  .echo-wrap {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 4px 18px 12px 14px;
    min-width: 0;
  }
  .jumplive {
    position: absolute;
    bottom: 14px;
    left: 50%;
    translate: -50% 0;
    z-index: 5;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    color: var(--tx0);
    font: 500 var(--fs-xs) var(--font-mono);
    padding: 5px 11px;
    border-radius: var(--r-full);
    box-shadow: var(--sh-2);
    cursor: pointer;
  }

  /* Narrow: shed ambient chrome, keep state truth. */
  @media (max-width: 640px) {
    .convo-hd {
      padding: 7px 10px;
      gap: 7px;
    }
    /* Phone drilldown: the conversation is a screen reached from the list, so the
       header's PhoneBack shows and the desktop rail-collapse toggle drops. */
    .convo-hd :global([data-testid='chat-rail-toggle']) {
      display: none;
    }
    .convo-hd .hd-chip,
    .convo-hd :global(.ctx-anchor) {
      display: none;
    }
    .t-compactbn {
      padding: 4px 12px;
      font-size: var(--fs-2xs);
    }
  }
</style>
