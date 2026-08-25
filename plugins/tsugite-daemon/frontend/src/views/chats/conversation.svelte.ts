/**
 * Conversation controller: the stateful core behind the middle pane. Owns the
 * open session's event log (replay from GET /api/sessions/{id}/events, then live
 * frames from the per-chat SSE stream), the streaming flag, and a pending
 * ask_user. `timeline` is a $derived fold over the combined log (turns.ts), so
 * the view re-renders as frames land without any imperative DOM work.
 *
 * Own-tab vs broadcast split: the live per-chat stream carries the turn-end and
 * streaming frames the cross-session broadcast withholds, so this - the surface
 * that ran the send - is the only place they're applied. ask_user is held out of
 * the event log (it isn't a timeline block) and rendered as a live prompt.
 *
 * A mutated $state class instance, never a reassigned binding (AGENTS.md).
 */
import { api } from '$lib/api/client';
import { auth } from '$lib/stores/auth.svelte';
import {
  sendChat,
  cancelChat,
  respondToAsk,
  type AskAnswer,
  type ChatFrame,
  type ChatStreamHandle,
} from '$lib/api/chat';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { buildTimeline, type Timeline } from './turns';

type Event = Record<string, unknown>;

export interface AskState {
  /** Stable id from the durable ask_user event: targets the answer and lets the
   *  prompt be reconciled from the persisted log after a reload. Absent for a
   *  legacy live-only ask or the id-less jsonl backend. */
  askId?: string;
  question: string;
  questionType: 'text' | 'yes_no' | 'choice' | 'approval';
  options: string[];
  answered: boolean;
  answer: string;
}

/** True when two ask ids denote the same prompt, treating a missing id as a
 *  wildcard (legacy asks and the id-less jsonl backend carry none). */
function sameAsk(a: string | undefined, b: string | undefined): boolean {
  return a == null || b == null || a === b;
}

const ASK_ORDER = ['session_complete', 'session_end', 'final_result'];

/** Events fetched per page: the tail window on open and each "load earlier"
 *  chunk. Large enough that most conversations load whole (no affordance shown),
 *  small enough that a fat session's open ships a page instead of the whole log. */
const WINDOW = 200;

/** Compaction rewrites the log's shape (turns replaced by a summary), so a delta
 *  carrying one can't be appended as a plain suffix - the window is reloaded. */
const COMPACTION_TYPES = new Set(['compaction', 'compaction_started', 'compacted']);

function isCompactionEvent(e: Event): boolean {
  return COMPACTION_TYPES.has(String(e.type));
}

/** The largest numeric storage id present, or null when none carry one (a fresh
 *  session, or the id-less jsonl backend). It's the cursor for the incremental
 *  resync delta. */
function maxEventId(events: Event[]): number | null {
  let max: number | null = null;
  for (const e of events) {
    const id = e.id;
    if (typeof id === 'number' && (max === null || id > max)) max = id;
  }
  return max;
}

/** One page from GET /api/sessions/{id}/events with windowing params. */
interface EventsPage {
  events: Event[];
  hasMore: boolean;
  oldestId: number | null;
}

interface PageOpts {
  afterId?: number | null;
  beforeId?: number | null;
  limit?: number | null;
}

function nowIso(): string {
  return new Date().toISOString();
}

/** The text a user_input event shows as the person's words. The server splits
 *  context injections into `display_text`, so prefer it; a plain turn (and the
 *  optimistic bubble) carries only `text`. Used to match a persisted / echoed
 *  user_input against the optimistic bubble so the message is never doubled. */
function userText(e: Event): string | undefined {
  const dt = e.display_text;
  if (typeof dt === 'string') return dt;
  const t = e.text;
  return typeof t === 'string' ? t : undefined;
}

export interface SendOpts {
  reasoningEffort?: string;
  uploadedFiles?: { name: string }[];
  /** Structured client-context items to send as metadata (never in the text). */
  contextMetadata?: { key: string; label: string; value: string }[];
}

/** One inline slash-command echo (Claude-Code style): the command line and its
 *  output rendered in the conversation, local-only - never persisted to history
 *  and never sent to the model. Lives in its own channel (never `events`), so a
 *  resync/mergeDelta that purges id-less placeholders leaves it untouched; the
 *  controller is per-session-view, so a switch/reload wipes it. */
export interface LocalEcho {
  id: string;
  command: string;
  output: string;
  ok: boolean;
  ts: string;
  /** Optional navigation affordance reproduced in the echo (e.g. /job's
   *  "Open jobs" link), replacing the toast's action button. */
  action?: { label: string; href: string };
}

export class ConversationController {
  sessionId = $state<string | null>(null);
  events = $state<Event[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);
  streaming = $state(false);
  /** Server-side "a turn is in flight" truth for the open session, mirrored in
   *  by the view from the session row (kept live by the cross-session broadcast).
   *  Lets the working indicator survive a switch-away/back that drops the local
   *  per-chat stream - that stream belongs to the surface that ran the send, so
   *  `streaming` alone goes false on switch and never comes back. */
  serverBusy = $state(false);
  ask = $state<AskState | null>(null);
  /** Older events exist before the loaded window - the "load earlier" affordance
   *  shows while true. */
  hasEarlier = $state(false);
  /** True while a "load earlier" page is in flight, so the affordance can't
   *  double-fire and the view can show a pending state. */
  loadingEarlier = $state(false);
  /** Smallest storage id in the loaded window: the cursor the next "load earlier"
   *  page reads before. Null when the window holds the whole log or the backend
   *  assigns no ids. */
  oldestId = $state<number | null>(null);
  /** Largest storage id the controller holds: the cursor for the incremental
   *  resync delta (fetch only events newer than this). */
  private lastEventId: number | null = null;
  /** Messages queued while a turn streams; flushed one per turn end. */
  queued = $state<{ text: string; opts: SendOpts }[]>([]);
  /** Set when a send fails before any frame arrives (409 busy, daemon down):
   *  the composer restores this text so the message isn't lost. `seq` makes
   *  each failure a distinct value for the consuming effect. */
  sendFailed = $state<{ text: string; seq: number } | null>(null);
  private failSeq = 0;
  /** Ephemeral slash-command echoes (see LocalEcho). Its own channel, never the
   *  event log: resync/mergeDelta/setWindow/reloadWindow all operate on `events`
   *  alone and leave this alone, so echoes survive live resyncs but reset when the
   *  controller reopens a session (open() clears it). */
  localEcho = $state<LocalEcho[]>([]);
  private echoSeq = 0;

  private handle: ChatStreamHandle | null = null;
  private loadToken = 0;

  readonly timeline: Timeline = $derived(buildTimeline(this.events));

  /** Whether to show a working indicator for the open session. True while this
   *  surface streams a turn it sent (`streaming`), OR the server reports the
   *  session busy (a turn is running - started here before a switch-away/back,
   *  or in another surface). Suppressed once the replayed timeline shows the
   *  latest turn closed, so a busy flag that lags the turn-end broadcast can't
   *  strand a spinner after the reply has already rendered. */
  readonly working: boolean = $derived.by(() => {
    if (this.streaming) return true;
    if (!this.serverBusy) return false;
    const turns = this.timeline.turns;
    const last = turns[turns.length - 1];
    return !last || last.role !== 'ai' || last.streaming === true;
  });

  /** Open a session in the pane: stop any live stream, then replay its history.
   *  A null id clears to the empty state (composer creates a session on send). */
  async open(sessionId: string | null): Promise<void> {
    if (sessionId === this.sessionId) return;
    this.closeStream();
    this.sessionId = sessionId;
    this.events = [];
    this.ask = null;
    this.error = null;
    this.streaming = false;
    this.serverBusy = false;
    this.queued = [];
    this.sendFailed = null;
    this.localEcho = [];
    this.hasEarlier = false;
    this.loadingEarlier = false;
    this.oldestId = null;
    this.lastEventId = null;
    if (!sessionId) return;
    const token = ++this.loadToken;
    this.loading = true;
    try {
      // Load only the newest window; older turns arrive on demand via loadEarlier.
      const page = await this.fetchPage(sessionId, { limit: WINDOW });
      if (token !== this.loadToken) return; // superseded by a newer open()
      this.setWindow(page);
    } catch (err) {
      if (token === this.loadToken) this.error = err instanceof Error ? err.message : String(err);
    } finally {
      if (token === this.loadToken) this.loading = false;
    }
  }

  /** Prepend the chunk of events immediately before the loaded window. Explicit
   *  (the "load earlier" affordance), never triggered by scroll; the caller
   *  preserves scroll position across the prepend. */
  async loadEarlier(): Promise<void> {
    const sessionId = this.sessionId;
    if (!sessionId || !this.hasEarlier || this.oldestId == null || this.loadingEarlier) return;
    const cursor = this.oldestId;
    this.loadingEarlier = true;
    try {
      const page = await this.tryFetchPage(sessionId, { beforeId: cursor, limit: WINDOW });
      // A switch or a fresher cursor (a concurrent prepend) landed: discard.
      if (!page || this.sessionId !== sessionId || this.oldestId !== cursor) return;
      this.events = [...page.events, ...this.events];
      this.hasEarlier = page.hasMore;
      if (page.oldestId != null) this.oldestId = page.oldestId;
    } finally {
      this.loadingEarlier = false;
    }
  }

  /** Send a turn. Returns the session id used (created on the fly when the pane
   *  was empty), so the caller can select the fresh session. */
  async send(text: string, opts: SendOpts = {}): Promise<string | null> {
    let sessionId = this.sessionId;
    if (!sessionId) {
      const res = await api.post<{ id: string }>('/api/chat/sessions/new', {
        user_id: auth.userId,
      });
      sessionId = res.id;
      this.sessionId = sessionId;
    }
    // Optimistic user turn - replay will carry the persisted copy on next open.
    // Tagged with a client key: `$state` proxies array elements, so removing it
    // on failure must match by property, never by object identity.
    const clientKey = `opt-${++this.failSeq}`;
    // Render the attached-context gutter immediately: seed the optimistic bubble's
    // `injected` from the same context_metadata being sent, so a chip you added
    // (location, terminal output, ...) shows on the turn now instead of waiting
    // for resync. Server-detected items aren't known here - the daemon streams
    // those in as a user_context frame (applyFrame folds them into this bubble).
    const injected = opts.contextMetadata?.length
      ? [{ tag: 'client_context', items: opts.contextMetadata }]
      : undefined;
    this.events = [
      ...this.events,
      // With injected blocks present the reducer reads `display_text` (not `text`)
      // for the person's words, and treats an empty one as a pure-injection turn;
      // the typed text IS the display text here, so set both.
      {
        type: 'user_input',
        text,
        timestamp: nowIso(),
        clientKey,
        ...(injected ? { injected, display_text: text } : {}),
      },
    ];
    this.ask = null;
    this.streaming = true;
    this.sendFailed = null;
    let gotFrame = false;
    this.handle = sendChat(
      {
        message: text,
        userId: auth.userId,
        sessionId,
        ...(opts.reasoningEffort ? { reasoningEffort: opts.reasoningEffort } : {}),
        ...(opts.uploadedFiles?.length ? { uploadedFiles: opts.uploadedFiles } : {}),
        ...(opts.contextMetadata?.length ? { contextMetadata: opts.contextMetadata } : {}),
      },
      {
        onEvent: (frame) => {
          gotFrame = true;
          this.applyFrame(frame);
        },
        // A send failure is non-destructive: keep the existing timeline (never
        // blank the conversation the user was reading). Once frames arrived, the
        // turn ran and any error is already an inline error block - a "Send failed"
        // toast would just double it. So this only fires for a PRE-frame failure
        // (409 busy, daemon down): nothing rendered inline, the turn never took,
        // so toast it, drop the optimistic bubble, and hand the text back.
        onError: (err) => {
          if (gotFrame) return;
          const msg = err instanceof Error ? err.message : String((err as ChatFrame).error);
          toasts.push('err', 'Send failed', { body: msg });
          this.events = this.events.filter((e) => e.clientKey !== clientKey);
          this.sendFailed = { text, seq: ++this.failSeq };
        },
        // The request WAS delivered (a 200 opened the stream) and ran server-side;
        // only the response feed died - the classic mobile background/foreground.
        // Not a send failure: no toast, no draft restore. The local stream is gone
        // for good, so reconcile from the persisted log (the turn may have finished
        // while away) instead of stranding the optimistic bubble.
        onStreamLost: () => {
          this.streaming = false;
          // A prompt you already answered must not survive the feed dying (mobile
          // background): no further frame will arrive to clear it.
          if (this.ask?.answered) this.ask = null;
          void this.reconcileAfterSend(clientKey, text);
        },
        onDone: () => {
          this.streaming = false;
          if (this.ask?.answered) this.ask = null;
          this.handle = null;
          this.flushQueue();
        },
      },
    );
    return sessionId;
  }

  /** Record a slash-command result as an inline echo (replaces the result toast).
   *  Appends to the ephemeral `localEcho` channel - never `events` - so it renders
   *  in the conversation without being persisted or sent to the model. */
  pushEcho(command: string, output: string, ok: boolean, action?: LocalEcho['action']): void {
    this.localEcho = [
      ...this.localEcho,
      {
        id: `echo-${++this.echoSeq}`,
        command,
        output,
        ok,
        ts: nowIso(),
        ...(action ? { action } : {}),
      },
    ];
  }

  /** Park a message typed mid-turn; it sends when the current turn finishes. */
  queue(text: string, opts: SendOpts = {}): void {
    this.queued = [...this.queued, { text, opts }];
  }

  unqueue(index: number): void {
    this.queued = this.queued.filter((_, i) => i !== index);
  }

  /** Send the next queued message once the turn ends. One per turn end - its
   *  own onDone flushes the next. Held while an ask_user prompt is open (the
   *  turn is parked on the user, not finished). */
  private flushQueue(): void {
    const next = this.queued[0];
    if (!next || (this.ask && !this.ask.answered)) return;
    this.queued = this.queued.slice(1);
    void this.send(next.text, next.opts);
  }

  private applyFrame(frame: ChatFrame): void {
    if (frame.type === 'ask_user') {
      const askId = typeof frame.ask_id === 'string' ? frame.ask_id : undefined;
      this.ask = {
        ...(askId ? { askId } : {}),
        question: String(frame.question ?? ''),
        questionType: (frame.question_type as AskState['questionType']) ?? 'text',
        options: Array.isArray(frame.options) ? (frame.options as string[]) : [],
        answered: false,
        answer: '',
      };
      return;
    }
    if (frame.type === 'user_context') {
      // Live-only hint: fold the sending turn's server context split (attached +
      // detected items) into the in-flight optimistic bubble so its gutter shows
      // during streaming. Enriches the existing bubble - never appends - so it
      // can't double the user turn; resync later replaces it with the recorded one.
      const injected = frame.injected;
      if (Array.isArray(injected)) {
        for (let i = this.events.length - 1; i >= 0; i--) {
          const e = this.events[i];
          if (e?.clientKey != null && e.type === 'user_input') {
            // Keep the person's words visible: with injected present the reducer
            // reads display_text, so carry text into it (an empty one would render
            // the turn as a pure-injection "context" row).
            this.events = this.events.map((ev, j) =>
              j === i ? { ...ev, injected, display_text: ev.display_text ?? ev.text } : ev,
            );
            break;
          }
        }
      }
      return;
    }
    // Clear the prompt once it is done with the turn. An ANSWERED prompt clears on
    // the next frame the turn produces: an approval whose turn ends in an error, a
    // cancel, or a lost mobile stream never emits a final_result, so keying only off
    // ASK_ORDER left it stuck at the bottom forever. An UNANSWERED prompt clears only
    // when the turn itself ends (session_end / final_result), never on a stray frame,
    // since the turn is genuinely parked on the user.
    if (this.ask?.answered || ASK_ORDER.includes(frame.type)) this.ask = null;
    this.events = [...this.events, frame as Event];
  }

  /** Stop the agent server-side (the Stop button). The stream close is separate:
   *  aborting the fetch would leave the daemon running the turn. */
  async stop(): Promise<void> {
    if (!this.sessionId) return;
    await cancelChat({ userId: auth.userId, sessionId: this.sessionId });
  }

  /** Submit an answer to the open ask_user prompt. Targets the durable ask by id
   *  and branches on the daemon's reply so a failure is never silent (the view
   *  calls this fire-and-forget). A thrown error (network / 500) leaves the prompt
   *  live so it can be retried; an `expired` reply drops it (no longer answerable);
   *  only a clean accept records the inert "answered" state. */
  async answerAsk(value: string): Promise<void> {
    const ask = this.ask;
    if (!ask || !this.sessionId) return;
    let res: AskAnswer;
    try {
      res = await respondToAsk({
        askId: ask.askId,
        response: value,
        userId: auth.userId,
        sessionId: this.sessionId,
      });
    } catch (err) {
      toasts.push('err', 'Could not send answer', {
        body: err instanceof Error ? err.message : String(err),
      });
      return; // keep the prompt live: the answer never landed
    }
    if (res?.status === 'expired') {
      this.ask = null;
      toasts.push('info', 'This prompt is no longer active');
      return;
    }
    // Guard against a frame having cleared/replaced the prompt during the await.
    if (this.ask === ask) this.ask = { ...ask, answered: true, answer: value };
  }

  /** Re-derive the pending prompt from the durable event log so a reload / resync
   *  re-shows a still-open ask_user and drops one the log shows answered. The live
   *  per-chat stream owns the prompt while a turn streams (applyFrame), so this
   *  runs only on the replay/resync loaders, which never fire mid-stream. An
   *  optimistic "answered" overlay for the same ask survives the reconcile (the
   *  answer POST landed; the turn just hasn't produced its next frame yet). */
  private reconcileAsk(): void {
    const pending = this.timeline.pendingAsk;
    if (!pending) {
      this.ask = null;
      return;
    }
    const cur = this.ask;
    const same = cur != null && sameAsk(cur.askId, pending.askId);
    this.ask = {
      ...(pending.askId ? { askId: pending.askId } : {}),
      question: pending.question,
      questionType: pending.questionType,
      options: pending.options,
      answered: same ? cur.answered : false,
      answer: same ? cur.answer : '',
    };
  }

  /** Non-destructive in-place refresh of the open session's events. Used on
   *  foreground/reconnect resume, where the per-chat send stream is gone and the
   *  turn-end frames were withheld from the broadcast: reload the persisted log so
   *  the timeline catches up (a finished reply lands, a still-running turn keeps
   *  the serverBusy working line). No blanking first, so the thread never flashes;
   *  a no-op while a live stream is delivering (that stream is the fresher truth). */
  async resync(): Promise<void> {
    const sessionId = this.sessionId;
    if (!sessionId || this.streaming) return;
    const cursor = this.lastEventId;
    // No numeric cursor (a fresh/empty session, or the id-less jsonl backend):
    // there's nothing to diff against, so reload the tail window wholesale.
    if (cursor == null) {
      await this.reloadWindow(sessionId);
      return;
    }
    const page = await this.tryFetchPage(sessionId, { afterId: cursor });
    // Re-check: a session switch or a new send may have started during the fetch.
    if (!page || this.sessionId !== sessionId || this.streaming) return;
    const delta = page.events;
    if (delta.length === 0) return; // nothing newer - leave the timeline as is
    // A compaction in the gap reshapes the log; reload the window rather than
    // append a suffix whose turn boundaries the compaction moved.
    if (delta.some(isCompactionEvent)) {
      await this.reloadWindow(sessionId);
      return;
    }
    this.mergeDelta(delta);
  }

  /** Reconcile the timeline after a delivered send lost its response stream.
   *  Reloads the tail window; if the daemon has already recorded this turn's
   *  user_input, the reload replaces the optimistic bubble outright, otherwise the
   *  optimistic bubble is kept appended so the just-sent message never vanishes
   *  (it WAS delivered - a 200 opened the stream - just not recorded yet). */
  private async reconcileAfterSend(clientKey: string, text: string): Promise<void> {
    const sessionId = this.sessionId;
    if (!sessionId) return;
    const optimistic = this.events.find((e) => e.clientKey === clientKey);
    const page = await this.tryFetchPage(sessionId, { limit: WINDOW });
    // Reload failed (still offline) or the surface moved on: keep the timeline as
    // is - the message stays put, and the next resume/reconnect retries.
    if (!page || this.sessionId !== sessionId || this.streaming) return;
    // Recorded == the fetched window already holds a server-side (id-bearing, no
    // clientKey) user_input with this text. A count comparison across tail windows
    // is unreliable: a long session's tail can drop an older user_input as the new
    // turn's events push it out, so the count looks unchanged even though the
    // message WAS recorded - and the optimistic bubble then survives beside its
    // persisted copy, rendering the message twice. Match by text so the bubble is
    // dropped the moment any server copy lands.
    const recorded = page.events.some(
      (e) => e.type === 'user_input' && e.clientKey == null && userText(e) === text,
    );
    this.setWindow(page, recorded || !optimistic ? [] : [optimistic]);
  }

  /** Route a cross-session `session_event` broadcast into the open conversation
   *  so a turn started elsewhere (another tab, a schedule, /model from Discord)
   *  grows the timeline live. A no-op while THIS surface streams the turn (its
   *  own per-chat stream is authoritative - the broadcast echo would double it)
   *  or when the frame is for another session. The id-less synthesized event is
   *  reconciled against the persisted log by the next resync (which drops id-less
   *  placeholders and re-appends the authoritative, id-bearing copies). */
  ingestBroadcast(data: Record<string, unknown>): void {
    if (data.session_id !== this.sessionId || this.streaming) return;
    const type = data.event_type;
    if (typeof type !== 'string' || !type) return;
    const { event_type: _et, session_id: _sid, ...rest } = data;
    const event = { ...rest, type } as Event;
    // Never double the just-sent message: user_input is NOT in the broadcast skip
    // set, so a late server echo of a message this surface still holds as an
    // optimistic (clientKey) bubble can land after the stream closed (streaming
    // false). Drop it - the optimistic bubble already renders it, and the next
    // resync reconciles ids.
    if (type === 'user_input') {
      const text = userText(event);
      const echoesOptimistic = this.events.some(
        (e) => e.clientKey != null && e.type === 'user_input' && userText(e) === text,
      );
      if (echoesOptimistic) return;
    }
    this.events = [...this.events, event];
    // A broadcast ask_user / ask_answered (a turn driven from another surface)
    // moves the pending prompt on this observing surface too.
    if (type === 'ask_user' || type === 'ask_answered') this.reconcileAsk();
  }

  /** Replace the window with a freshly fetched page, optionally keeping trailing
   *  id-less placeholders (an unrecorded optimistic bubble). Resets the earlier
   *  cursor + affordance and re-derives the resync cursor. */
  private setWindow(page: EventsPage, keep: Event[] = []): void {
    this.events = [...page.events, ...keep];
    this.hasEarlier = page.hasMore;
    this.oldestId = page.oldestId;
    this.lastEventId = maxEventId(page.events);
    this.reconcileAsk();
  }

  /** Reload the tail window in place (used when a delta can't be trusted as a
   *  pure suffix). Guards a mid-fetch switch/send. */
  private async reloadWindow(sessionId: string): Promise<void> {
    const page = await this.tryFetchPage(sessionId, { limit: WINDOW });
    if (!page || this.sessionId !== sessionId || this.streaming) return;
    this.setWindow(page);
  }

  /** Append a forward delta, replacing any id-less placeholders (live broadcast
   *  frames, a lost-stream optimistic bubble) with the authoritative id-bearing
   *  prefix so nothing double-renders. */
  private mergeDelta(delta: Event[]): void {
    const authoritative = this.events.filter((e) => typeof e.id === 'number');
    this.events = [...authoritative, ...delta];
    this.lastEventId = maxEventId(this.events);
    this.reconcileAsk();
  }

  private eventsPath(sessionId: string, opts: PageOpts): string {
    const params = new URLSearchParams();
    if (opts.afterId != null) params.set('after_id', String(opts.afterId));
    if (opts.beforeId != null) params.set('before_id', String(opts.beforeId));
    if (opts.limit != null) params.set('limit', String(opts.limit));
    const qs = params.toString();
    return `/api/sessions/${encodeURIComponent(sessionId)}/events${qs ? `?${qs}` : ''}`;
  }

  private async fetchPage(sessionId: string, opts: PageOpts): Promise<EventsPage> {
    const res = await api.get<{ events: Event[]; has_more?: boolean; oldest_id?: number | null }>(
      this.eventsPath(sessionId, opts),
    );
    return { events: res.events, hasMore: res.has_more ?? false, oldestId: res.oldest_id ?? null };
  }

  private async tryFetchPage(sessionId: string, opts: PageOpts): Promise<EventsPage | null> {
    try {
      return await this.fetchPage(sessionId, opts);
    } catch {
      return null;
    }
  }

  /** Stop reading the local stream (does not stop the agent). Called on session
   *  switch and teardown. */
  closeStream(): void {
    this.handle?.close();
    this.handle = null;
    this.streaming = false;
  }
}
