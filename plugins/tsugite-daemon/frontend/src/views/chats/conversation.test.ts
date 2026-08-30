import { beforeEach, expect, test, vi } from 'vitest';
import type { ChatStreamHandlers } from '$lib/api/chat';

// Capture the handlers the controller hands to sendChat so a test can drive the
// stream lifecycle (frame / stream-loss / error / done) by hand.
const chat = vi.hoisted(() => ({
  handlers: null as ChatStreamHandlers | null,
  body: null as Record<string, unknown> | null,
  close: vi.fn(),
  // Controllable respond mock so a test can drive ok / expired / rejected.
  respond: vi.fn(async (..._args: unknown[]) => ({ status: 'ok' }) as unknown),
}));
vi.mock('$lib/api/chat', () => ({
  sendChat: vi.fn((body: Record<string, unknown>, handlers: ChatStreamHandlers) => {
    chat.handlers = handlers;
    chat.body = body;
    return { close: chat.close };
  }),
  cancelChat: vi.fn(async () => {}),
  respondToAsk: (...args: unknown[]) => chat.respond(...args),
}));

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('$lib/api/client', () => ({ api: apiMock, authHeaders: () => ({}) }));

const toast = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock('$lib/components/feedback/toast-store.svelte', () => ({ toasts: { push: toast.push } }));

vi.mock('$lib/stores/auth.svelte', () => ({ auth: { userId: 'u1', requireAuth: vi.fn() } }));

import { ConversationController } from './conversation.svelte';

type Ev = Record<string, unknown>;

function controller(events: Ev[]): ConversationController {
  const ctrl = new ConversationController();
  ctrl.sessionId = 'sess-1';
  ctrl.events = events;
  return ctrl;
}

beforeEach(() => {
  chat.handlers = null;
  chat.body = null;
  chat.close.mockClear();
  chat.respond.mockReset();
  chat.respond.mockResolvedValue({ status: 'ok' });
  toast.push.mockClear();
  apiMock.get.mockReset();
  apiMock.get.mockResolvedValue({ events: [] });
  apiMock.post.mockReset();
  apiMock.post.mockResolvedValue({ id: 'sess-1' });
});

test('send threads contextMetadata into the sendChat body when present, omits it otherwise', async () => {
  const ctrl = controller([]);
  await ctrl.send('with ctx', {
    contextMetadata: [{ key: 'location', label: 'Location', value: '1.0, 2.0 (±3m)' }],
  });
  expect(chat.body).toMatchObject({
    message: 'with ctx',
    contextMetadata: [{ key: 'location', label: 'Location', value: '1.0, 2.0 (±3m)' }],
  });
  chat.handlers!.onDone?.();

  await ctrl.send('no ctx');
  expect('contextMetadata' in chat.body!).toBe(false);
});

test('the optimistic bubble carries the attached context as an injected gutter block', async () => {
  const ctrl = controller([]);
  await ctrl.send('with ctx', {
    contextMetadata: [{ key: 'location', label: 'Location', value: '1.0, 2.0 (±3m)' }],
  });
  const opt = ctrl.events.find((e) => e.type === 'user_input' && e.clientKey != null) as Ev;
  expect(opt.injected).toEqual([
    {
      tag: 'client_context',
      items: [{ key: 'location', label: 'Location', value: '1.0, 2.0 (±3m)' }],
    },
  ]);
  // display_text carries the typed words so the turn renders as "you" + gutter,
  // not a pure-injection ("context") row.
  expect(opt.display_text).toBe('with ctx');
  chat.handlers!.onDone?.();
});

test('a user_context frame folds the server context split into the streaming bubble, not a new turn', async () => {
  const ctrl = controller([]);
  await ctrl.send('summarize https://example.com');
  const injected = [
    {
      tag: 'client_context',
      items: [
        {
          key: 'webpage:https://example.com',
          label: 'https://example.com',
          value: 'Example Domain',
        },
      ],
    },
  ];
  // The daemon detected a URL and streams the split back mid-turn.
  chat.handlers!.onEvent?.({ type: 'user_context', injected });
  const users = ctrl.events.filter((e) => e.type === 'user_input');
  expect(users.length).toBe(1); // enriched the bubble, did not append a second turn
  expect((users[0] as Ev).injected).toEqual(injected);
  // still renders as the person's turn, not a synthetic context row
  expect((users[0] as Ev).display_text).toBe('summarize https://example.com');
  chat.handlers!.onDone?.();
});

test('a response-stream loss after delivery reloads in place: no failure toast, no draft restore', async () => {
  const ctrl = controller([
    { type: 'user_input', text: 'earlier', timestamp: 't0' },
    { type: 'final_result', result: 'ok earlier' },
  ]);
  // The turn completed server-side while the stream was dead: reload carries it.
  apiMock.get.mockResolvedValueOnce({
    events: [
      { type: 'user_input', text: 'earlier', timestamp: 't0' },
      { type: 'final_result', result: 'ok earlier' },
      { type: 'user_input', text: 'run the slow thing', timestamp: 't1' },
      { type: 'model_response', raw_content: 'done', thought: 'done' },
      { type: 'final_result', result: 'done' },
    ],
  });

  await ctrl.send('run the slow thing');
  chat.handlers!.onEvent?.({ type: 'turn_start' });
  chat.handlers!.onStreamLost?.(new Error('network error'));
  chat.handlers!.onDone?.();

  await vi.waitFor(() => expect(ctrl.events.some((e) => e.type === 'model_response')).toBe(true));
  expect(toast.push).not.toHaveBeenCalledWith('err', 'Send failed', expect.anything());
  expect(ctrl.sendFailed).toBeNull();
  expect(ctrl.streaming).toBe(false);
  // Optimistic bubble replaced by the persisted truth - exactly one copy, no clientKey.
  const mine = ctrl.events.filter(
    (e) => e.type === 'user_input' && e.text === 'run the slow thing',
  );
  expect(mine).toHaveLength(1);
  expect(mine[0]!.clientKey).toBeUndefined();
});

test('a send that never reached the daemon keeps the failure toast and restores the draft', async () => {
  const ctrl = controller([{ type: 'user_input', text: 'earlier', timestamp: 't0' }]);

  await ctrl.send('offline message');
  chat.handlers!.onError?.(new Error('Failed to fetch'));
  chat.handlers!.onDone?.();

  expect(toast.push).toHaveBeenCalledWith(
    'err',
    'Send failed',
    expect.objectContaining({ body: expect.any(String) }),
  );
  expect(ctrl.sendFailed).not.toBeNull();
  expect(ctrl.sendFailed?.text).toBe('offline message');
  // The turn never took: optimistic bubble dropped, and NO reconcile reload fired.
  expect(ctrl.events.some((e) => e.text === 'offline message')).toBe(false);
  expect(apiMock.get).not.toHaveBeenCalled();
});

test('a delivered send that errors server-side shows the inline error and fires NO redundant toast', async () => {
  // The turn ran (frames arrived) and failed at the provider: the error frame is
  // applied as an inline error block AND then routed to onError. Since it already
  // renders inline, the "Send failed" toast would be a redundant second surface -
  // it must be suppressed. The optimistic bubble stays (the turn did take).
  const ctrl = controller([{ type: 'user_input', text: 'earlier', timestamp: 't0' }]);

  await ctrl.send('use the bad model');
  const errFrame = { type: 'error', error: 'There is an issue with the selected model' };
  chat.handlers!.onEvent?.({ type: 'turn_start' });
  chat.handlers!.onEvent?.(errFrame); // applyFrame → inline error block, gotFrame=true
  chat.handlers!.onError?.(errFrame); // chat.ts routes the error frame here too
  chat.handlers!.onDone?.();

  expect(toast.push).not.toHaveBeenCalledWith('err', 'Send failed', expect.anything());
  // The failure is present inline (the reducer renders it as an error block).
  expect(ctrl.events.some((e) => e.type === 'error')).toBe(true);
  // Not a send failure: the message is kept and no draft is handed back.
  expect(ctrl.sendFailed).toBeNull();
  expect(ctrl.events.some((e) => e.type === 'user_input' && e.text === 'use the bad model')).toBe(
    true,
  );
});

test('an answered approval prompt clears once the turn resumes, even when it ends in an error', async () => {
  // Regression: the approval gate emits an ask_user before the model runs. After
  // approving, the turn may end in an error/cancel (or a lost mobile stream) and
  // never emit a final_result - the prompt must still go away, not stick forever.
  const ctrl = controller([]);
  await ctrl.send('summarize https://example.com');
  chat.handlers!.onEvent?.({
    type: 'ask_user',
    question: 'Fetch content from example.com?',
    question_type: 'approval',
    options: ['Approve', 'Deny', 'Always allow'],
  });
  expect(ctrl.ask?.answered).toBe(false);
  await ctrl.answerAsk('Approve');
  expect(ctrl.ask?.answered).toBe(true);
  chat.handlers!.onEvent?.({ type: 'error', error: 'boom' });
  expect(ctrl.ask).toBeNull();
});

test('an unanswered ask_user stays put while the turn is parked on the user', async () => {
  const ctrl = controller([]);
  await ctrl.send('do a thing');
  chat.handlers!.onEvent?.({
    type: 'ask_user',
    question: 'Pick one',
    question_type: 'choice',
    options: ['a', 'b'],
  });
  // A stray turn frame must not dismiss a prompt the user hasn't answered.
  chat.handlers!.onEvent?.({ type: 'model_response', raw_content: 'x' });
  expect(ctrl.ask).not.toBeNull();
  expect(ctrl.ask?.answered).toBe(false);
});

/** A pending ask state as the resync loader / live frame would set it. */
function pendingAsk(askId = 'ask-1'): NonNullable<ConversationController['ask']> {
  return {
    askId,
    question: 'Push to origin?',
    questionType: 'approval',
    options: ['Approve', 'Deny'],
    answered: false,
    answer: '',
  };
}

test('answerAsk sends the ask_id and marks the prompt answered on an ok reply', async () => {
  const ctrl = controller([]);
  ctrl.ask = pendingAsk('ask-42');
  chat.respond.mockResolvedValueOnce({ status: 'ok' });

  await ctrl.answerAsk('Approve');

  expect(chat.respond).toHaveBeenCalledWith(
    expect.objectContaining({ askId: 'ask-42', response: 'Approve', sessionId: 'sess-1' }),
  );
  expect(ctrl.ask?.answered).toBe(true);
  expect(ctrl.ask?.answer).toBe('Approve');
  expect(toast.push).not.toHaveBeenCalled();
});

test('answerAsk on a rejected respond toasts the failure and stays retryable (not answered)', async () => {
  const ctrl = controller([]);
  ctrl.ask = pendingAsk('ask-42');
  chat.respond.mockRejectedValueOnce(new Error('500 internal error'));

  await ctrl.answerAsk('Approve');

  // Surfaced (the fire-and-forget caller can't), and the prompt stays live.
  expect(toast.push).toHaveBeenCalledWith(
    'err',
    expect.any(String),
    expect.objectContaining({ body: expect.stringContaining('500') }),
  );
  expect(ctrl.ask).not.toBeNull();
  expect(ctrl.ask?.answered).toBe(false);
});

test('answerAsk on an expired reply clears the prompt and toasts that it is no longer active', async () => {
  const ctrl = controller([]);
  ctrl.ask = pendingAsk('ask-42');
  chat.respond.mockResolvedValueOnce({ status: 'expired', detail: 'ask ask-42 is not live' });

  await ctrl.answerAsk('Approve');

  expect(ctrl.ask).toBeNull();
  expect(toast.push).toHaveBeenCalledWith('info', expect.stringMatching(/no longer active/i));
});

test('a reload re-shows a still-pending ask from the persisted events, then clears it on answer', async () => {
  // The durable feed replays a persisted ask_user with no matching ask_answered:
  // the controller must re-derive the prompt so a reloaded page can act on it.
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 1, type: 'user_input', text: 'summarize https://example.com' },
      {
        id: 2,
        type: 'ask_user',
        ask_id: 'ask-abc',
        question: 'Fetch content from example.com?',
        question_type: 'approval',
        options: ['Approve', 'Deny'],
      },
    ],
    has_more: false,
    oldest_id: 1,
  });
  await ctrl.open('sess-1');

  expect(ctrl.ask).not.toBeNull();
  expect(ctrl.ask?.askId).toBe('ask-abc');
  expect(ctrl.ask?.question).toBe('Fetch content from example.com?');
  expect(ctrl.ask?.questionType).toBe('approval');
  expect(ctrl.ask?.answered).toBe(false);

  // A later resync brings the persisted ask_answered: the prompt clears, and a
  // further reload must not re-prompt.
  apiMock.get.mockResolvedValueOnce({
    events: [{ id: 3, type: 'ask_answered', ask_id: 'ask-abc', answer: 'Approve' }],
  });
  await ctrl.resync();
  expect(ctrl.ask).toBeNull();
});

test('a reload of an already-answered ask does not re-prompt', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 1, type: 'user_input', text: 'q' },
      { id: 2, type: 'ask_user', ask_id: 'a1', question: 'ok?', question_type: 'yes_no' },
      { id: 3, type: 'ask_answered', ask_id: 'a1', answer: 'yes' },
      { id: 4, type: 'final_result', result: 'done' },
    ],
    has_more: false,
    oldest_id: 1,
  });
  await ctrl.open('sess-1');
  expect(ctrl.ask).toBeNull();
});

test('a stream loss before the send persisted keeps the optimistic message visible', async () => {
  const ctrl = controller([
    { type: 'user_input', text: 'earlier', timestamp: 't0' },
    { type: 'final_result', result: 'ok' },
  ]);
  // The daemon accepted the turn (200) but hasn't recorded the user_input yet, so
  // the reload shows the same user-input count as before the send.
  apiMock.get.mockResolvedValueOnce({
    events: [
      { type: 'user_input', text: 'earlier', timestamp: 't0' },
      { type: 'final_result', result: 'ok' },
    ],
  });

  await ctrl.send('not yet persisted');
  chat.handlers!.onStreamLost?.(new Error('network error'));
  chat.handlers!.onDone?.();

  await vi.waitFor(() => expect(apiMock.get).toHaveBeenCalled());
  await vi.waitFor(() =>
    expect(ctrl.events.some((e) => e.type === 'user_input' && e.text === 'not yet persisted')).toBe(
      true,
    ),
  );
  expect(toast.push).not.toHaveBeenCalledWith('err', 'Send failed', expect.anything());
  expect(ctrl.sendFailed).toBeNull();
});

test('resync reloads events in place, and is a no-op while a live stream is active', async () => {
  const ctrl = controller([{ type: 'user_input', text: 'q', timestamp: 't0' }]);

  // While streaming, the live per-chat stream is the source of truth - a resync
  // must not clobber it with the (lagging) persisted log.
  ctrl.streaming = true;
  await ctrl.resync();
  expect(apiMock.get).not.toHaveBeenCalled();
  expect(ctrl.events).toHaveLength(1);

  // Not streaming: resync pulls the settled turn in place, without blanking first.
  ctrl.streaming = false;
  apiMock.get.mockResolvedValueOnce({
    events: [
      { type: 'user_input', text: 'q', timestamp: 't0' },
      { type: 'final_result', result: 'answer' },
    ],
  });
  await ctrl.resync();
  expect(apiMock.get).toHaveBeenCalledTimes(1);
  expect(ctrl.events.some((e) => e.type === 'final_result')).toBe(true);
});

/** Paths handed to the mocked api.get, newest last. */
function getPaths(): string[] {
  return apiMock.get.mock.calls.map((c) => String(c[0]));
}

test('open loads only the tail window and records the load-earlier cursor', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 5, type: 'user_input', text: 'q', timestamp: 't5' },
      { id: 6, type: 'final_result', result: 'a' },
    ],
    has_more: true,
    oldest_id: 5,
  });

  await ctrl.open('sess-1');

  expect(getPaths()[0]).toContain('/events?limit=');
  expect(ctrl.events).toHaveLength(2);
  expect(ctrl.hasEarlier).toBe(true);
  expect(ctrl.oldestId).toBe(5);
});

test('resync fetches only newer events (after_id) and APPENDS the delta', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 1, type: 'user_input', text: 'q', timestamp: 't1' },
      { id: 2, type: 'final_result', result: 'a' },
    ],
    has_more: false,
    oldest_id: 1,
  });
  await ctrl.open('sess-1');

  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 3, type: 'user_input', text: 'q2', timestamp: 't3' },
      { id: 4, type: 'final_result', result: 'a2' },
    ],
  });
  await ctrl.resync();

  expect(getPaths()[1]).toContain('after_id=2');
  expect(ctrl.events.map((e) => e.id)).toEqual([1, 2, 3, 4]);
});

test('resync with an empty delta leaves the timeline untouched', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [{ id: 1, type: 'user_input', text: 'q' }],
    has_more: false,
    oldest_id: 1,
  });
  await ctrl.open('sess-1');

  apiMock.get.mockResolvedValueOnce({ events: [] });
  await ctrl.resync();
  expect(ctrl.events).toHaveLength(1);
});

test('resync discards its delta when the session switched mid-fetch', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [{ id: 1, type: 'user_input', text: 'q' }],
    has_more: false,
    oldest_id: 1,
  });
  await ctrl.open('sess-1');

  apiMock.get.mockImplementationOnce(async () => {
    ctrl.sessionId = 'other'; // a switch lands while the fetch is in flight
    return { events: [{ id: 2, type: 'final_result', result: 'stale' }] };
  });
  await ctrl.resync();
  expect(ctrl.events.some((e) => e.id === 2)).toBe(false);
});

test('resync falls back to a full window reload when a compaction lands in the delta', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 1, type: 'user_input', text: 'q' },
      { id: 2, type: 'final_result', result: 'a' },
    ],
    has_more: false,
    oldest_id: 1,
  });
  await ctrl.open('sess-1');

  apiMock.get.mockResolvedValueOnce({
    events: [{ id: 3, type: 'compaction', replaced_count: 5, retained_count: 1 }],
  });
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 2, type: 'final_result', result: 'a' },
      { id: 3, type: 'compaction', replaced_count: 5, retained_count: 1 },
    ],
    has_more: true,
    oldest_id: 2,
  });
  await ctrl.resync();

  // The delta carried a compaction, so the controller reloads the window instead
  // of blindly appending a suffix whose positions the compaction shifted.
  expect(getPaths()[2]).toContain('/events?limit=');
  expect(ctrl.timeline.compactions).toHaveLength(1);
});

test('loadEarlier prepends the earlier page and advances the cursor', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 5, type: 'user_input', text: 'q5' },
      { id: 6, type: 'final_result', result: 'a6' },
    ],
    has_more: true,
    oldest_id: 5,
  });
  await ctrl.open('sess-1');

  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 3, type: 'user_input', text: 'q3' },
      { id: 4, type: 'final_result', result: 'a4' },
    ],
    has_more: true,
    oldest_id: 3,
  });
  await ctrl.loadEarlier();

  expect(getPaths()[1]).toContain('before_id=5');
  expect(ctrl.events.map((e) => e.id)).toEqual([3, 4, 5, 6]);
  expect(ctrl.oldestId).toBe(3);
  expect(ctrl.hasEarlier).toBe(true);
});

test('loadEarlier is a no-op when nothing earlier exists', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [{ id: 5, type: 'user_input', text: 'q5' }],
    has_more: false,
    oldest_id: 5,
  });
  await ctrl.open('sess-1');
  apiMock.get.mockClear();

  await ctrl.loadEarlier();
  expect(apiMock.get).not.toHaveBeenCalled();
});

test('ingestBroadcast grows the open conversation live, but never while this surface streams', async () => {
  const ctrl = controller([{ id: 1, type: 'user_input', text: 'q' }]);

  // A turn started elsewhere: its mid-turn events arrive on the cross-session
  // broadcast and grow the timeline in place.
  ctrl.ingestBroadcast({
    session_id: 'sess-1',
    event_type: 'thought',
    content: 'from another tab',
  });
  expect(ctrl.events.at(-1)).toMatchObject({ type: 'thought', content: 'from another tab' });

  // A frame for a different session is ignored.
  ctrl.ingestBroadcast({ session_id: 'other', event_type: 'thought', content: 'nope' });
  expect(ctrl.events.some((e) => e.content === 'nope')).toBe(false);

  // While THIS surface streams, its own per-chat stream is authoritative - the
  // broadcast echo must not double-render.
  ctrl.streaming = true;
  ctrl.ingestBroadcast({ session_id: 'sess-1', event_type: 'thought', content: 'echo' });
  expect(ctrl.events.some((e) => e.content === 'echo')).toBe(false);
});

test('a delivered send whose stream was lost dedupes the message when the tail-window count is unchanged', async () => {
  // Long session: the loaded tail window holds 3 user_inputs (a, b, c). The turn
  // WAS recorded, but the fresh tail window dropped the oldest ('a') off its front
  // as the new turn's events pushed it out - so the server user_input COUNT is
  // still 3. The old count-based reconcile reads "not recorded" and keeps the
  // optimistic bubble ALONGSIDE the persisted copy already in the window, so the
  // just-sent message renders twice until a reload (the reported regression).
  const ctrl = controller([
    { id: 10, type: 'user_input', text: 'a', timestamp: 't1' },
    { id: 11, type: 'final_result', result: 'ra' },
    { id: 12, type: 'user_input', text: 'b', timestamp: 't2' },
    { id: 13, type: 'final_result', result: 'rb' },
    { id: 14, type: 'user_input', text: 'c', timestamp: 't3' },
    { id: 15, type: 'final_result', result: 'rc' },
  ]);
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 12, type: 'user_input', text: 'b', timestamp: 't2' },
      { id: 13, type: 'final_result', result: 'rb' },
      { id: 14, type: 'user_input', text: 'c', timestamp: 't3' },
      { id: 15, type: 'final_result', result: 'rc' },
      { id: 16, type: 'user_input', text: 'hello', timestamp: 't4' },
      { id: 17, type: 'final_result', result: 'rhello' },
    ],
  });

  await ctrl.send('hello');
  chat.handlers!.onStreamLost?.(new Error('network error'));
  chat.handlers!.onDone?.();

  // Wait for the reconcile reload to land (the persisted id-16 copy appears),
  // then assert the optimistic bubble was dropped rather than kept beside it.
  await vi.waitFor(() => expect(ctrl.events.some((e) => e.id === 16)).toBe(true));
  const mine = ctrl.events.filter((e) => e.type === 'user_input' && e.text === 'hello');
  expect(mine).toHaveLength(1);
  expect(mine[0]!.clientKey).toBeUndefined();
});

test('ingestBroadcast never doubles the just-sent message when a user_input echo lands after the stream ends', async () => {
  // The sender holds an optimistic (clientKey) bubble; the turn's stream closes
  // (streaming false). A late cross-session echo of the SAME user_input (user_input
  // is not in the broadcast skip set) must not append a second copy - the
  // optimistic bubble already renders it, and the next resync reconciles ids.
  const ctrl = controller([{ id: 1, type: 'user_input', text: 'earlier' }]);
  await ctrl.send('hello');
  chat.handlers!.onDone?.();
  expect(ctrl.streaming).toBe(false);

  ctrl.ingestBroadcast({ session_id: 'sess-1', event_type: 'user_input', text: 'hello' });
  expect(ctrl.events.filter((e) => e.type === 'user_input' && e.text === 'hello')).toHaveLength(1);
});

test('a resync replaces live broadcast frames with the authoritative delta (no double render)', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 1, type: 'user_input', text: 'q' },
      { id: 2, type: 'final_result', result: 'a' },
    ],
    has_more: false,
    oldest_id: 1,
  });
  await ctrl.open('sess-1');

  // A foreign turn's live frames land id-less via the broadcast.
  ctrl.ingestBroadcast({ session_id: 'sess-1', event_type: 'user_input', text: 'q2' });
  ctrl.ingestBroadcast({ session_id: 'sess-1', event_type: 'thought', content: 'working' });
  expect(ctrl.events).toHaveLength(4);

  // Busy settles -> resync fetches the authoritative persisted copies by id.
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 3, type: 'user_input', text: 'q2' },
      { id: 4, type: 'thought', content: 'working' },
      { id: 5, type: 'final_result', result: 'a2' },
    ],
  });
  await ctrl.resync();

  // The id-less broadcast placeholders are gone, replaced by the id-bearing delta.
  expect(ctrl.events.every((e) => typeof e.id === 'number')).toBe(true);
  expect(ctrl.events.map((e) => e.id)).toEqual([1, 2, 3, 4, 5]);
});

test('the per-turn cache meta survives streaming, settle, and a resync (footer never blinks out)', async () => {
  // Report: the "N cached" footer vanished when the reply finished. Pin that the
  // turn's cache meta is present through the whole lifecycle - a live turn, its
  // settle (final_result meta spread), and the persisted resync (the delta-merge
  // replacing the live frames with the id-bearing copies, whose model_response
  // carries the same usage dump). Any of those three dropping cacheRead is the bug.
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 10, type: 'user_input', text: 'earlier' },
      { id: 11, type: 'final_result', result: 'ok' },
    ],
    has_more: false,
    oldest_id: 10,
  });
  await ctrl.open('sess-1');

  await ctrl.send('go');
  chat.handlers!.onEvent?.({ type: 'turn_start' });
  chat.handlers!.onEvent?.({
    type: 'model_response',
    thought: 's1',
    usage: { cache_read_input_tokens: 61000 },
  });
  expect(ctrl.timeline.turns.at(-1)?.meta?.cacheRead).toBe(61000); // live
  chat.handlers!.onEvent?.({ type: 'final_result', result: 'done', tokens: 5 });
  chat.handlers!.onDone?.();
  expect(ctrl.timeline.turns.at(-1)?.meta?.cacheRead).toBe(61000); // after settle

  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 12, type: 'user_input', text: 'go' },
      { id: 13, type: 'model_response', thought: 's1', usage: { cache_read_input_tokens: 61000 } },
      { id: 14, type: 'final_result', result: 'done', tokens: 5 },
    ],
  });
  await ctrl.resync();
  expect(ctrl.timeline.turns.at(-1)?.meta?.cacheRead).toBe(61000); // after resync
});

test('pushEcho appends a local echo that survives a resync (separate ephemeral channel)', async () => {
  // The echo channel is deliberately NOT the event log: resync/mergeDelta purge
  // id-less placeholders from `events`, so an echo parked there would vanish or
  // double-render. Pin that a pushed echo survives a resync untouched and never
  // leaks into `events`.
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 1, type: 'user_input', text: 'q' },
      { id: 2, type: 'final_result', result: 'a' },
    ],
    has_more: false,
    oldest_id: 1,
  });
  await ctrl.open('sess-1');

  ctrl.pushEcho('/status', 'Model: x', true);
  expect(ctrl.localEcho).toHaveLength(1);
  expect(ctrl.localEcho[0]).toMatchObject({ command: '/status', output: 'Model: x', ok: true });

  apiMock.get.mockResolvedValueOnce({ events: [{ id: 3, type: 'final_result', result: 'a2' }] });
  await ctrl.resync();

  expect(ctrl.localEcho).toHaveLength(1);
  expect(ctrl.localEcho[0]?.command).toBe('/status');
  expect(ctrl.events.some((e) => e.command === '/status')).toBe(false);
});

test('switching sessions clears local echoes (per-session-view, gone on reload)', async () => {
  const ctrl = new ConversationController();
  apiMock.get.mockResolvedValue({ events: [], has_more: false, oldest_id: null });
  await ctrl.open('sess-1');
  ctrl.pushEcho('/status', 'Model: x', true);
  expect(ctrl.localEcho).toHaveLength(1);

  await ctrl.open('sess-2');
  expect(ctrl.localEcho).toHaveLength(0);
});

test('pushEcho carries an optional action affordance (the /job open-jobs link)', () => {
  const ctrl = new ConversationController();
  ctrl.pushEcho('/job build', 'Job started', true, { label: 'Open jobs', href: '#jobs' });
  expect(ctrl.localEcho[0]?.action).toEqual({ label: 'Open jobs', href: '#jobs' });
});

test('dismissEcho drops just that echo and leaves its siblings', () => {
  const ctrl = new ConversationController();
  ctrl.pushEcho('/status', 'Model: x', true);
  ctrl.pushEcho('/model haiku', 'switched', true);
  const [first, second] = ctrl.localEcho;

  ctrl.dismissEcho(first!.id);

  expect(ctrl.localEcho.map((e) => e.id)).toEqual([second!.id]);
  expect(ctrl.localEcho[0]?.command).toBe('/model haiku');
});

test('a normal send on a long session renders the message once through the settle resync', async () => {
  // The live settle path (diagnosed on the :18461 stack): open a long session's
  // tail window (id-bearing), send, let the per-chat stream finish normally (onDone,
  // NOT onStreamLost), then the busy-settle resync fetches the after_id delta. The
  // server records exactly ONE user_input and user_input is never broadcast, so
  // mergeDelta must drop the id-less optimistic bubble and keep only the persisted
  // (id-bearing) copy - one user turn, not two. (A live "2" seen via a substring
  // count was the model echoing the prompt's tag into its reply, not a real dupe.)
  const ctrl = new ConversationController();
  const window: Ev[] = [];
  for (let id = 490; id <= 504; id += 2) {
    window.push({ id, type: 'user_input', text: `filler ${id}` });
    window.push({ id: id + 1, type: 'final_result', result: 'r' });
  }
  apiMock.get.mockResolvedValueOnce({ events: window, has_more: true, oldest_id: 490 });
  await ctrl.open('sess-1');

  const MSG = 'Reply exactly: ok. (dedupe-777)';
  await ctrl.send(MSG);
  chat.handlers!.onEvent?.({ type: 'turn_start' });
  chat.handlers!.onEvent?.({ type: 'model_response', thought: 'ok.' });
  chat.handlers!.onEvent?.({ type: 'final_result', result: 'ok.' });
  chat.handlers!.onDone?.();

  apiMock.get.mockResolvedValueOnce({
    events: [
      { id: 505, type: 'user_input', text: MSG },
      { id: 506, type: 'model_response', thought: 'ok.' },
      { id: 507, type: 'final_result', result: 'ok.' },
      { id: 508, type: 'session_end' },
    ],
  });
  await ctrl.resync();

  const mine = ctrl.events.filter((e) => e.type === 'user_input' && e.text === MSG);
  expect(mine).toHaveLength(1);
  expect(mine[0]!.id).toBe(505);
  const userTurns = ctrl.timeline.turns.filter(
    (t) => t.role === 'user' && t.blocks.some((b) => b.kind === 'prose' && b.text === MSG),
  );
  expect(userTurns).toHaveLength(1);
});
