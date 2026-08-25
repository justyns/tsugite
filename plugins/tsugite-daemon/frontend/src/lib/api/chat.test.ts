import { afterEach, expect, test, vi } from 'vitest';

// chat.ts only needs auth headers + the auth gate; stub them so the module (and
// sse.ts, which it imports for parseSSE) load without touching real stores.
vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}), api: { post: vi.fn() } }));
vi.mock('$lib/stores/auth.svelte', () => ({ auth: { requireAuth: vi.fn() } }));

import { api } from '$lib/api/client';
import { sendChat, respondToAsk } from './chat';

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

function frameReader(frames: string[], thenThrow?: Error) {
  const enc = new TextEncoder();
  let i = 0;
  return {
    read: async () => {
      if (i < frames.length) return { done: false, value: enc.encode(frames[i++]) };
      if (thenThrow) throw thenThrow;
      return { done: true, value: undefined };
    },
    cancel: async () => {},
  };
}

test('a response-stream loss after a 200 routes to onStreamLost, not onError (delivered)', async () => {
  // A 200 means the daemon accepted the turn and opened the stream; the turn is
  // running server-side. The OS killing the backgrounded stream mid-read must be
  // reported as a stream loss (recoverable) - never as a send failure.
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    status: 200,
    body: {
      getReader: () => frameReader(['data: {"type":"turn_start"}\n\n'], new Error('network error')),
    },
  })) as unknown as typeof fetch;

  const onEvent = vi.fn();
  const onError = vi.fn();
  const onStreamLost = vi.fn();
  const onDone = vi.fn();
  sendChat({ message: 'hi' }, { onEvent, onError, onStreamLost, onDone });

  await vi.waitFor(() => expect(onDone).toHaveBeenCalled());
  expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ type: 'turn_start' }));
  expect(onStreamLost).toHaveBeenCalledTimes(1);
  expect(onError).not.toHaveBeenCalled();
});

test('a fetch rejection before any response routes to onError (never delivered)', async () => {
  globalThis.fetch = vi.fn(async () => {
    throw new Error('Failed to fetch');
  }) as unknown as typeof fetch;

  const onError = vi.fn();
  const onStreamLost = vi.fn();
  const onDone = vi.fn();
  sendChat({ message: 'hi' }, { onError, onStreamLost, onDone });

  await vi.waitFor(() => expect(onDone).toHaveBeenCalled());
  expect(onError).toHaveBeenCalledTimes(1);
  expect(onStreamLost).not.toHaveBeenCalled();
});

/** Capture the JSON body of the single fetch a send issues (200 + immediate done). */
function captureSendPayload(): { get: () => Record<string, unknown> } {
  let body: Record<string, unknown> = {};
  globalThis.fetch = vi.fn(async (_url: unknown, init: { body: string }) => {
    body = JSON.parse(init.body);
    return {
      ok: true,
      status: 200,
      body: { getReader: () => frameReader(['data: {"type":"done"}\n\n']) },
    };
  }) as unknown as typeof fetch;
  return { get: () => body };
}

test('context_metadata rides the payload when present', async () => {
  const cap = captureSendPayload();
  const onDone = vi.fn();
  sendChat(
    {
      message: 'where am i',
      contextMetadata: [
        { key: 'location', label: 'Location', value: '37.77490, -122.41940 (±20m)' },
      ],
    },
    { onDone },
  );
  await vi.waitFor(() => expect(onDone).toHaveBeenCalled());
  expect(cap.get()).toMatchObject({
    message: 'where am i',
    context_metadata: [
      { key: 'location', label: 'Location', value: '37.77490, -122.41940 (±20m)' },
    ],
  });
});

test('context_metadata is omitted from the payload when empty or absent', async () => {
  const cap = captureSendPayload();
  const onDone = vi.fn();
  sendChat({ message: 'hi', contextMetadata: [] }, { onDone });
  await vi.waitFor(() => expect(onDone).toHaveBeenCalled());
  expect('context_metadata' in cap.get()).toBe(false);
});

test('respondToAsk threads ask_id into the POST body and returns the parsed reply', async () => {
  vi.mocked(api.post).mockResolvedValueOnce({ status: 'ok' });
  const res = await respondToAsk({
    askId: 'ask-7',
    response: 'Approve',
    userId: 'u1',
    sessionId: 'sess-1',
  });
  expect(api.post).toHaveBeenCalledWith('/api/chat/respond', {
    ask_id: 'ask-7',
    response: 'Approve',
    user_id: 'u1',
    session_id: 'sess-1',
  });
  // The parsed reply is returned so the caller can branch on status (ok/expired).
  expect(res).toEqual({ status: 'ok' });
});

test('a non-ok response (e.g. 409 busy) routes to onError (delivered but rejected, unchanged)', async () => {
  globalThis.fetch = vi.fn(async () => ({
    ok: false,
    status: 409,
    statusText: 'Conflict',
    json: async () => ({ error: 'a turn is already running for this session' }),
  })) as unknown as typeof fetch;

  const onError = vi.fn();
  const onStreamLost = vi.fn();
  const onDone = vi.fn();
  sendChat({ message: 'hi' }, { onError, onStreamLost, onDone });

  await vi.waitFor(() => expect(onDone).toHaveBeenCalled());
  expect(onError).toHaveBeenCalledTimes(1);
  expect(onStreamLost).not.toHaveBeenCalled();
});
