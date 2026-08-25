/**
 * Per-surface chat stream helper. A send POSTs /api/chat and the
 * response is an SSE stream OWNED by this surface - it carries the turn-end and
 * streaming frames (final_result/error/cancelled/stream_chunk) that the
 * cross-session broadcast deliberately withholds, so whichever surface ran the
 * send is the only one that sees them live.
 *
 * Bearer auth needs a header, so this uses raw fetch + the sse.ts parser rather
 * than the api client (same pattern as connectEvents). The stream terminates on
 * a `{"type":"done"}` frame. ask_user answers go back via respondToAsk().
 */
import { api, authHeaders } from '$lib/api/client';
import { auth } from '$lib/stores/auth.svelte';
import { parseSSE } from '$lib/api/sse';

/** A per-chat frame: flat {type, ...fields} (no data wrapper, unlike the global
 *  broadcast). Extra fields are read positionally by the consumer. */
export type ChatFrame = Record<string, unknown> & { type: string };

export interface ChatSendBody {
  message: string;
  userId?: string;
  sessionId?: string;
  reasoningEffort?: string;
  uploadedFiles?: { name: string }[];
  /** Structured client-context items (location, ...) sent as metadata alongside
   *  the message - the backend splits them back out, never into the text. */
  contextMetadata?: { key: string; label: string; value: string }[];
}

export interface ChatStreamHandlers {
  /** Every frame, in order - the primary hook the chat timeline builds from. */
  onEvent?: (frame: ChatFrame) => void;
  /** Streaming token delta (stream_chunk.chunk), when token streaming is on. */
  onDelta?: (text: string) => void;
  /** final_result payload (result / result_data / turns / tokens / cost). */
  onFinal?: (frame: ChatFrame) => void;
  /** An error frame, or a transport failure BEFORE the request was delivered
   *  (fetch rejected, or a non-ok response like 409 busy) - the turn never ran. */
  onError?: (err: ChatFrame | Error) => void;
  /** The response stream died AFTER a 200 opened it - the daemon accepted the
   *  turn and is running it, only the response feed was lost (mobile background
   *  killing the connection). Recoverable by replay, never a send failure. */
  onStreamLost?: (err: Error) => void;
  /** The stream closed (done or cancelled, or a transport end). */
  onDone?: () => void;
}

export interface ChatStreamHandle {
  /** Stop reading locally. Does NOT stop the agent server-side - call
   *  cancelChat() for that (the daemon keeps running the turn otherwise). */
  close(): void;
}

const TERMINAL_FRAMES = new Set(['done', 'cancelled']);

/** Start a chat turn. Returns a handle immediately; the stream is read in the
 *  background and drives the handler callbacks. */
export function sendChat(body: ChatSendBody, handlers: ChatStreamHandlers = {}): ChatStreamHandle {
  const controller = new AbortController();
  let closed = false;
  // A 200 response means the daemon accepted the turn and opened the stream, so
  // any later read failure is a lost RESPONSE, not a failed send. Before that, a
  // failure means the request never got there.
  let responded = false;

  const payload: Record<string, unknown> = { message: body.message };
  if (body.userId != null) payload.user_id = body.userId;
  if (body.sessionId != null) payload.session_id = body.sessionId;
  if (body.reasoningEffort != null) payload.reasoning_effort = body.reasoningEffort;
  if (body.uploadedFiles) payload.uploaded_files = body.uploadedFiles;
  if (body.contextMetadata?.length) payload.context_metadata = body.contextMetadata;

  const run = async (): Promise<void> => {
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (resp.status === 401) {
        auth.requireAuth();
        return;
      }
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({ error: resp.statusText }));
        handlers.onError?.(new Error(detail.error || resp.statusText));
        return;
      }
      responded = true;
      for await (const event of parseSSE(resp)) {
        const frame = event as unknown as ChatFrame;
        if (TERMINAL_FRAMES.has(frame.type)) return;
        handlers.onEvent?.(frame);
        if (frame.type === 'stream_chunk' && typeof frame.chunk === 'string') {
          handlers.onDelta?.(frame.chunk);
        } else if (frame.type === 'final_result') {
          handlers.onFinal?.(frame);
        } else if (frame.type === 'error') {
          handlers.onError?.(frame);
        }
      }
    } catch (err) {
      // A deliberate close() (session switch/teardown) is not a failure. A drop
      // after the 200 is a lost response stream; before it, a real send failure.
      if (!closed) {
        const e = err instanceof Error ? err : new Error(String(err));
        if (responded) handlers.onStreamLost?.(e);
        else handlers.onError?.(e);
      }
    } finally {
      handlers.onDone?.();
    }
  };

  void run();
  return {
    close() {
      closed = true;
      controller.abort();
    },
  };
}

/** Stop the agent server-side (the Stop button). Separate from a stream close:
 *  aborting the fetch leaves the daemon running the turn. */
export async function cancelChat(
  opts: { userId?: string; sessionId?: string } = {},
): Promise<void> {
  const body: Record<string, unknown> = {};
  if (opts.userId != null) body.user_id = opts.userId;
  if (opts.sessionId != null) body.session_id = opts.sessionId;
  await api.post('/api/chat/cancel', body);
}

/** The daemon's reply to a respond POST. `ok` = delivered to the waiting ask;
 *  `expired` = the ask is no longer live (already resolved, timed out, or the
 *  turn moved on) - a non-error the caller surfaces without a retry. */
export interface AskAnswer {
  status?: string;
  detail?: string;
}

/** Answer an ask_user prompt. `askId` targets the durable ask by id (the primary
 *  key the daemon resolves against its live registry); session_id is the legacy
 *  fallback. Returns the parsed reply so the caller can branch on `status`. */
export async function respondToAsk(opts: {
  response: string;
  askId?: string;
  userId?: string;
  sessionId: string;
}): Promise<AskAnswer> {
  return await api.post<AskAnswer>('/api/chat/respond', {
    ask_id: opts.askId,
    response: opts.response,
    user_id: opts.userId,
    session_id: opts.sessionId,
  });
}
