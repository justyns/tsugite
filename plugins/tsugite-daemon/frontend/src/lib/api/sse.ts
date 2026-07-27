/**
 * SSE event stream, ported from js/api.js. The reconnect contract: the server
 * tags every event with a monotonic `seq` and identifies itself with a boot
 * `epoch`. On reconnect we send both back so it can replay what we missed
 * (sleep/wake, blip); a `hello` that reports a new epoch or `resync` means the
 * delta can't be trusted and the caller must fully reload.
 *
 * The pure parts (frame parsing, buffer splitting, resume-query + hello
 * reconciliation) are extracted and unit-tested in sse.test.ts.
 */
import { authHeaders } from '$lib/api/client';
import { auth } from '$lib/stores/auth.svelte';

export interface SSEFrame {
  event?: string;
  id?: string;
  data: string;
}

export interface HelloData {
  epoch: string;
  seq?: number;
  resync?: boolean;
}

export interface SSEEvent {
  type: string;
  seq?: number;
  data?: Record<string, unknown>;
}

export interface SSEHandle {
  close(): void;
  /** Force an immediate reconnect on the SAME instance (PWA resume) - keeps
   *  epoch/lastSeq so missed events replay instead of being lost. */
  kick(): void;
  pause(): void;
  resume(): void;
}

// ---------- pure helpers (unit-tested) ----------

/** Parse one SSE frame (block of lines) into its fields per the spec. Returns
 *  null for comment/keepalive or field-less frames. */
export function parseSSEFrame(frame: string): SSEFrame | null {
  let event: string | undefined;
  let id: string | undefined;
  const dataLines: string[] = [];
  for (const line of frame.split('\n')) {
    if (line === '' || line.startsWith(':')) continue; // blank or comment (keepalive)
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') dataLines.push(value);
    else if (field === 'event') event = value;
    else if (field === 'id') id = value;
  }
  if (dataLines.length === 0) return null;
  return { event, id, data: dataLines.join('\n') };
}

/** Split a stream buffer into complete `\n\n`-delimited frames plus the
 *  trailing partial to carry over to the next read. */
export function splitFrames(buffer: string): { frames: string[]; rest: string } {
  const parts = buffer.split('\n\n');
  const rest = parts.pop() ?? '';
  return { frames: parts, rest };
}

/** Reconnect query: empty until the first hello supplies an epoch. */
export function resumeQuery(epoch: string | null, lastSeq: number): string {
  if (!epoch) return '';
  return `?epoch=${encodeURIComponent(epoch)}&last_seq=${lastSeq}`;
}

/** Reconcile a hello frame against local reconnect state. `reload` is true when
 *  the delta can't be trusted (daemon restart or server-demanded resync). */
export function reconcileHello(
  prev: { epoch: string | null; lastSeq: number },
  hello: HelloData,
): { epoch: string; lastSeq: number; reload: boolean } {
  const fresh = prev.epoch === null;
  const restarted = prev.epoch !== null && hello.epoch !== prev.epoch;
  const resetCursor = fresh || restarted || Boolean(hello.resync);
  return {
    epoch: hello.epoch,
    lastSeq: resetCursor ? (hello.seq ?? 0) : prev.lastSeq,
    reload: restarted || Boolean(hello.resync),
  };
}

// ---------- streaming ----------

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Decode a fetch SSE body into parsed JSON events. `onActivity` fires on any
 *  bytes (keepalive comments included) so the watchdog can tell live from dead. */
export async function* parseSSE(resp: Response, onActivity?: () => void): AsyncGenerator<SSEEvent> {
  const reader = resp.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (onActivity) onActivity();
    buf += decoder.decode(value, { stream: true });
    const { frames, rest } = splitFrames(buf);
    buf = rest;
    for (const frame of frames) {
      const parsed = parseSSEFrame(frame);
      if (!parsed) continue;
      try {
        yield JSON.parse(parsed.data) as SSEEvent;
      } catch {
        continue;
      }
    }
  }
}

export interface NamedSSEEvent {
  /** The frame's `event:` name (e.g. 'state' | 'output' | 'exit'). */
  event?: string;
  data: unknown;
}

/** Like `parseSSE`, but preserves each frame's `event:` name instead of reading
 *  a `type` field out of the JSON. The per-terminal stream frames its events by
 *  name (`event: output\ndata: {...}`), so its consumer needs the frame name. */
export async function* parseNamedSSE(
  resp: Response,
  onActivity?: () => void,
): AsyncGenerator<NamedSSEEvent> {
  const reader = resp.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (onActivity) onActivity();
    buf += decoder.decode(value, { stream: true });
    const { frames, rest } = splitFrames(buf);
    buf = rest;
    for (const frame of frames) {
      const parsed = parseSSEFrame(frame);
      if (!parsed) continue;
      try {
        yield { event: parsed.event, data: JSON.parse(parsed.data) };
      } catch {
        continue;
      }
    }
  }
}

export function connectEvents(
  onEvent: (event: SSEEvent) => void,
  onStatus?: (connected: boolean) => void,
): SSEHandle {
  let running = true;
  let paused = false;
  let backoff = 1000;
  let controller: AbortController | null = null;
  let epoch: string | null = null;
  let lastSeq = 0;
  let lastActivity = Date.now();

  // Dead-connection watchdog: the server keepalives every 15s; a stream that saw
  // nothing for ~3x that is a zombie (laptop wake without a TCP reset) - abort it
  // so the loop reconnects with replay.
  const watchdog = setInterval(() => {
    if (running && !paused && controller && Date.now() - lastActivity > 45000) controller.abort();
  }, 10000);

  async function connect() {
    while (running) {
      if (paused) {
        await sleep(200);
        continue;
      }
      controller = new AbortController();
      try {
        const resp = await fetch('/api/events' + resumeQuery(epoch, lastSeq), {
          headers: authHeaders(),
          signal: controller.signal,
        });
        if (resp.status === 401) {
          running = false;
          auth.requireAuth();
          return;
        }
        if (!resp.ok) throw new Error(resp.statusText);
        backoff = 1000;
        lastActivity = Date.now();
        onStatus?.(true);
        for await (const event of parseSSE(resp, () => {
          lastActivity = Date.now();
        })) {
          if (event.seq) lastSeq = event.seq;
          if (event.type === 'hello') {
            const reconciled = reconcileHello(
              { epoch, lastSeq },
              (event.data ?? {}) as unknown as HelloData,
            );
            epoch = reconciled.epoch;
            lastSeq = reconciled.lastSeq;
            // Replayed events (which follow this frame) reconcile a clean gap; a
            // restart or unreplayable gap needs the full reload instead.
            if (reconciled.reload) onEvent({ type: 'reconnect' });
            continue;
          }
          if (event.type === 'resync_required') {
            onEvent({ type: 'reconnect' });
            continue;
          }
          onEvent(event);
        }
      } catch {
        if (!running) return;
      }
      onStatus?.(false);
      await sleep(backoff);
      backoff = Math.min(backoff * 2, 30000);
    }
  }

  void connect();
  return {
    close() {
      running = false;
      clearInterval(watchdog);
      controller?.abort();
    },
    kick() {
      backoff = 1000;
      controller?.abort();
    },
    pause() {
      paused = true;
      controller?.abort();
    },
    resume() {
      paused = false;
      backoff = 1000;
    },
  };
}
