/* Alpine view + helpers for the Terminal Viewer.

   The terminal viewer lives inside the conversations view, sharing its
   sidebar. The backend exposes per-PTY records over /api/terminals plus a
   per-terminal SSE stream that emits `output` / `state` / `exit` events.

   Lifecycle:
     - on init: GET /api/terminals to populate `terminals`
     - when the user selects one: open its SSE stream, append chunks to a
       client-side buffer (capped at LINE_CAP lines), surface state +
       last-line / metrics into the sidebar row.
     - when deselected (or replaced by another terminal): close the stream.

   The full-session view (terminal.html in index.html) reads `selected` /
   `output` / `follow` / etc. and mounts an xterm renderer lazily via
   xterm-loader.js. */

import { get, post } from '../api.js';
import { toast, copyText, formatFileSize } from '../utils.js';
import { createTerminalRenderer } from '../utils/xterm-loader.js';

const LINE_CAP = 5000;

/**
 * Look up the terminal record (if any) belonging to a given parent session id.
 *
 * The Jobs feature spawns worker sessions that may or may not produce a PTY-
 * backed terminal: an LLM-only job (e.g. "write me a haiku") never does, while
 * a tool-using job (e.g. one that runs pytest) does. The frontend job tile
 * embeds an xterm only when this lookup returns a record. Returns null on
 * miss / network error so callers can hide the terminal pane gracefully.
 */
export async function findTerminalForParentSession(parentSessionId) {
  if (!parentSessionId) return null;
  try {
    const data = await get(`/api/terminals?parent_session_id=${encodeURIComponent(parentSessionId)}`);
    const terms = (data && data.terminals) || [];
    return terms.length > 0 ? terms[0] : null;
  } catch {
    return null;
  }
}

function stateDotClass(state) {
  switch (state) {
    case 'running':       return 'dot running pulse';
    case 'starting':      return 'dot starting';
    case 'succeeded':     return 'dot running';
    case 'failed':        return 'dot failed';
    case 'cancelled':     return 'dot done';
    case 'stream_lost':   return 'dot idle';
    case 'paused-follow': return 'dot running';
    default:              return 'dot idle';
  }
}

function isRunningState(state) {
  return state === 'running' || state === 'starting';
}

function fmtBytes(n) {
  return Number.isFinite(n) ? formatFileSize(n) : '0';
}

function fmtElapsed(s) {
  if (!Number.isFinite(s) || s < 0) return '00:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m < 10 ? '0' : ''}${m}:${sec < 10 ? '0' : ''}${sec}`;
}

function parseElapsedSec(rec) {
  if (!rec) return 0;
  const start = rec.created_at && Date.parse(rec.created_at);
  if (!Number.isFinite(start)) return 0;
  const end = rec.updated_at && Date.parse(rec.updated_at);
  if (isRunningState(rec.state)) {
    return Math.max(0, Math.floor((Date.now() - start) / 1000));
  }
  if (Number.isFinite(end)) return Math.max(0, Math.floor((end - start) / 1000));
  return 0;
}

/* Cap the in-memory buffer to LINE_CAP lines. Returns { text, truncated }. */
function capBuffer(text) {
  const lines = text.split('\n');
  if (lines.length <= LINE_CAP) return { text, truncated: false };
  return { text: lines.slice(lines.length - LINE_CAP).join('\n'), truncated: true };
}

export default () => ({
  // public state ---------------------------------------------------------
  terminals: [],
  selectedId: null,
  // True when the user explicitly chose the "terminal" surface (clicking
  // the section header) without picking a specific terminal. Drives the
  // empty-state placeholder in the main pane.
  showEmpty: false,
  loading: false,

  // per-terminal client buffers/state keyed by id. Survives selection so
  // re-opening a session re-renders without a server round-trip.
  buffers: {},          // id -> { text, lines, truncated, follow }
  streams: {},          // id -> { close() }
  // Streams are shared between the full-pane viewer and any job-tile embeds;
  // track who holds each one so closing a surface never silently freezes
  // another surface watching a different terminal.
  _streamConsumers: {}, // id -> Set of consumer tags
  newLineCounts: {},    // id -> number queued while user scrolled up

  // tick counter forces relative time / elapsed clocks to recompute on render
  _clockTick: 0,
  _clockTimer: null,

  // kill-button two-click safety: id of the terminal whose kill is armed
  _killArmedId: null,
  _killArmTimer: null,

  // helpers --------------------------------------------------------------
  stateDotClass,
  fmtBytes,
  fmtElapsed,

  init() {
    this.loadTerminals();
    // Elapsed labels only change while something is running; skip ticks (and
    // the re-render they force) when idle or backgrounded.
    this._clockTimer = setInterval(() => {
      if (document.hidden) return;
      if (!this.terminals.some(t => isRunningState(t.state))) return;
      this._clockTick++;
    }, 1000);
    // The backend broadcasts `terminal_state` on every PTY transition. This is a
    // store (no $watch magic), so observe app.lastEvent via Alpine.effect; defer
    // the handler to a microtask so reading this.terminals there isn't tracked as
    // an effect dependency (which would loop against loadTerminals' own write).
    Alpine.effect(() => {
      const e = Alpine.store('app').lastEvent;
      if (!e || (e.type !== 'terminal_state' && e.type !== 'reconnect')) return;
      queueMicrotask(() => {
        const d = e.data || {};
        if (e.type === 'terminal_state' && d.terminal_id && d.state) {
          const term = this.terminals.find(x => x.id === d.terminal_id);
          if (term) {
            // The event carries everything the row needs - no full refetch.
            term.state = d.state;
            term.updated_at = new Date().toISOString();
            return;
          }
        }
        // Unknown terminal (just created elsewhere) or reconnect: full refresh.
        this.loadTerminals();
      });
    });
  },

  destroy() {
    if (this._clockTimer) clearInterval(this._clockTimer);
    if (this._killArmTimer) clearTimeout(this._killArmTimer);
    for (const s of Object.values(this.streams)) {
      try { s.close(); } catch { /* non-fatal */ }
    }
    this.streams = {};
  },

  async loadTerminals() {
    this.loading = true;
    try {
      const data = await get('/api/terminals');
      const list = (data && data.terminals) || [];
      this.terminals = list.map(t => this._normalizeRecord(t));
      this.loading = false;
    } catch {
      this.terminals = [];
      this.loading = false;
    }
  },

  _normalizeRecord(rec) {
    return {
      id: rec.id,
      cmd: rec.cmd || '',
      state: rec.state || 'starting',
      created_at: rec.created_at || null,
      updated_at: rec.updated_at || null,
      exit_code: rec.exit_code ?? null,
      bytes_out: rec.bytes_out ?? 0,
      lines_out: rec.lines_out ?? 0,
      last_line: rec.last_line || '',
      pid: rec.pid ?? null,
    };
  },

  // sidebar handlers -----------------------------------------------------
  selectTerminal(id) {
    if (!id) return;
    this.showEmpty = false;
    this._killArmedId = null;
    if (!this.buffers[id]) {
      this.buffers[id] = { text: '', lines: 0, truncated: false, follow: true };
    }
    this.newLineCounts[id] = 0;
    if (this.selectedId && this.selectedId !== id) this._releaseStream(this.selectedId, 'pane');
    this.selectedId = id;
    this._acquireStream(id, 'pane');
  },

  // Show the empty state in the main pane - user clicked the terminal
  // section header without picking a specific terminal row.
  showEmptyState() {
    if (this.selectedId) this._releaseStream(this.selectedId, 'pane');
    this.selectedId = null;
    this.showEmpty = true;
  },

  deselectTerminal() {
    this.showEmpty = false;
    if (!this.selectedId) return;
    this._releaseStream(this.selectedId, 'pane');
    this.selectedId = null;
  },

  isSelected(id) { return this.selectedId === id; },

  get selected() {
    if (!this.selectedId) return null;
    return this.terminals.find(t => t.id === this.selectedId) || null;
  },

  get selectedBuffer() {
    if (!this.selectedId) return null;
    return this.buffers[this.selectedId] || null;
  },

  // SSE plumbing ---------------------------------------------------------
  // A stream stays open while ANY surface (the full-pane viewer, a job-tile
  // embed) holds it; the last release closes it. Acquire is idempotent per
  // consumer tag, so re-selecting a terminal reconnects a dropped stream
  // without double-counting.
  _acquireStream(id, consumer) {
    const set = this._streamConsumers[id] || (this._streamConsumers[id] = new Set());
    set.add(consumer);
    if (!this.streams[id]) this.streams[id] = this._connectSSE(id);
  },

  _releaseStream(id, consumer) {
    const set = this._streamConsumers[id];
    if (set) {
      set.delete(consumer);
      if (set.size) return;
      delete this._streamConsumers[id];
    }
    this._closeStream(id);
  },

  _closeStream(id) {
    delete this._streamConsumers[id];
    const s = this.streams[id];
    if (!s) return;
    try { s.close(); } catch { /* non-fatal */ }
    delete this.streams[id];
  },

  _connectSSE(id) {
    // The backend emits framed SSE (`event: <name>\ndata: <json>\n\n`) on
    // an endpoint that requires bearer auth. EventSource can't send custom
    // headers, so we use fetch + a ReadableStream parser instead.
    const url = `/api/terminals/${encodeURIComponent(id)}/stream`;
    const controller = new AbortController();
    let intentionallyClosed = false;

    const dispatch = (eventName, dataLine) => {
      let data;
      try { data = JSON.parse(dataLine); } catch { return; }
      if (eventName === 'output') {
        this._appendOutput(id, data.chunk || '', !!data.replay);
      } else if (eventName === 'state') {
        this._updateState(id, data);
      } else if (eventName === 'exit') {
        const term = this.terminals.find(t => t.id === id);
        if (term) {
          // Backend emits `exit_code`; accept `code` too for forward-compat.
          const code = data.exit_code ?? data.code ?? null;
          term.exit_code = code;
          if (term.state !== 'cancelled') {
            term.state = code === 0 ? 'succeeded' : 'failed';
          }
          term.updated_at = new Date().toISOString();
        }
      }
    };

    const token = localStorage.getItem('tsugite_token') || '';
    const headers = token ? { Authorization: `Bearer ${token}` } : {};

    (async () => {
      try {
        const resp = await fetch(url, { headers, signal: controller.signal });
        if (!resp.ok || !resp.body) {
          if (!intentionallyClosed) this._markStreamLost(id);
          return;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let pendingEvent = 'message';
        let pendingData = '';
        const flush = () => {
          if (pendingData) dispatch(pendingEvent, pendingData);
          pendingEvent = 'message';
          pendingData = '';
        };
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop();
          for (const line of lines) {
            if (line === '') { flush(); continue; }
            if (line.startsWith('event: ')) pendingEvent = line.slice(7).trim();
            else if (line.startsWith('data: ')) pendingData = line.slice(6);
          }
        }
        flush();
      } catch (e) {
        if (intentionallyClosed) return;
        // Network/abort errors land here. Only flag stream_lost if the
        // terminal still believes it's running - a clean server close
        // after the process exited is expected and silent.
        this._maybeMarkStreamLost(id);
      }
    })();

    return {
      close() {
        intentionallyClosed = true;
        try { controller.abort(); } catch { /* non-fatal */ }
      },
    };
  },

  _markStreamLost(id) {
    const term = this.terminals.find(t => t.id === id);
    if (term && isRunningState(term.state)) {
      term.state = 'stream_lost';
      term.updated_at = new Date().toISOString();
    }
  },

  _maybeMarkStreamLost(id) {
    const term = this.terminals.find(t => t.id === id);
    if (term && isRunningState(term.state)) this._markStreamLost(id);
  },

  _appendOutput(id, chunk, replay = false) {
    if (!chunk) return;
    const buf = this.buffers[id] || (this.buffers[id] = { text: '', lines: 0, truncated: false, follow: true });
    const newlines = (chunk.match(/\n/g) || []).length;
    if (replay) {
      // On (re)connect the backend replays the full PTY ring buffer in one frame.
      // loadTerminals already seeded bytes_out/lines_out from the persisted record,
      // so replace the buffer + SET the metrics rather than appending/doubling.
      const capped = capBuffer(chunk);
      buf.text = capped.text;
      buf.lines = Math.min(newlines, LINE_CAP);
      if (capped.truncated) buf.truncated = true;
    } else {
      // Amortized trim: re-splitting the whole buffer per chunk is O(buffer)
      // and quadratic under fast output, so only trim once 2x over the cap.
      buf.text += chunk;
      buf.lines += newlines;
      if (buf.lines > LINE_CAP * 2) {
        const capped = capBuffer(buf.text);
        buf.text = capped.text;
        buf.lines = LINE_CAP;
        buf.truncated = true;
      }
    }
    const term = this.terminals.find(t => t.id === id);
    if (term) {
      term.bytes_out = replay ? chunk.length : (term.bytes_out || 0) + chunk.length;
      term.lines_out = replay ? newlines : (term.lines_out || 0) + newlines;
      // Remember the last non-empty line for the sidebar preview.
      const segs = chunk.split('\n');
      for (let i = segs.length - 1; i >= 0; i--) {
        const trimmed = segs[i].trim();
        if (trimmed) { term.last_line = trimmed.slice(0, 200); break; }
      }
      term.updated_at = new Date().toISOString();
    }
    // Notify any subscriber (the full-session view) so it can hand the chunk
    // to xterm. Keeping this as an event lets the renderer stay decoupled.
    window.dispatchEvent(new CustomEvent('tsugite:terminal-output', {
      detail: { id, chunk, replay },
    }));
    if (!buf.follow) {
      this.newLineCounts[id] = (this.newLineCounts[id] || 0) + (newlines || 1);
    }
  },

  _updateState(id, data) {
    const term = this.terminals.find(t => t.id === id);
    if (!term) return;
    if (data.state) term.state = data.state;
    if (data.pid != null) term.pid = data.pid;
    term.updated_at = new Date().toISOString();
  },

  // action handlers ------------------------------------------------------
  async runTerminal(cmd) {
    const command = (cmd || '').trim();
    if (!command) {
      toast('Command required', 'error');
      return null;
    }
    try {
      const data = await post('/api/terminals', { cmd: command });
      // The contract is `{terminal_id}` per the brief; the create call may
      // also echo the full record. Refresh the list either way so we have
      // the canonical state, then auto-select the new one.
      const newId = data?.terminal_id || data?.id;
      await this.loadTerminals();
      if (newId) this.selectTerminal(newId);
      return newId;
    } catch (e) {
      toast(`Run failed: ${e.message || e}`, 'error');
      return null;
    }
  },

  async killTerminal(id) {
    if (!id) return;
    // Two-click safety per design: first click arms the button + flips the
    // state pill; the second click within ~3s actually fires the request.
    if (this._killArmedId !== id) {
      this._killArmedId = id;
      if (this._killArmTimer) clearTimeout(this._killArmTimer);
      this._killArmTimer = setTimeout(() => { this._killArmedId = null; }, 3000);
      return;
    }
    if (this._killArmTimer) clearTimeout(this._killArmTimer);
    this._killArmedId = null;
    try {
      await post(`/api/terminals/${encodeURIComponent(id)}/kill`);
      const term = this.terminals.find(t => t.id === id);
      if (term && isRunningState(term.state)) {
        term.state = 'cancelled';
        term.updated_at = new Date().toISOString();
      }
    } catch (e) {
      toast(`Kill failed: ${e.message || e}`, 'error');
    }
  },

  isKillArmed(id) { return this._killArmedId === id; },

  async restartTerminal(id) {
    if (!id) return;
    try {
      // Restart spawns a BRAND-NEW PTY with a new id (HTTP 201, the new record
      // tagged `restarted_from`). The old proc is gone, so we migrate every
      // surface to the new id rather than re-streaming the dead one.
      const data = await post(`/api/terminals/${encodeURIComponent(id)}/restart`);
      const newId = data?.id;
      this._closeStream(id);
      delete this.buffers[id];
      delete this.newLineCounts[id];
      window.dispatchEvent(new CustomEvent('tsugite:terminal-clear', { detail: { id } }));
      await this.loadTerminals();
      if (newId) {
        // selectTerminal seeds a fresh buffer and opens the stream on the new id
        // (which closes the stale old stream too).
        this.selectTerminal(newId);
      }
    } catch (e) {
      toast(`Restart failed: ${e.message || e}`, 'error');
    }
  },

  toggleFollow(id) {
    const buf = this.buffers[id];
    if (!buf) return;
    buf.follow = !buf.follow;
    if (buf.follow) {
      this.newLineCounts[id] = 0;
      window.dispatchEvent(new CustomEvent('tsugite:terminal-jump', { detail: { id } }));
    }
  },

  jumpToTail(id) {
    const buf = this.buffers[id];
    if (!buf) return;
    buf.follow = true;
    this.newLineCounts[id] = 0;
    window.dispatchEvent(new CustomEvent('tsugite:terminal-jump', { detail: { id } }));
  },

  newLinesFor(id) { return this.newLineCounts[id] || 0; },

  setFollow(id, atBottom) {
    const buf = this.buffers[id];
    if (!buf) return;
    if (buf.follow !== atBottom) buf.follow = atBottom;
    if (atBottom) this.newLineCounts[id] = 0;
  },

  // derived view data ----------------------------------------------------
  elapsedFor(rec) {
    // _clockTick is read so Alpine re-evaluates this every second for running terminals.
    void this._clockTick;
    return fmtElapsed(parseElapsedSec(rec));
  },

  exitCodeLabel(rec) {
    if (!rec) return '';
    if (rec.exit_code === null || rec.exit_code === undefined) return '';
    return String(rec.exit_code);
  },

  exitCodeColor(rec) {
    if (!rec) return 'var(--overlay1)';
    if (rec.state === 'succeeded') return 'var(--green)';
    if (rec.state === 'failed') return 'var(--red)';
    return 'var(--overlay1)';
  },

  isFailedExit(rec) {
    return rec && rec.state === 'failed';
  },

  isRunningTerminal(rec) {
    return rec && isRunningState(rec.state);
  },

  isExited(rec) {
    return rec && !isRunningState(rec.state);
  },

  isLost(rec) {
    return rec && rec.state === 'stream_lost';
  },
});

/* Shared xterm wiring for the full-pane session view and the in-tile job
   embed. Both surfaces need: a renderer mounted on a host element, three
   window handlers ('tsugite:terminal-output' / '-clear' / '-jump') filtered
   by terminal id, a ResizeObserver to keep xterm sized to its pane, and a
   matched teardown. Factoring it out keeps the two factories down to just
   their lookup/lifecycle differences.

   opts:
     cursorBlink   - forwarded to createTerminalRenderer (default false)
     onScrollState - forwarded to createTerminalRenderer (full-pane wires
                     this into the store's follow-state)
     replayBuffer  - optional () => string. Writes the returned text before
                     handlers attach so any chunks buffered before mount get
                     rendered.
     onStreamClose - optional () => void called from dispose() after the
                     renderer is torn down (the tile uses this to close the
                     cross-pane SSE when no other surface needs it). */
async function bindXtermToTerminalId(host, terminalId, opts = {}) {
  const renderer = await createTerminalRenderer(host, {
    cursorBlink: !!opts.cursorBlink,
    disableStdin: false,
    scrollback: 5000,
    onScrollState: opts.onScrollState,
  });

  // Forward keystrokes to the PTY. The backend writes them straight to the
  // master fd (POST /api/terminals/<id>/stdin); a dead terminal just 4xx/no-ops.
  //
  // BUT xterm.js auto-answers a program's terminal queries (Device Attributes,
  // Cursor Position, Device Status) by emitting the reply through onData. A
  // full-screen TUI (e.g. the cc-driver's claude) queries often, and on stream
  // (re)connect the whole buffer is replayed so EVERY historical query is
  // answered at once - hundreds of replies fed back into the PTY stdin, a storm.
  // Those replies are emulator bookkeeping, never real keystrokes, so drop them.
  // (Arrow/function keys end in A-H/~, never c/n/R, so they pass through.)
  const TERMINAL_QUERY_REPLY = /^\x1b\[[?>=]?[0-9;]*(?:[cnR]|\$y)$/;
  renderer.term.onData((data) => {
    if (TERMINAL_QUERY_REPLY.test(data)) return;
    post(`/api/terminals/${encodeURIComponent(terminalId)}/stdin`, { data }).catch(() => { /* terminal gone */ });
  });

  if (typeof opts.replayBuffer === 'function') {
    const replay = opts.replayBuffer();
    if (replay) renderer.write(replay);
  }

  // Wire SSE chunks → xterm. A replay frame is the full ring buffer on
  // (re)connect, so clear first to avoid stacking it on what's already shown.
  const outputHandler = (ev) => {
    if (ev.detail?.id !== terminalId) return;
    if (ev.detail.replay) { try { renderer.term.clear(); } catch { /* non-fatal */ } }
    renderer.write(ev.detail.chunk);
  };
  const clearHandler = (ev) => {
    if (ev.detail?.id !== terminalId) return;
    try { renderer.term.clear(); } catch { /* non-fatal */ }
  };
  const jumpHandler = (ev) => {
    if (ev.detail?.id !== terminalId) return;
    renderer.jumpToBottom();
  };
  window.addEventListener('tsugite:terminal-output', outputHandler);
  window.addEventListener('tsugite:terminal-clear', clearHandler);
  window.addEventListener('tsugite:terminal-jump', jumpHandler);

  // Keep xterm sized to its pane on layout changes (sidebar resize, viewport
  // rotation, mobile keyboard).
  let resizeObs = null;
  try {
    resizeObs = new ResizeObserver(() => {
      requestAnimationFrame(() => renderer.fitNow());
    });
    resizeObs.observe(host);
  } catch { /* ResizeObserver missing - non-fatal */ }

  return {
    renderer,
    dispose() {
      window.removeEventListener('tsugite:terminal-output', outputHandler);
      window.removeEventListener('tsugite:terminal-clear', clearHandler);
      window.removeEventListener('tsugite:terminal-jump', jumpHandler);
      if (resizeObs) try { resizeObs.disconnect(); } catch { /* non-fatal */ }
      try { renderer.dispose(); } catch { /* non-fatal */ }
      if (typeof opts.onStreamClose === 'function') opts.onStreamClose();
    },
  };
}

/* terminalSessionView - Alpine x-data factory that owns the xterm instance
   inside the main pane's full-session block. Mounted via `x-init="bind($el)"`
   on the `.tv-fs` container. Each time the user picks a different terminal,
   Alpine tears down + re-creates this scope, so we get a fresh renderer
   without manual cleanup. */
export function terminalSessionView() {
  return {
    _binding: null,
    _terminalId: null,

    async bind(root) {
      this._terminalId = this.$store.terminals?.selectedId;
      if (!this._terminalId) return;
      const host = root.querySelector('[data-xterm-host]');
      if (!host) return;

      this._binding = await bindXtermToTerminalId(host, this._terminalId, {
        cursorBlink: true,
        onScrollState: (atBottom) => {
          this.$store.terminals?.setFollow(this._terminalId, atBottom);
        },
        replayBuffer: () => this.$store.terminals?.buffers[this._terminalId]?.text || '',
      });
    },

    destroy() {
      if (this._binding) this._binding.dispose();
      this._binding = null;
    },

    async copyOutput() {
      const buf = this.$store.terminals?.buffers[this._terminalId];
      if (!buf?.text) {
        toast('No output to copy', 'info');
        return;
      }
      // Strip ANSI escapes - copy-with-ansi is in the deferred list.
      const stripped = buf.text.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
      await copyText(stripped);
    },
  };
}

/**
 * jobTerminalView - Alpine x-data factory that drives the conditional xterm
 * embed inside a job tile. Mounted via x-init="bind($el, msg)" on the .jx-term
 * container; the host scope tears down + recreates when the job message
 * changes worker session id, so each worker gets a fresh renderer.
 *
 * Lookup heuristic:
 *   findTerminalForParentSession(worker_session_id) - if it returns a record,
 *   the worker spawned a PTY and we show the embed; otherwise stay hidden so
 *   llm-only jobs (e.g. "write me a haiku") don't get a useless terminal pane.
 *
 * The embed reuses the existing SSE stream (window 'tsugite:terminal-output'
 * events) by selectTerminal()-ing on the store; that keeps the buffer + SSE
 * machinery in one place and means the full-pane terminal view and the in-tile
 * embed share the same backing state.
 */
export function jobTerminalView() {
  return {
    termActive: false,
    termLive: false,
    _binding: null,
    _terminalId: null,

    async bind(root, msg) {
      // Prefer the backend-supplied id (no round-trip); fall back to the
      // /api/terminals probe for legacy replayed messages that don't carry it.
      let terminalId = msg?.worker_terminal_id || null;
      if (!terminalId) {
        const workerId = msg?.worker_session_id;
        if (!workerId) return;
        const term = await findTerminalForParentSession(workerId);
        if (!term) return;  // llm-only job - no terminal embed
        terminalId = term.id;
      }
      this._terminalId = terminalId;
      // termLive is approximated from job state: running/verifying → the
      // worker (and so the terminal) is likely still streaming. Avoids a
      // second /api/terminals probe just to fetch terminal.state.
      this.termLive = msg?.state === 'running' || msg?.state === 'verifying';
      this.termActive = true;

      // Wait one frame so the x-show flip has actually painted the host.
      await new Promise((r) => requestAnimationFrame(r));
      const host = root.querySelector('[data-xterm-host]');
      if (!host) return;

      // Reuse the cross-pane SSE: acquire the stream under a per-binding tag
      // so this tile never closes a stream another surface (the full-pane
      // viewer, another tile) is still watching.
      const store = this.$store?.terminals;
      this._streamTag = `job-tile:${this._terminalId}:${Math.random().toString(36).slice(2)}`;
      if (store) {
        store.buffers[this._terminalId] = store.buffers[this._terminalId]
          || { text: '', lines: 0, truncated: false, follow: true };
        store._acquireStream(this._terminalId, this._streamTag);
      }

      this._binding = await bindXtermToTerminalId(host, this._terminalId, {
        cursorBlink: this.termLive,
        replayBuffer: () => store?.buffers?.[this._terminalId]?.text || '',
        // Release this tile's hold on the shared SSE; the stream only closes
        // when no other surface still needs it.
        onStreamClose: () => {
          const s = this.$store?.terminals;
          if (this._terminalId && s) s._releaseStream(this._terminalId, this._streamTag);
        },
      });
    },

    destroy() {
      if (this._binding) this._binding.dispose();
      this._binding = null;
    },
  };
}
