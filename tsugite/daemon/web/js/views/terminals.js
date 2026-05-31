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
import { toast, copyText } from '../utils.js';
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
    case 'timed_out':     return 'dot failed';
    case 'stream_lost':   return 'dot idle';
    case 'paused-follow': return 'dot running';
    default:              return 'dot idle';
  }
}

function isRunningState(state) {
  return state === 'running' || state === 'starting';
}

function fmtBytes(n) {
  if (!Number.isFinite(n)) return '0';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
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
    this._clockTimer = setInterval(() => { this._clockTick++; }, 1000);
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
    this.selectedId = id;
    this.showEmpty = false;
    this._killArmedId = null;
    if (!this.buffers[id]) {
      this.buffers[id] = { text: '', lines: 0, truncated: false, follow: true };
    }
    this.newLineCounts[id] = 0;
    this._openStream(id);
  },

  // Show the empty state in the main pane — user clicked the terminal
  // section header without picking a specific terminal row.
  showEmptyState() {
    if (this.selectedId) this._closeStream(this.selectedId);
    this.selectedId = null;
    this.showEmpty = true;
  },

  deselectTerminal() {
    this.showEmpty = false;
    if (!this.selectedId) return;
    this._closeStream(this.selectedId);
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

  get hasTerminal() { return this.terminals.length > 0; },

  // SSE plumbing ---------------------------------------------------------
  _closeStream(id) {
    const s = this.streams[id];
    if (!s) return;
    try { s.close(); } catch { /* non-fatal */ }
    delete this.streams[id];
  },

  _openStream(id) {
    // Close any active stream other than the one we're about to attach to —
    // we only mount the renderer for the selected terminal, and dragging two
    // streams adds CPU + bandwidth for no benefit.
    for (const otherId of Object.keys(this.streams)) {
      if (otherId !== id) this._closeStream(otherId);
    }
    if (this.streams[id]) return;
    const stream = this._connectSSE(id);
    this.streams[id] = stream;
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
        this._appendOutput(id, data.chunk || '');
      } else if (eventName === 'state') {
        this._updateState(id, data);
      } else if (eventName === 'exit') {
        const term = this.terminals.find(t => t.id === id);
        if (term) {
          // Backend emits `exit_code`; accept `code` too for forward-compat.
          const code = data.exit_code ?? data.code ?? null;
          term.exit_code = code;
          if (term.state !== 'cancelled' && term.state !== 'timed_out') {
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
        // terminal still believes it's running — a clean server close
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

  _appendOutput(id, chunk) {
    if (!chunk) return;
    const buf = this.buffers[id] || (this.buffers[id] = { text: '', lines: 0, truncated: false, follow: true });
    const combined = buf.text + chunk;
    const capped = capBuffer(combined);
    buf.text = capped.text;
    if (capped.truncated) buf.truncated = true;
    const term = this.terminals.find(t => t.id === id);
    if (term) {
      term.bytes_out = (term.bytes_out || 0) + chunk.length;
      const newlines = (chunk.match(/\n/g) || []).length;
      term.lines_out = (term.lines_out || 0) + newlines;
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
      detail: { id, chunk },
    }));
    if (!buf.follow) {
      this.newLineCounts[id] = (this.newLineCounts[id] || 0) + ((chunk.match(/\n/g) || []).length || 1);
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
      await post(`/api/terminals/${encodeURIComponent(id)}/restart`);
      // Buffer is per-spawn; drop it so the new instance starts clean.
      delete this.buffers[id];
      this.newLineCounts[id] = 0;
      window.dispatchEvent(new CustomEvent('tsugite:terminal-clear', { detail: { id } }));
      const term = this.terminals.find(t => t.id === id);
      if (term) {
        term.state = 'starting';
        term.exit_code = null;
        term.updated_at = new Date().toISOString();
      }
      // Re-attach the stream so we pick up the replay + new output.
      this._closeStream(id);
      this._openStream(id);
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
    if (rec.state === 'timed_out') return 'var(--peach)';
    return 'var(--overlay1)';
  },

  isFailedExit(rec) {
    return rec && (rec.state === 'failed' || rec.state === 'timed_out');
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

/* terminalSessionView — Alpine x-data factory that owns the xterm instance
   inside the main pane's full-session block. Mounted via `x-init="bind($el)"`
   on the `.tv-fs` container. Each time the user picks a different terminal,
   Alpine tears down + re-creates this scope, so we get a fresh renderer
   without manual cleanup. */
export function terminalSessionView() {
  return {
    _renderer: null,
    _terminalId: null,
    _resizeObs: null,
    _outputHandler: null,
    _clearHandler: null,
    _jumpHandler: null,

    async bind(root) {
      this._terminalId = this.$store.terminals?.selectedId;
      if (!this._terminalId) return;
      const host = root.querySelector('[data-xterm-host]');
      if (!host) return;

      const renderer = await createTerminalRenderer(host, {
        cursorBlink: true,
        disableStdin: true,  // backend wires stdin later (out of scope for v1)
        scrollback: 5000,
        onScrollState: (atBottom) => {
          this.$store.terminals?.setFollow(this._terminalId, atBottom);
        },
      });
      this._renderer = renderer;

      // Replay any chunks the store buffered while we were loading xterm.
      const buf = this.$store.terminals?.buffers[this._terminalId];
      if (buf && buf.text) renderer.write(buf.text);

      // Wire SSE chunks → xterm.
      this._outputHandler = (ev) => {
        if (ev.detail?.id !== this._terminalId) return;
        renderer.write(ev.detail.chunk);
      };
      this._clearHandler = (ev) => {
        if (ev.detail?.id !== this._terminalId) return;
        try { renderer.term.clear(); } catch { /* non-fatal */ }
      };
      this._jumpHandler = (ev) => {
        if (ev.detail?.id !== this._terminalId) return;
        renderer.jumpToBottom();
      };
      window.addEventListener('tsugite:terminal-output', this._outputHandler);
      window.addEventListener('tsugite:terminal-clear', this._clearHandler);
      window.addEventListener('tsugite:terminal-jump', this._jumpHandler);

      // Keep xterm sized to its pane on layout changes (sidebar resize,
      // viewport rotation, mobile keyboard).
      try {
        this._resizeObs = new ResizeObserver(() => {
          requestAnimationFrame(() => renderer.fitNow());
        });
        this._resizeObs.observe(host);
      } catch { /* ResizeObserver missing — non-fatal */ }
    },

    destroy() {
      if (this._outputHandler) window.removeEventListener('tsugite:terminal-output', this._outputHandler);
      if (this._clearHandler) window.removeEventListener('tsugite:terminal-clear', this._clearHandler);
      if (this._jumpHandler) window.removeEventListener('tsugite:terminal-jump', this._jumpHandler);
      if (this._resizeObs) try { this._resizeObs.disconnect(); } catch { /* non-fatal */ }
      if (this._renderer) try { this._renderer.dispose(); } catch { /* non-fatal */ }
      this._renderer = null;
    },

    async copyOutput() {
      const buf = this.$store.terminals?.buffers[this._terminalId];
      if (!buf?.text) {
        toast('No output to copy', 'info');
        return;
      }
      // Strip ANSI escapes — copy-with-ansi is in the deferred list.
      const stripped = buf.text.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
      await copyText(stripped);
    },
  };
}

/**
 * jobTerminalView — Alpine x-data factory that drives the conditional xterm
 * embed inside a job tile. Mounted via x-init="bind($el, msg)" on the .jx-term
 * container; the host scope tears down + recreates when the job message
 * changes worker session id, so each worker gets a fresh renderer.
 *
 * Lookup heuristic:
 *   findTerminalForParentSession(worker_session_id) — if it returns a record,
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
    _renderer: null,
    _outputHandler: null,
    _clearHandler: null,
    _jumpHandler: null,
    _terminalId: null,
    _resizeObs: null,

    async bind(root, msg) {
      const workerId = msg?.worker_session_id;
      if (!workerId) return;
      const term = await findTerminalForParentSession(workerId);
      if (!term) return;  // llm-only job — no terminal embed
      this._terminalId = term.id;
      this.termLive = isRunningState(term.state);
      this.termActive = true;

      // Wait one frame so the x-show flip has actually painted the host.
      await new Promise((r) => requestAnimationFrame(r));
      const host = root.querySelector('[data-xterm-host]');
      if (!host) return;

      const renderer = await createTerminalRenderer(host, {
        cursorBlink: this.termLive,
        disableStdin: true,
        scrollback: 5000,
      });
      this._renderer = renderer;

      // Pull the cross-pane store buffer (if /api/terminals already streamed
      // some output before the tile mounted) so the embed catches up.
      const store = this.$store?.terminals;
      const existingBuf = store?.buffers?.[this._terminalId];
      if (existingBuf?.text) renderer.write(existingBuf.text);

      // Reuse the cross-pane SSE: select this terminal so the store opens its
      // stream and broadcasts 'tsugite:terminal-output' events; we mirror them
      // into our embedded xterm. Note this also surfaces the terminal in the
      // sidebar (intended) — same backing state as the full-pane view.
      if (store && !store.streams?.[this._terminalId]) {
        store.buffers[this._terminalId] = store.buffers[this._terminalId]
          || { text: '', lines: 0, truncated: false, follow: true };
        store._openStream(this._terminalId);
      }

      this._outputHandler = (ev) => {
        if (ev.detail?.id !== this._terminalId) return;
        renderer.write(ev.detail.chunk);
      };
      this._clearHandler = (ev) => {
        if (ev.detail?.id !== this._terminalId) return;
        try { renderer.term.clear(); } catch { /* non-fatal */ }
      };
      this._jumpHandler = (ev) => {
        if (ev.detail?.id !== this._terminalId) return;
        renderer.jumpToBottom();
      };
      window.addEventListener('tsugite:terminal-output', this._outputHandler);
      window.addEventListener('tsugite:terminal-clear', this._clearHandler);
      window.addEventListener('tsugite:terminal-jump', this._jumpHandler);

      try {
        this._resizeObs = new ResizeObserver(() => {
          requestAnimationFrame(() => renderer.fitNow());
        });
        this._resizeObs.observe(host);
      } catch { /* non-fatal */ }
    },

    destroy() {
      if (this._outputHandler) window.removeEventListener('tsugite:terminal-output', this._outputHandler);
      if (this._clearHandler) window.removeEventListener('tsugite:terminal-clear', this._clearHandler);
      if (this._jumpHandler) window.removeEventListener('tsugite:terminal-jump', this._jumpHandler);
      if (this._resizeObs) try { this._resizeObs.disconnect(); } catch { /* non-fatal */ }
      if (this._renderer) try { this._renderer.dispose(); } catch { /* non-fatal */ }
      this._renderer = null;
    },
  };
}
