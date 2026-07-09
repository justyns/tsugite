let _onAuthRequired = null;
export function onAuthRequired(cb) { _onAuthRequired = cb; }

function getToken() {
  return localStorage.getItem('tsugite_token') || '';
}

function authHeaders() {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function handleError(resp) {
  if (resp.status === 401 && _onAuthRequired) _onAuthRequired();
  const err = await resp.json().catch(() => ({ error: resp.statusText }));
  const e = new Error(err.error || resp.statusText);
  e.status = resp.status;
  e.code = err.code;
  throw e;
}

async function request(method, path, body, raw = false) {
  const opts = { method, headers: { ...authHeaders() } };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  if (!resp.ok) await handleError(resp);
  return raw ? resp : resp.json();
}

export function get(path) { return request('GET', path); }
export function post(path, body) { return request('POST', path, body); }
export function put(path, body) { return request('PUT', path, body); }
export function patch(path, body) { return request('PATCH', path, body); }

const del_ = (path) => request('DELETE', path);
export { del_ as del };

export function streamPost(path, body) { return request('POST', path, body, true); }

export async function* parseSSE(responseOrReader, onActivity) {
  const reader = responseOrReader.body ? responseOrReader.body.getReader() : responseOrReader;
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (onActivity) onActivity();  // any bytes count, keepalive comments included
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      let parsed;
      try { parsed = JSON.parse(line.slice(6)); } catch { continue; }
      yield parsed;
    }
  }
}

export function connectEvents(onEvent, onStatus) {
  let running = true;
  let paused = false;
  let backoff = 1000;
  let controller = null;
  // Reconnect reconciliation state: the server tags every event with a
  // monotonic seq and identifies itself with a boot epoch. On reconnect we
  // send both back; the server replays what we missed (sleep/wake, blip) or
  // its hello says resync (daemon restart / gap too old) and we do a full
  // reload instead of trusting a delta.
  let epoch = null;
  let lastSeq = 0;
  let lastActivity = Date.now();

  // Dead-connection watchdog: the server keepalives every 15s; a fetch stream
  // that saw nothing for ~3x that is a zombie (laptop wake without a TCP
  // reset) - abort it so the loop reconnects with replay.
  const watchdog = setInterval(() => {
    if (running && !paused && controller && Date.now() - lastActivity > 45000) controller.abort();
  }, 10000);

  async function connect() {
    while (running) {
      if (paused) { await new Promise(r => setTimeout(r, 200)); continue; }
      controller = new AbortController();
      try {
        const params = epoch ? `?epoch=${encodeURIComponent(epoch)}&last_seq=${lastSeq}` : '';
        const resp = await fetch('/api/events' + params, {
          headers: authHeaders(),
          signal: controller.signal,
        });
        if (resp.status === 401) {
          running = false;
          if (_onAuthRequired) _onAuthRequired();
          return;
        }
        if (!resp.ok) throw new Error(resp.statusText);
        backoff = 1000;
        lastActivity = Date.now();
        if (onStatus) onStatus(true);
        for await (const event of parseSSE(resp, () => { lastActivity = Date.now(); })) {
          if (event.seq) lastSeq = event.seq;
          if (event.type === 'hello') {
            const d = event.data || {};
            const restarted = epoch !== null && d.epoch !== epoch;
            const fresh = epoch === null;
            epoch = d.epoch;
            if (fresh || restarted || d.resync) lastSeq = d.seq || 0;
            // Replayed events (which follow this frame) reconcile a clean gap;
            // a restart or unreplayable gap needs the full reload instead.
            if (restarted || d.resync) onEvent({ type: 'reconnect' });
            continue;
          }
          if (event.type === 'resync_required') {
            onEvent({ type: 'reconnect' });
            continue;
          }
          onEvent(event);
        }
      } catch (e) {
        if (!running) return;
      }
      if (onStatus) onStatus(false);
      await new Promise(r => setTimeout(r, backoff));
      backoff = Math.min(backoff * 2, 30000);
    }
  }

  connect();
  return {
    close() { running = false; clearInterval(watchdog); controller?.abort(); },
    // Force an immediate reconnect on the SAME instance (PWA resume): keeps
    // epoch/lastSeq so the missed events replay instead of being lost.
    kick() { backoff = 1000; controller?.abort(); },
    // Test/debug seams: suspend the loop (keeps seq/epoch so the next resume
    // replays the gap) and resume it.
    pause() { paused = true; controller?.abort(); },
    resume() { paused = false; backoff = 1000; },
  };
}

export async function uploadFiles(path, files) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const resp = await fetch(path, { method: 'POST', headers: authHeaders(), body: form });
  if (!resp.ok) await handleError(resp);
  return resp.json();
}
