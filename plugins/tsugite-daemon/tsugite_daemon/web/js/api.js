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
  throw new Error(err.error || resp.statusText);
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

export async function* parseSSE(responseOrReader) {
  const reader = responseOrReader.body ? responseOrReader.body.getReader() : responseOrReader;
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
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

export function connectEvents(onEvent) {
  let running = true;
  let everConnected = false;
  let backoff = 1000;
  const controller = new AbortController();

  async function connect() {
    while (running) {
      try {
        const resp = await fetch('/api/events', {
          headers: authHeaders(),
          signal: controller.signal,
        });
        if (resp.status === 401) {
          running = false;
          if (_onAuthRequired) _onAuthRequired();
          return;
        }
        if (!resp.ok) throw new Error(resp.statusText);
        // Only fire reconnect after we recover from a prior connection - the
        // initial connect is handled by loadAgents() at boot. Without this gate
        // the UI clears state every retry tick while the daemon is down.
        if (everConnected) onEvent({ type: 'reconnect' });
        everConnected = true;
        backoff = 1000;
        for await (const event of parseSSE(resp)) {
          onEvent(event);
        }
      } catch (e) {
        if (!running) return;
      }
      await new Promise(r => setTimeout(r, backoff));
      backoff = Math.min(backoff * 2, 30000);
    }
  }

  connect();
  return { close() { running = false; controller.abort(); } };
}

export async function uploadFiles(path, files) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const resp = await fetch(path, { method: 'POST', headers: authHeaders(), body: form });
  if (!resp.ok) await handleError(resp);
  return resp.json();
}
