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

export function connectEvents(onEvent) {
  const token = getToken();
  const url = token ? `/api/events?token=${encodeURIComponent(token)}` : '/api/events';
  const es = new EventSource(url);
  es.onmessage = (e) => {
    try { onEvent(JSON.parse(e.data)); } catch { /* ignore parse errors */ }
  };
  return es;
}

export async function uploadFiles(path, files) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const resp = await fetch(path, { method: 'POST', headers: authHeaders(), body: form });
  if (!resp.ok) await handleError(resp);
  return resp.json();
}
