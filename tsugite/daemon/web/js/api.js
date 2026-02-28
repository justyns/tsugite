function getToken() {
  return localStorage.getItem('tsugite_token') || '';
}

export function authHeaders() {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function request(method, path, body, raw = false) {
  const opts = { method, headers: { ...authHeaders() } };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || resp.statusText);
  }
  return raw ? resp : resp.json();
}

export function get(path) { return request('GET', path); }
export function post(path, body) { return request('POST', path, body); }
export function put(path, body) { return request('PUT', path, body); }
export function patch(path, body) { return request('PATCH', path, body); }

// `del` is exported as a function name (keyword-safe via export syntax)
const del_ = (path) => request('DELETE', path);
export { del_ as del };

export function streamPost(path, body) { return request('POST', path, body, true); }
