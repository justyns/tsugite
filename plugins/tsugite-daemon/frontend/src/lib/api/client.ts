/**
 * Typed REST client, ported from the old js/api.js. Bearer token comes from
 * localStorage (`tsugite_token`); a 401 flips the auth gate. Kept deliberately
 * boring - one `request` funnel plus a small `api` facade.
 */
import { auth } from '$lib/stores/auth.svelte';

/** Shared with sse.ts so the event stream sends the same bearer header. */
export function authHeaders(): Record<string, string> {
  return auth.token ? { Authorization: `Bearer ${auth.token}` } : {};
}

export interface ApiError extends Error {
  status?: number;
  code?: string;
}

async function handleError(resp: Response): Promise<never> {
  if (resp.status === 401)
    auth.requireAuth(auth.token ? 'The daemon rejected that token. Check it and try again.' : '');
  const body = await resp.json().catch(() => ({ error: resp.statusText }));
  const err = new Error(body.error || resp.statusText) as ApiError;
  err.status = resp.status;
  err.code = body.code;
  throw err;
}

type Method = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

async function request<T>(method: Method, path: string, body?: unknown, raw = false): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
  const init: RequestInit = { method, headers };
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  const resp = await fetch(path, init);
  if (!resp.ok) await handleError(resp);
  return (raw ? resp : await resp.json()) as T;
}

async function uploadFiles<T = unknown>(path: string, files: Iterable<File>): Promise<T> {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  const resp = await fetch(path, { method: 'POST', headers: authHeaders(), body: form });
  if (!resp.ok) await handleError(resp);
  return resp.json() as Promise<T>;
}

export const api = {
  get: <T = unknown>(path: string) => request<T>('GET', path),
  post: <T = unknown>(path: string, body?: unknown) => request<T>('POST', path, body),
  put: <T = unknown>(path: string, body?: unknown) => request<T>('PUT', path, body),
  patch: <T = unknown>(path: string, body?: unknown) => request<T>('PATCH', path, body),
  del: <T = unknown>(path: string) => request<T>('DELETE', path),
  streamPost: (path: string, body?: unknown) => request<Response>('POST', path, body, true),
  uploadFiles,
};
