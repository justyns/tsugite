/**
 * Read a workspace file's bytes with the daemon's Bearer auth, which is
 * header-only, so an `<img src>` pointing at the raw endpoint cannot carry the
 * token.
 */
import { authHeaders } from '$lib/api/client';

async function fetchWorkspaceBlob(path: string, sessionId?: string | null): Promise<Blob> {
  const qs = new URLSearchParams({ path, ...(sessionId ? { session_id: sessionId } : {}) });
  const resp = await fetch(`/api/workspace/raw?${qs.toString()}`, { headers: authHeaders() });
  if (!resp.ok) throw new Error(`workspace raw ${resp.status}`);
  return await resp.blob();
}

/** A blob object URL for an `<img>`. The caller REVOKES it on teardown so a long
 *  conversation does not leak object URLs. Throws on a non-OK response so the
 *  caller can show a broken-file placeholder instead of a dead `<img>`. */
export async function loadWorkspaceObjectURL(path: string): Promise<string> {
  return URL.createObjectURL(await fetchWorkspaceBlob(path));
}

/**
 * The same bytes as a `data:` URI. The HTML preview iframe is sandboxed without
 * `allow-same-origin`, so it runs in an opaque origin and cannot load a blob
 * URL from the app's origin; a data: URI carries the bytes inline and is what
 * the preview's CSP allows. An oversized asset throws rather than
 * base64-inflating a huge file into the document.
 */
export async function loadWorkspaceDataURL(
  path: string,
  sessionId?: string | null,
  maxBytes = 2 * 1024 * 1024,
): Promise<string> {
  const blob = await fetchWorkspaceBlob(path, sessionId);
  if (blob.size > maxBytes) throw new Error(`workspace raw too large: ${blob.size}`);
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('data URI encode failed'));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(blob);
  });
}
