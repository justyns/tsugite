/**
 * Read a workspace file's bytes authenticated and wrap them in an object URL.
 *
 * The daemon's Bearer auth is header-only, so an `<img src>` pointing at the raw
 * endpoint can't carry the token. The caller fetches the bytes with authHeaders(),
 * turns them into a blob object URL for an `<img>`, and REVOKES that URL on
 * teardown so a long conversation doesn't leak object URLs.
 */
import { authHeaders } from '$lib/api/client';

/** Fetch `uploads/<name>` (or any workspace-relative path) as a blob object URL.
 *  Throws on any non-OK response so the caller can show a broken-file placeholder
 *  instead of a dead `<img>`. */
export async function loadWorkspaceObjectURL(path: string): Promise<string> {
  const url = `/api/workspace/raw?path=${encodeURIComponent(path)}`;
  const resp = await fetch(url, { headers: authHeaders() });
  if (!resp.ok) throw new Error(`workspace raw ${resp.status}`);
  const blob = await resp.blob();
  return URL.createObjectURL(blob);
}
