/**
 * File-drag helpers shared by the chat surface (OS drag/drop) and the composer
 * (clipboard paste). An OS file transfer is the only DataTransfer we act on: it
 * advertises the well-known `Files` type and carries the payload on `.files`
 * (or `.items` with kind 'file' on some engines). Internal surface drags carry a
 * private MIME type instead (see lib/shell/mux/drag.ts) and never list `Files`,
 * so `hasFiles` cleanly leaves them to the mux.
 */

/** True when a drag/paste carries OS files. Safe during `dragover`, where the
 *  payload isn't readable yet but `types` is. */
export function hasFiles(dt: DataTransfer | null): boolean {
  return !!dt && Array.from(dt.types).includes('Files');
}

/** The File objects on a drop/paste, preferring `.files` and falling back to
 *  `.items` (kind 'file'). Text-only transfers yield an empty array. */
export function extractFiles(dt: DataTransfer | null): File[] {
  if (!dt) return [];
  if (dt.files && dt.files.length > 0) return Array.from(dt.files);
  const out: File[] = [];
  for (const item of Array.from(dt.items ?? [])) {
    if (item.kind === 'file') {
      const f = item.getAsFile();
      if (f) out.push(f);
    }
  }
  return out;
}
