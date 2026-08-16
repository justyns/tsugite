/** Pure reads over the `file_write` frame an agent's tools broadcast. */

/** The path a tool just wrote, or null when the frame is not a write. */
export function writtenPath(data: Record<string, unknown>): string | null {
  if (data.event_type !== 'file_write') return null;
  return typeof data.path === 'string' && data.path ? data.path : null;
}

function normalize(path: string): string {
  return path.replace(/\/{2,}/g, '/').replace(/\/+$/, '');
}

/** Written paths arrive absolute or workspace-relative, as the tool resolved them. */
export function writeTargetsDoc(
  writtenPath: string,
  relPath: string,
  workspaceDir: string,
): boolean {
  if (!writtenPath || !relPath) return false;
  const written = normalize(writtenPath.replace(/^\.\//, ''));
  if (!written.startsWith('/')) return written === normalize(relPath);
  return !!workspaceDir && written === normalize(`${workspaceDir}/${relPath}`);
}
