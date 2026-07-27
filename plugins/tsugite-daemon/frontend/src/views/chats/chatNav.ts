/**
 * Chat-list neighbour lookup for the Alt+↑/↓ prev/next shortcut. Pure and
 * node-testable: given the rail's ordered ids and the current one, return the id
 * one step away in `dir`, CLAMPED - null at either end, when the current id
 * isn't in the list (so a keypress from an unlisted/ended session is a no-op
 * rather than a surprise jump), or the list is too short to move.
 */
export function neighborSession(
  orderedIds: string[],
  currentId: string | null,
  dir: 1 | -1,
): string | null {
  if (currentId === null || orderedIds.length < 2) return null;
  const i = orderedIds.indexOf(currentId);
  if (i === -1) return null;
  const j = i + dir;
  if (j < 0 || j >= orderedIds.length) return null;
  return orderedIds[j] ?? null;
}
