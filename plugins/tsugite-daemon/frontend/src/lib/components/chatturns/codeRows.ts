/**
 * Lay a code execution's tool calls out in the order they ran, opening a
 * `tsu_group` section where its first call landed.
 *
 * Partitioning `calls` by group instead would reorder the trace (every loose
 * call floating above every group), lose a call whose group is missing, and
 * drop a group that wrapped no tool calls at all.
 */

export interface RowCall {
  groupId?: string;
}
export interface RowGroup {
  id: string;
}

export type CodeRow<C extends RowCall, G extends RowGroup> =
  { kind: 'call'; call: C } | { kind: 'group'; group: G; calls: C[] };

export function codeRows<C extends RowCall, G extends RowGroup>(
  calls: C[],
  groups: G[],
): CodeRow<C, G>[] {
  const byId = new Map(groups.map((g) => [g.id, g]));
  const rows: CodeRow<C, G>[] = [];
  const placed = new Set<string>();
  let openGroup: G | undefined;
  let openCalls: C[] = [];

  for (const call of calls) {
    // An unknown group id means the section never arrived; render the call
    // loose rather than dropping it.
    const group = call.groupId ? byId.get(call.groupId) : undefined;
    if (group !== openGroup) {
      openGroup = group;
      openCalls = [];
      if (group) {
        rows.push({ kind: 'group', group, calls: openCalls });
        placed.add(group.id);
      }
    }
    if (openGroup) openCalls.push(call);
    else rows.push({ kind: 'call', call });
  }

  // A group that wrapped only computation has no call to anchor it.
  for (const group of groups) {
    if (!placed.has(group.id)) rows.push({ kind: 'group', group, calls: [] });
  }
  return rows;
}
