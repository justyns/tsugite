import { describe, expect, it } from 'vitest';
import { codeRows, type CodeRow } from './codeRows';

interface C {
  tool: string;
  groupId?: string;
}
interface G {
  id: string;
  title: string;
}

const call = (tool: string, groupId?: string): C => ({ tool, ...(groupId ? { groupId } : {}) });
const group = (id: string, title = id): G => ({ id, title });

// Flatten to a readable shape: 'tool' loose, 'title/tool' inside a group.
const flat = (rows: CodeRow<C, G>[]) =>
  rows.flatMap((r) =>
    r.kind === 'call'
      ? [r.call.tool]
      : r.calls.length
        ? r.calls.map((c) => `${r.group.title}/${c.tool}`)
        : [`${r.group.title}/(empty)`],
  );

describe('codeRows', () => {
  it('keeps execution order when a group sits between loose calls', () => {
    const rows = codeRows(
      [call('first'), call('inside', 'g1'), call('last')],
      [group('g1', 'middle')],
    );
    expect(flat(rows)).toEqual(['first', 'middle/inside', 'last']);
  });

  it('renders a group that wrapped no tool calls', () => {
    expect(flat(codeRows([], [group('g1', 'crunch')]))).toEqual(['crunch/(empty)']);
  });

  it('renders a call whose group is missing rather than dropping it', () => {
    expect(flat(codeRows([call('orphan', 'ghost')], []))).toEqual(['orphan']);
  });

  it('reopens a section when an inner group closes back to its parent', () => {
    const rows = codeRows(
      [call('a', 'outer'), call('b', 'inner'), call('c', 'outer')],
      [group('outer'), group('inner')],
    );
    expect(flat(rows)).toEqual(['outer/a', 'inner/b', 'outer/c']);
  });

  it('groups consecutive calls into one section', () => {
    const rows = codeRows([call('a', 'g1'), call('b', 'g1')], [group('g1')]);
    expect(rows).toHaveLength(1);
    expect(flat(rows)).toEqual(['g1/a', 'g1/b']);
  });
});
