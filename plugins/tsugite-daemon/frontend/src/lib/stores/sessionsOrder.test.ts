import { describe, expect, it } from 'vitest';
import { orderSessions, patchRow, reorderPins, type SessionRowLike } from './sessionsOrder';

describe('orderSessions', () => {
  it('floats pinned rows to the top by pin_position, keeping the rest in order', () => {
    const rows = [
      { id: 'a' },
      { id: 'b', pinned: true, pin_position: 1 },
      { id: 'c' },
      { id: 'd', pinned: true, pin_position: 0 },
    ];
    expect(orderSessions(rows).map((r) => r.id)).toEqual(['d', 'b', 'a', 'c']);
  });

  it('places pinned rows with a null position after positioned ones', () => {
    const rows = [
      { id: 'a', pinned: true, pin_position: null },
      { id: 'b', pinned: true, pin_position: 2 },
    ];
    expect(orderSessions(rows).map((r) => r.id)).toEqual(['b', 'a']);
  });
});

describe('patchRow', () => {
  it('merges a partial patch onto the matching row', () => {
    const out = patchRow([{ id: 'a', status: 'running' }], 'a', { status: 'completed' });
    expect(out[0]).toEqual({ id: 'a', status: 'completed' });
  });

  it('is a no-op (same array) when the id is absent', () => {
    const rows: SessionRowLike[] = [{ id: 'a' }];
    expect(patchRow(rows, 'ghost', { title: 'x' })).toBe(rows);
  });
});

describe('reorderPins', () => {
  it('applies an explicit pin order and re-floats them', () => {
    const rows = [
      { id: 'a', pinned: true, pin_position: 0 },
      { id: 'b', pinned: true, pin_position: 1 },
      { id: 'c' },
    ];
    expect(reorderPins(rows, ['b', 'a']).map((r) => r.id)).toEqual(['b', 'a', 'c']);
  });
});
