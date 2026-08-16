import { describe, expect, it } from 'vitest';
import { moveItem } from './reorder';

describe('moveItem', () => {
  const abc = ['a', 'b', 'c', 'd'];

  it('moves an item later, accounting for its own removal', () => {
    expect(moveItem(abc, 0, 3)).toEqual(['b', 'c', 'a', 'd']);
  });

  it('moves an item earlier', () => {
    expect(moveItem(abc, 3, 1)).toEqual(['a', 'd', 'b', 'c']);
  });

  it('moves to the front and to the end', () => {
    expect(moveItem(abc, 2, 0)).toEqual(['c', 'a', 'b', 'd']);
    expect(moveItem(abc, 1, 4)).toEqual(['a', 'c', 'd', 'b']);
  });

  it('is a no-op when the insertion point is the item itself', () => {
    expect(moveItem(abc, 1, 1)).toEqual(abc);
    expect(moveItem(abc, 1, 2)).toEqual(abc);
  });

  it('clamps an out-of-range insertion point', () => {
    expect(moveItem(abc, 0, 99)).toEqual(['b', 'c', 'd', 'a']);
    expect(moveItem(abc, 3, -5)).toEqual(['d', 'a', 'b', 'c']);
  });

  it('leaves the list alone when the source index is out of range', () => {
    expect(moveItem(abc, 9, 0)).toBe(abc);
  });
});
