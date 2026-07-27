import { describe, expect, it } from 'vitest';
import { nextRovingIndex } from './rovingNav';

describe('nextRovingIndex', () => {
  it('moves right, wrapping past the end', () => {
    expect(nextRovingIndex(0, 'ArrowRight', 3)).toBe(1);
    expect(nextRovingIndex(2, 'ArrowRight', 3)).toBe(0);
  });

  it('moves left, wrapping past the start', () => {
    expect(nextRovingIndex(1, 'ArrowLeft', 3)).toBe(0);
    expect(nextRovingIndex(0, 'ArrowLeft', 3)).toBe(2);
  });

  it('ArrowDown/ArrowUp act as right/left', () => {
    expect(nextRovingIndex(0, 'ArrowDown', 3)).toBe(1);
    expect(nextRovingIndex(0, 'ArrowUp', 3)).toBe(2);
  });

  it('Home jumps to the first item, End to the last', () => {
    expect(nextRovingIndex(1, 'Home', 3)).toBe(0);
    expect(nextRovingIndex(1, 'End', 3)).toBe(2);
  });

  it('ignores unrelated keys', () => {
    expect(nextRovingIndex(0, 'Tab', 3)).toBeNull();
    expect(nextRovingIndex(0, 'Enter', 3)).toBeNull();
    expect(nextRovingIndex(0, 'a', 3)).toBeNull();
  });

  it('returns null for an empty group', () => {
    expect(nextRovingIndex(0, 'ArrowRight', 0)).toBeNull();
  });

  it('a single item wraps to itself', () => {
    expect(nextRovingIndex(0, 'ArrowRight', 1)).toBe(0);
    expect(nextRovingIndex(0, 'ArrowLeft', 1)).toBe(0);
  });
});
