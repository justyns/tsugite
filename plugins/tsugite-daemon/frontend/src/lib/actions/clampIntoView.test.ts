import { describe, expect, it } from 'vitest';
import { clampShiftX } from './clampIntoView';

describe('clampShiftX', () => {
  it('leaves a box that already fits alone', () => {
    expect(clampShiftX(100, 400, 0, 500)).toBe(0);
    expect(clampShiftX(0, 500, 0, 500)).toBe(0);
  });

  it('pulls a box back from past the right edge', () => {
    // A left-anchored popover on a phone, 324 wide, hanging 220px off-screen.
    expect(clampShiftX(256, 580, 0, 360)).toBe(-220);
  });

  it('pushes a box back from past the left edge', () => {
    // A right-anchored popover crossing the pane its ancestor clips to.
    expect(clampShiftX(190, 490, 200, 700)).toBe(10);
  });

  it('pins a box too wide for the span to the left edge', () => {
    expect(clampShiftX(150, 600, 200, 400)).toBe(50);
  });
});
