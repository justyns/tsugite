import { describe, expect, test } from 'vitest';
import { DROP_EDGE, dropIntent, dropZoneAt } from './dropzone';

describe('dropZoneAt', () => {
  test('left third of the pane is the split-left zone', () => {
    expect(dropZoneAt(300, 10)).toBe('left');
    expect(dropZoneAt(300, 300 * DROP_EDGE - 1)).toBe('left');
  });
  test('right third of the pane is the split-right zone', () => {
    expect(dropZoneAt(300, 290)).toBe('right');
    expect(dropZoneAt(300, 300 * (1 - DROP_EDGE) + 1)).toBe('right');
  });
  test('the middle band is the dock-as-tab zone', () => {
    expect(dropZoneAt(300, 150)).toBe('center');
  });
  test('out-of-bounds and zero-width are handled without NaN zones', () => {
    expect(dropZoneAt(300, -50)).toBe('left');
    expect(dropZoneAt(300, 9999)).toBe('right');
    expect(dropZoneAt(0, 0)).toBe('center');
  });
});

describe('dropIntent', () => {
  test('left splits a row and places the new pane before the target', () => {
    expect(dropIntent('left')).toEqual({ action: 'split', dir: 'row', position: 'before' });
  });
  test('right splits a row and places the new pane after the target', () => {
    expect(dropIntent('right')).toEqual({ action: 'split', dir: 'row', position: 'after' });
  });
  test('center docks the surface as a tab', () => {
    expect(dropIntent('center')).toEqual({ action: 'dock' });
  });
});
