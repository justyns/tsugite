import { describe, expect, test } from 'vitest';
import { dropIntent, dropZoneAt } from './dropzone';

describe('dropZoneAt', () => {
  test.each([
    ['left', 10, 150],
    ['right', 290, 150],
    ['top', 150, 10],
    ['bottom', 150, 290],
    ['center', 150, 150],
  ])('%s zone at (%i, %i) in a 300x300 pane', (zone, x, y) => {
    expect(dropZoneAt(300, 300, x, y)).toBe(zone);
  });
  test.each([
    ['top-left nearer the top', 40, 10, 'top'],
    ['top-left nearer the left', 10, 40, 'left'],
    ['bottom-right nearer the bottom', 260, 290, 'bottom'],
    ['bottom-right nearer the right', 290, 260, 'right'],
    ['an exact corner tie', 10, 10, 'left'],
  ])('%s takes the nearer edge', (_, x, y, zone) => {
    expect(dropZoneAt(300, 300, x, y)).toBe(zone);
  });
  test('out-of-bounds and empty panes are handled without NaN zones', () => {
    expect(dropZoneAt(300, 300, -50, 150)).toBe('left');
    expect(dropZoneAt(300, 300, 150, 9999)).toBe('bottom');
    expect(dropZoneAt(0, 300, 0, 0)).toBe('center');
    expect(dropZoneAt(300, 0, 0, 0)).toBe('center');
  });
});

describe('dropIntent', () => {
  test.each([
    ['left', { action: 'split', dir: 'row', position: 'before' }],
    ['right', { action: 'split', dir: 'row', position: 'after' }],
    ['top', { action: 'split', dir: 'col', position: 'before' }],
    ['bottom', { action: 'split', dir: 'col', position: 'after' }],
    ['center', { action: 'dock' }],
  ] as const)('%s maps to %j', (zone, intent) => {
    expect(dropIntent(zone)).toEqual(intent);
  });
});
