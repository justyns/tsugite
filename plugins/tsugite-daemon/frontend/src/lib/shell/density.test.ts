import { describe, expect, test } from 'vitest';
import { SHORT_PANE_PX, isShortPane } from './density';

describe('isShortPane', () => {
  test('a pane just under the threshold is short', () => {
    expect(isShortPane(SHORT_PANE_PX - 1)).toBe(true);
  });

  test('a pane at the threshold is not', () => {
    expect(isShortPane(SHORT_PANE_PX)).toBe(false);
  });

  test('an unmeasured pane (height 0) is not', () => {
    expect(isShortPane(0)).toBe(false);
  });
});
