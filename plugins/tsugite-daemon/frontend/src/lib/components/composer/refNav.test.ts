import { describe, expect, it } from 'vitest';
import { refNav } from './RefAutocomplete.svelte';

describe('refNav', () => {
  it('ignores keys when the list is empty', () => {
    expect(refNav('ArrowDown', 0, 0)).toEqual({
      activeIndex: 0,
      select: false,
      close: false,
      handled: false,
    });
  });

  it('moves down and clamps at the last item', () => {
    expect(refNav('ArrowDown', 0, 3)).toMatchObject({ activeIndex: 1, handled: true });
    expect(refNav('ArrowDown', 2, 3)).toMatchObject({ activeIndex: 2, handled: true });
  });

  it('moves up and clamps at the first item', () => {
    expect(refNav('ArrowUp', 2, 3)).toMatchObject({ activeIndex: 1, handled: true });
    expect(refNav('ArrowUp', 0, 3)).toMatchObject({ activeIndex: 0, handled: true });
  });

  it('jumps to the ends with Home/End', () => {
    expect(refNav('Home', 2, 3)).toMatchObject({ activeIndex: 0, handled: true });
    expect(refNav('End', 0, 3)).toMatchObject({ activeIndex: 2, handled: true });
  });

  it('selects and closes on Enter and Tab', () => {
    expect(refNav('Enter', 1, 3)).toMatchObject({ activeIndex: 1, select: true, close: true });
    expect(refNav('Tab', 1, 3)).toMatchObject({ activeIndex: 1, select: true, close: true });
  });

  it('closes without selecting on Escape', () => {
    expect(refNav('Escape', 1, 3)).toMatchObject({ select: false, close: true, handled: true });
  });

  it('does not consume unrelated keys', () => {
    expect(refNav('a', 1, 3)).toMatchObject({ handled: false, select: false, close: false });
  });
});
