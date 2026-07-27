import { describe, expect, test } from 'vitest';
import { toConnDisplay } from './connDisplay';

describe('toConnDisplay', () => {
  test('a live stream reads as connected', () => {
    expect(toConnDisplay('live')).toBe('on');
  });

  test('the pre-first-connect state never warns - it reads optimistically as connected', () => {
    expect(toConnDisplay('connecting')).toBe('on');
  });

  test('a drop after a good connect shows the reconnecting state', () => {
    expect(toConnDisplay('reconnecting')).toBe('re');
  });

  test('a given-up stream shows offline', () => {
    expect(toConnDisplay('lost')).toBe('off');
  });
});
