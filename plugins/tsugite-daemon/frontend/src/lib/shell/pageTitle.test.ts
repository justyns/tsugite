import { describe, expect, it } from 'vitest';
import { pageTitle } from './pageTitle';

describe('pageTitle', () => {
  it('counts what is waiting, so a backgrounded tab still says so', () => {
    expect(pageTitle(2)).toBe('(2) Tsugite');
    expect(pageTitle(1)).toBe('(1) Tsugite');
  });

  it('goes back to the bare name once nothing needs you', () => {
    expect(pageTitle(0)).toBe('Tsugite');
  });
});
