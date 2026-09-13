import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { RailPeek } from './railPeek.svelte';

const OPEN_DELAY = 150;
const CLOSE_DELAY = 150;

function hoverable(): RailPeek {
  const peek = new RailPeek();
  peek.hoverable = true;
  return peek;
}

describe('railPeek', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test('a pointer settling on a collapsed rail peeks it open after the intent delay', () => {
    const peek = hoverable();
    peek.enter('rail');
    expect(peek.open).toBeNull();
    vi.advanceTimersByTime(OPEN_DELAY);
    expect(peek.open).toBe('rail');
  });

  test('a pointer crossing a rail on its way elsewhere never opens it', () => {
    const peek = hoverable();
    peek.enter('rail');
    vi.advanceTimersByTime(OPEN_DELAY - 50);
    peek.leave();
    vi.advanceTimersByTime(OPEN_DELAY + CLOSE_DELAY);
    expect(peek.open).toBeNull();
  });

  test('leaving an open peek closes it after the close delay', () => {
    const peek = hoverable();
    peek.enter('nav');
    vi.advanceTimersByTime(OPEN_DELAY);
    peek.leave();
    expect(peek.open).toBe('nav');
    vi.advanceTimersByTime(CLOSE_DELAY);
    expect(peek.open).toBeNull();
  });

  test('a pointer straying off and back before the close delay keeps the peek open', () => {
    const peek = hoverable();
    peek.enter('nav');
    vi.advanceTimersByTime(OPEN_DELAY);
    peek.leave();
    vi.advanceTimersByTime(CLOSE_DELAY - 50);
    peek.enter('nav');
    vi.advanceTimersByTime(CLOSE_DELAY);
    expect(peek.open).toBe('nav');
  });

  test('close() drops the peek without waiting, and no pending timer revives it', () => {
    const peek = hoverable();
    peek.enter('rail');
    vi.advanceTimersByTime(OPEN_DELAY);
    peek.close();
    expect(peek.open).toBeNull();
    vi.advanceTimersByTime(OPEN_DELAY + CLOSE_DELAY);
    expect(peek.open).toBeNull();
  });

  test('a coarse pointer never peeks', () => {
    const peek = new RailPeek();
    peek.enter('rail');
    vi.advanceTimersByTime(OPEN_DELAY);
    expect(peek.open).toBeNull();
  });
});
