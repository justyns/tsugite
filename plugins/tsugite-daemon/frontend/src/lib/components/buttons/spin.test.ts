import { afterEach, describe, expect, it, vi } from 'vitest';
import { startSpin } from './spin';

describe('startSpin', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('sets the first frame synchronously, then cycles on a 96ms interval', () => {
    vi.useFakeTimers();
    const frames: string[] = [];
    const stop = startSpin((glyph) => frames.push(glyph));

    expect(frames).toEqual(['⠋']);
    vi.advanceTimersByTime(96 * 3);
    expect(frames).toEqual(['⠋', '⠙', '⠹', '⠸']);

    stop();
  });

  it('wraps back to the first frame after the last one', () => {
    vi.useFakeTimers();
    const frames: string[] = [];
    const stop = startSpin((glyph) => frames.push(glyph));

    vi.advanceTimersByTime(96 * 10);
    expect(frames.at(-1)).toBe('⠋');

    stop();
  });

  it('stops updating once the returned cleanup runs', () => {
    vi.useFakeTimers();
    const frames: string[] = [];
    const stop = startSpin((glyph) => frames.push(glyph));
    stop();

    const countAtStop = frames.length;
    vi.advanceTimersByTime(500);
    expect(frames.length).toBe(countAtStop);
  });

  it('shows a single static glyph and starts no timer when reduced motion is preferred', () => {
    vi.useFakeTimers();
    vi.stubGlobal('window', { matchMedia: () => ({ matches: true }) });
    const frames: string[] = [];
    startSpin((glyph) => frames.push(glyph));

    expect(frames).toEqual(['∙']);
    vi.advanceTimersByTime(1000);
    expect(frames).toEqual(['∙']);
  });
});
