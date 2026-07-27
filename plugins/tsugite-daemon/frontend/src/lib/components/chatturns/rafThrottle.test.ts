import { describe, expect, it } from 'vitest';
import { rafThrottle } from './rafThrottle';

// A hand-driven stand-in for requestAnimationFrame so the coalescing contract
// is exercised deterministically without a real frame clock.
function fakeFrames() {
  let queued: (() => void) | null = null;
  return {
    schedule: (cb: () => void) => {
      queued = cb;
      return 1;
    },
    cancel: () => {
      queued = null;
    },
    get pending() {
      return queued !== null;
    },
    tick() {
      const cb = queued;
      queued = null;
      cb?.();
    },
  };
}

describe('rafThrottle', () => {
  it('renders the first value synchronously (no flash of empty on mount)', () => {
    const out: string[] = [];
    const f = fakeFrames();
    const { push } = rafThrottle<string>((v) => out.push(v), f.schedule, f.cancel);

    push('a');

    expect(out).toEqual(['a']);
    expect(f.pending).toBe(false);
  });

  it('coalesces rapid updates into a single flush per frame', () => {
    const out: string[] = [];
    const f = fakeFrames();
    const { push } = rafThrottle<string>((v) => out.push(v), f.schedule, f.cancel);

    push('a'); // first render, synchronous
    push('b');
    push('c');
    push('d'); // three rapid updates within one frame

    expect(out).toEqual(['a']); // b/c/d not flushed yet
    f.tick();
    expect(out).toEqual(['a', 'd']); // exactly one flush this frame
  });

  it('lets the trailing value win across successive frames', () => {
    const out: string[] = [];
    const f = fakeFrames();
    const { push } = rafThrottle<string>((v) => out.push(v), f.schedule, f.cancel);

    push('a');
    push('b');
    f.tick();
    push('c');
    push('d');
    f.tick();

    expect(out).toEqual(['a', 'b', 'd']);
  });

  it('always flushes the final value (no update is silently dropped)', () => {
    const out: string[] = [];
    const f = fakeFrames();
    const { push } = rafThrottle<string>((v) => out.push(v), f.schedule, f.cancel);

    push('a');
    push('mid');
    push('final');
    f.tick();

    expect(out.at(-1)).toBe('final');
  });

  it('dispose cancels a pending trailing flush', () => {
    const out: string[] = [];
    const f = fakeFrames();
    const { push, dispose } = rafThrottle<string>((v) => out.push(v), f.schedule, f.cancel);

    push('a');
    push('b'); // schedules a trailing flush
    dispose();
    f.tick(); // frame fires but the flush was cancelled

    expect(out).toEqual(['a']);
    expect(f.pending).toBe(false);
  });
});
