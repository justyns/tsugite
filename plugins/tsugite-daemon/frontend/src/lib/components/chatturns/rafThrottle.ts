// Frame-coalescing throttle for expensive per-update work (here: re-parsing the
// full accumulated markdown of a streaming turn). A live token stream fires many
// updates per frame; re-parsing on each is superlinear. This collapses them to at
// most one flush per animation frame while guaranteeing the trailing value renders.

export type Schedule = (cb: () => void) => number;
export type Cancel = (handle: number) => void;

const defaultSchedule: Schedule =
  typeof requestAnimationFrame === 'function'
    ? (cb) => requestAnimationFrame(cb)
    : (cb) => setTimeout(cb, 16) as unknown as number;

const defaultCancel: Cancel =
  typeof cancelAnimationFrame === 'function' ? cancelAnimationFrame : (h) => clearTimeout(h);

/**
 * Wraps `flush` so that rapid `push(value)` calls coalesce to at most one
 * `flush` per animation frame. The first push flushes synchronously (so static
 * content renders on first mount with no flash of empty); further pushes within
 * a frame collapse to the trailing value, which always flushes. `dispose`
 * cancels any pending trailing flush (call it on teardown).
 */
export function rafThrottle<T>(
  flush: (value: T) => void,
  schedule: Schedule = defaultSchedule,
  cancel: Cancel = defaultCancel,
): { push: (value: T) => void; dispose: () => void } {
  let started = false;
  let handle: number | null = null;
  let pending: T;
  let hasPending = false;

  function push(value: T): void {
    if (!started) {
      started = true;
      flush(value);
      return;
    }
    pending = value;
    hasPending = true;
    if (handle !== null) return;
    handle = schedule(() => {
      handle = null;
      if (hasPending) {
        hasPending = false;
        flush(pending);
      }
    });
  }

  function dispose(): void {
    if (handle !== null) {
      cancel(handle);
      handle = null;
    }
  }

  return { push, dispose };
}
