/**
 * Tail-follow (pin-to-bottom) controller for the conversation scroll container.
 *
 * The hard rule, and why this exists (round-32 "fighting the auto scroll on
 * mobile while streaming"): UNPIN only on real user input gestures, NEVER from
 * scroll position. The follow's own programmatic catch-up scrolls fire `scroll`
 * events indistinguishable from a finger drag, so a position-based unpin flapped
 * on every stream frame and yanked the viewport back down against the user's
 * touch. RE-pinning is the safe direction to judge by position (the user
 * scrolled back to the bottom), so only re-pin reads scrollTop.
 *
 * A mutated $state class instance, never a reassigned binding (AGENTS.md).
 */
import { tick } from 'svelte';

/** Slop for "at the bottom". Fractional scrollTop on hi-dpi and the transient
 *  clientHeight change from a mobile URL-bar show/hide both leave a few px of
 *  residual distance at the true tail; without this, re-pin never triggers. */
const BOTTOM_EPS = 32;

/** Minimum upward finger travel (px) that counts as a scroll-back gesture, so a
 *  tap or a hair of jitter doesn't unpin. */
const TOUCH_SLOP = 2;

export class ScrollFollow {
  pinned = $state(true);
  #el: HTMLElement | null = null;
  #touchY = 0;

  /** Wire input + scroll listeners to the scroll container. Returns a cleanup. */
  attach(el: HTMLElement): () => void {
    this.#el = el;
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) this.#unpin();
    };
    const onTouchStart = (e: TouchEvent) => {
      this.#touchY = e.touches[0]?.clientY ?? 0;
    };
    const onTouchMove = (e: TouchEvent) => {
      const y = e.touches[0]?.clientY ?? 0;
      // A finger dragging DOWN (clientY grows) pulls earlier content into view -
      // the user is scrolling back through history, so stop following.
      if (y > this.#touchY + TOUCH_SLOP) this.#unpin();
      this.#touchY = y;
    };
    const onKeydown = (e: KeyboardEvent) => {
      if (e.key === 'PageUp' || e.key === 'Home' || e.key === 'ArrowUp') this.#unpin();
    };
    const onPointerDown = (e: PointerEvent) => {
      // Scrollbar-gutter grab: the pointer lands past the content box (clientWidth
      // excludes the scrollbar). Overlay scrollbars have no gutter, so this never
      // trips on mobile - harmless.
      if (e.clientX - el.getBoundingClientRect().left > el.clientWidth) this.#unpin();
    };
    let lastTop = el.scrollTop;
    const onScroll = () => {
      // Re-pin only when the user scrolls DOWN to the tail, never on an upward
      // scroll that merely stayed within BOTTOM_EPS. Without the direction guard, a
      // gentle wheel/touchpad scroll up unpins (onWheel) and then this same small
      // scroll re-pins (still near the bottom), so the view feels stuck to the
      // bottom until you scroll hard enough to clear the slop in one gesture.
      const scrollingDown = el.scrollTop >= lastTop;
      lastTop = el.scrollTop;
      if (scrollingDown && this.#atBottom()) this.pinned = true;
    };
    el.addEventListener('wheel', onWheel, { passive: true });
    el.addEventListener('touchstart', onTouchStart, { passive: true });
    el.addEventListener('touchmove', onTouchMove, { passive: true });
    el.addEventListener('keydown', onKeydown);
    el.addEventListener('pointerdown', onPointerDown);
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('touchstart', onTouchStart);
      el.removeEventListener('touchmove', onTouchMove);
      el.removeEventListener('keydown', onKeydown);
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('scroll', onScroll);
      if (this.#el === el) this.#el = null;
    };
  }

  /** Content grew (a new turn or a stream frame). Scroll to the tail only while
   *  pinned, re-checking `pinned` INSIDE the rAF so an unpin that lands between
   *  scheduling and paint aborts the catch-up rather than yanking the viewport
   *  back down. The second rAF catches the settle relayout (the live stream
   *  preview swapping for parsed blocks) so the tail stays put as a turn ends. */
  sync(): void {
    if (!this.pinned) return;
    const scroll = () => {
      const el = this.#el;
      if (!this.pinned || !el) return;
      el.scrollTop = el.scrollHeight;
    };
    requestAnimationFrame(() => {
      scroll();
      requestAnimationFrame(scroll);
    });
  }

  /** Re-pin and snap to the tail: the jump-to-latest affordance, this surface's
   *  own send, and a session switch all want to resume following from the bottom. */
  repin(): void {
    this.pinned = true;
    this.sync();
  }

  /** Run a mutation that PREPENDS earlier content (the load-earlier affordance)
   *  while holding the viewport on the same message: the height added above the
   *  fold is added back to scrollTop after the DOM settles, so the reading
   *  position doesn't jump up. Skipped while pinned - there the tail-follow owns
   *  the bottom, and adjusting would fight it. */
  async preserveAcross(mutate: () => void | Promise<void>): Promise<void> {
    const el = this.#el;
    const before = el?.scrollHeight ?? 0;
    await mutate();
    await tick();
    if (!el || this.pinned) return;
    const delta = el.scrollHeight - before;
    if (delta > 0) el.scrollTop += delta;
  }

  #unpin(): void {
    this.pinned = false;
  }

  #atBottom(): boolean {
    const el = this.#el;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_EPS;
  }
}
