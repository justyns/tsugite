/// <reference types="@vitest/browser/context" />
import { afterEach, expect, test } from 'vitest';
import { ScrollFollow } from './scrollFollow.svelte';

const frame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
async function settle(): Promise<void> {
  await frame();
  await frame();
  await frame();
}

let cleanups: (() => void)[] = [];
afterEach(() => {
  cleanups.forEach((fn) => fn());
  cleanups = [];
  document.body.querySelectorAll('.sf-fixture').forEach((n) => n.remove());
});

/** A short, overflowing scroll box: 100px tall over 600px of content. */
function fixture(): HTMLElement {
  const el = document.createElement('div');
  el.className = 'sf-fixture';
  el.style.height = '100px';
  el.style.overflowY = 'auto';
  el.innerHTML = '<div style="height:600px"></div>';
  document.body.appendChild(el);
  return el;
}
function dist(el: HTMLElement): number {
  return el.scrollHeight - el.scrollTop - el.clientHeight;
}
function attach(el: HTMLElement): ScrollFollow {
  const f = new ScrollFollow();
  cleanups.push(f.attach(el));
  return f;
}

test('sync scrolls to the tail while pinned', async () => {
  const el = fixture();
  const f = attach(el);
  f.sync();
  await settle();
  expect(dist(el)).toBeLessThan(4);
});

test('a wheel-up gesture unpins, and sync then leaves the viewport alone', async () => {
  const el = fixture();
  const f = attach(el);
  el.dispatchEvent(new WheelEvent('wheel', { deltaY: -120, bubbles: true }));
  expect(f.pinned).toBe(false);
  el.scrollTop = 0;
  f.sync();
  await settle();
  expect(el.scrollTop).toBe(0); // unpinned: no catch-up
});

test('an unpin between scheduling and paint aborts the in-flight catch-up (no yank)', async () => {
  // The round-32 defect: the old rAF pinned unconditionally, so a pin scheduled
  // while pinned still fired after the user scrolled away. sync() must re-check
  // pinned INSIDE the rAF, so this unpin cancels the scroll.
  const el = fixture();
  const f = attach(el);
  el.scrollTop = 0;
  f.sync(); // scheduled while pinned
  f.pinned = false; // user unpins before the rAF paints
  await settle();
  expect(el.scrollTop).toBe(0);
});

test('scrolling back to the bottom re-pins (position drives re-pin only)', async () => {
  const el = fixture();
  const f = attach(el);
  f.pinned = false;
  el.scrollTop = el.scrollHeight; // to the tail
  el.dispatchEvent(new Event('scroll'));
  expect(f.pinned).toBe(true);
});

test('a scroll that stays away from the bottom never re-pins', async () => {
  const el = fixture();
  const f = attach(el);
  f.pinned = false;
  el.scrollTop = 120; // mid-thread
  el.dispatchEvent(new Event('scroll'));
  expect(f.pinned).toBe(false);
});

test('a gentle scroll up near the bottom stays unpinned (no sticky re-pin)', async () => {
  // The touchpad "stuck at the bottom" defect: a small wheel-up unpins, but the
  // tiny scroll it causes stayed within BOTTOM_EPS and the old position-only re-pin
  // fired right back, so you had to scroll hard to escape. Re-pin must ignore an
  // UPWARD scroll even when it lands near the tail.
  const el = fixture();
  const f = attach(el);
  el.scrollTop = 500; // at the tail
  el.dispatchEvent(new Event('scroll')); // scrolled down to it: pinned
  expect(f.pinned).toBe(true);
  el.dispatchEvent(new WheelEvent('wheel', { deltaY: -20, bubbles: true }));
  expect(f.pinned).toBe(false);
  el.scrollTop = 480; // nudged up 20px, still inside the 32px slop
  el.dispatchEvent(new Event('scroll'));
  expect(f.pinned).toBe(false);
});

test('repin re-pins and snaps to the tail', async () => {
  const el = fixture();
  const f = attach(el);
  f.pinned = false;
  el.scrollTop = 0;
  f.repin();
  await settle();
  expect(f.pinned).toBe(true);
  expect(dist(el)).toBeLessThan(4);
});

function prepend(el: HTMLElement, px: number): void {
  const spacer = document.createElement('div');
  spacer.style.height = `${px}px`;
  el.insertBefore(spacer, el.firstChild);
}

test('preserveAcross holds the reading position when earlier content is prepended', async () => {
  const el = fixture();
  el.style.overflowAnchor = 'none'; // measure OUR adjustment, not the UA's anchoring
  const f = attach(el);
  f.pinned = false; // scrolled up, reading history
  el.scrollTop = 200;
  await f.preserveAcross(() => prepend(el, 300));
  expect(el.scrollTop).toBe(500); // 200 + the 300px added above the fold
});

test('preserveAcross leaves the bottom to the tail-follow while pinned', async () => {
  const el = fixture();
  el.style.overflowAnchor = 'none';
  const f = attach(el); // pinned by default
  el.scrollTop = 0;
  await f.preserveAcross(() => prepend(el, 300));
  // Pinned: preserveAcross must NOT add the prepend delta - the follow owns the tail.
  expect(el.scrollTop).toBe(0);
});
