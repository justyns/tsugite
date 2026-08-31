/// <reference types="@vitest/browser/context" />
import { afterEach, expect, test, vi } from 'vitest';
import { flushSync } from 'svelte';
import { followSpaceNav } from './spaceNav.svelte';
import { spaces } from '$lib/stores/spaces.svelte';
import { initRouter, navigate, router } from '$lib/router.svelte';

// Real hash router over the real spaces singleton: the wiring only makes sense
// against an actual `location.hash`, which is why this is a browser test.
initRouter();

let stop: (() => void) | null = null;
afterEach(() => {
  stop?.();
  stop = null;
});

/** hashchange is async, so wait for the router before reading the route. */
async function at(view: string): Promise<void> {
  await vi.waitFor(() => expect(router.view).toBe(view));
  flushSync();
}

async function go(view: string): Promise<void> {
  navigate(view);
  await at(view);
}

function switchTo(id: string): void {
  spaces.setActive(id);
  flushSync();
}

test('each space keeps its own nav selection across repeated switches', async () => {
  stop = followSpaceNav();
  flushSync();

  const a = spaces.activeSpaceId;
  await go('files');

  const b = spaces.addSpace('Second');
  flushSync();
  // A new space starts on the default view, not on whatever the last one showed.
  await at('chats');

  await go('terminals');

  switchTo(a);
  await at('files');
  switchTo(b);
  await at('terminals');
  switchTo(a);
  await at('files');
  switchTo(b);
  await at('terminals');

  expect(spaces.spaces.find((s) => s.id === a)!.nav).toBe('files');
  expect(spaces.spaces.find((s) => s.id === b)!.nav).toBe('terminals');
});
