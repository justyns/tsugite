/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import TopBar from './TopBar.svelte';
import { spaces } from '$lib/stores/spaces.svelte';
import { collectLeaves } from '$lib/shell/mux/layout';

test('the palette trigger fires its callback', async () => {
  const onOpenPalette = vi.fn();
  await render(TopBar, { onOpenPalette });
  await page.getByRole('button', { name: 'Command palette' }).click();
  expect(onOpenPalette).toHaveBeenCalledOnce();
});

test('the settings trigger fires its callback', async () => {
  const onOpenSettings = vi.fn();
  await render(TopBar, { onOpenPalette: () => {}, onOpenSettings });
  // Settings sits in the top bar only on mobile (the rail footer that hosts it
  // on desktop is hidden below 640px); fire the DOM click so the wiring is
  // proven regardless of the runner's viewport.
  const btn = page.getByRole('button', { name: 'Settings' }).element() as HTMLElement;
  btn.click();
  expect(onOpenSettings).toHaveBeenCalledOnce();
});

// The theme control itself (reflect + switch + roving tabindex) is covered by
// ThemeSeg.svelte.test.ts; the top bar hides it below 640px, so it isn't
// clickable at the browser runner's default viewport.
test('renders the brand wordmark', async () => {
  await render(TopBar, { onOpenPalette: () => {} });
  await expect.element(page.getByText('tsugite')).toBeInTheDocument();
});

// The store is a singleton over real localStorage in the browser runner, so a test
// that mutates it has to put it back (see the undo test below).
test('the top bar mounts the spaces switcher, naming the active space', async () => {
  await render(TopBar, { onOpenPalette: () => {} });
  await expect.element(page.getByRole('group', { name: 'Spaces' })).toBeInTheDocument();
  await expect
    .element(page.getByRole('button', { name: spaces.active.name, exact: true }))
    .toBeInTheDocument();
});

test('the top bar offers a control for creating a space', async () => {
  await render(TopBar, { onOpenPalette: () => {} });
  await expect.element(page.getByRole('button', { name: 'New space' })).toBeInTheDocument();
});

test('closing a space offers an undo that puts it back with its layout', async () => {
  const { toasts } = await import('$lib/components/feedback/toast-store.svelte');
  const doomed = spaces.addSpace('Doomed');
  try {
    toasts.items = [];
    const paneId = collectLeaves(spaces.active.layout.root)[0]!.id;
    spaces.dock(paneId, { kind: 'terminal', params: { id: 'undo-t1' } });

    await render(TopBar, { onOpenPalette: () => {} });
    await page.getByRole('button', { name: 'Close Doomed' }).click();
    expect(spaces.spaces.some((s) => s.id === doomed)).toBe(false);

    const toast = toasts.items.at(-1);
    expect(toast?.actionLabel).toBe('Undo');
    toast!.onAction!();

    expect(spaces.spaces.some((s) => s.id === doomed)).toBe(true);
    expect(spaces.activeSpaceId).toBe(doomed);
    const kinds = collectLeaves(spaces.active.layout.root).flatMap((l) =>
      l.tabs.map((t) => t.kind),
    );
    expect(kinds).toContain('terminal');
    expect(toasts.items.some((t) => t.id === toast!.id)).toBe(false);
  } finally {
    spaces.removeSpace(doomed);
    toasts.items = [];
  }
});
