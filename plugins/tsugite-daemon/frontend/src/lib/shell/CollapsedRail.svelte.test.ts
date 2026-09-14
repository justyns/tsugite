/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import CollapsedRail from './CollapsedRail.svelte';

function props(over: Record<string, unknown> = {}) {
  return {
    view: 'chats' as const,
    peeking: false,
    onHoverStart: vi.fn(),
    onHoverEnd: vi.fn(),
    onPin: vi.fn(),
    focusedSessionId: null,
    focusedTerminalId: null,
    focusedFilePath: null,
    onOpenChat: vi.fn(),
    onOpenTerminal: vi.fn(),
    onOpenFile: vi.fn(),
    onPinFile: vi.fn(),
    ...over,
  };
}

test('at rest the collapsed rail is the strip alone', async () => {
  const { container } = await render(CollapsedRail, props());
  await expect.element(page.getByTestId('rail-expand')).toBeInTheDocument();
  expect(container.querySelector('[data-testid="rail-peek"]')).toBeNull();
});

test('a peek mounts the sessions rail in an overlay panel', async () => {
  const { container } = await render(CollapsedRail, props({ peeking: true }));
  const panel = container.querySelector('[data-testid="rail-peek"]');
  expect(panel).not.toBeNull();
  expect(panel!.querySelector('[data-testid="chat-rail"]')).not.toBeNull();
  expect(panel!.querySelector('[data-act="rail-collapse"]')!.getAttribute('aria-label')).toBe(
    'Expand sidebar',
  );
});

test('the pointer arriving on the strip asks for a peek, and leaving ends it', async () => {
  const p = props();
  await render(CollapsedRail, p);
  const strip = page.getByTestId('rail-expand');
  await userEvent.hover(strip);
  expect(p.onHoverStart).toHaveBeenCalled();
  expect(p.onHoverEnd).not.toHaveBeenCalled();
  await userEvent.unhover(strip);
  expect(p.onHoverEnd).toHaveBeenCalled();
});

test('keyboard focus reaching the strip asks for a peek too', async () => {
  const p = props();
  const { container } = await render(CollapsedRail, p);
  (container.querySelector('[data-testid="rail-expand"]') as HTMLElement).focus();
  expect(p.onHoverStart).toHaveBeenCalled();
});

test('clicking the strip pins the rail open', async () => {
  const p = props();
  await render(CollapsedRail, p);
  await page.getByTestId('rail-expand').click();
  expect(p.onPin).toHaveBeenCalled();
});

test('a peek draws its header arrow the way the strip draws its own', async () => {
  await page.viewport(1280, 800);
  const { container } = await render(CollapsedRail, props({ peeking: true }));
  const rotation = (icon: Element) => {
    const { rotate } = getComputedStyle(icon);
    return rotate === 'none' ? '0deg' : rotate;
  };
  expect(rotation(container.querySelector('.railc .ic')!)).toBe(
    rotation(container.querySelector('.rail-expand .ic')!),
  );
});

test('clicking the peeked header pins the rail open too', async () => {
  await page.viewport(1280, 800);
  const p = props({ peeking: true });
  const { container } = await render(CollapsedRail, p);
  await userEvent.click(container.querySelector('.rail-peek .railc') as HTMLElement);
  expect(p.onPin).toHaveBeenCalled();
});

test('unmounting a peeked rail releases the peek', async () => {
  const onHoverEnd = vi.fn();
  const { unmount } = await render(CollapsedRail, props({ peeking: true, onHoverEnd }));
  expect(onHoverEnd).not.toHaveBeenCalled();

  unmount();

  expect(onHoverEnd).toHaveBeenCalled();
});
