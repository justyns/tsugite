/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import ContextRail from './ContextRail.svelte';
import { agents } from '$lib/stores/agents.svelte';

beforeEach(() => vi.spyOn(agents, 'load').mockResolvedValue());
afterEach(() => vi.restoreAllMocks());

function props(over: Record<string, unknown> = {}) {
  return {
    view: 'chats' as const,
    onCollapse: vi.fn(),
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

// 'none' and '0deg' both draw the chevron pointing right.
function rotation(icon: Element) {
  const { rotate } = getComputedStyle(icon);
  return rotate === 'none' ? '0deg' : rotate;
}

test('pinned open, the header control points the collapse direction', async () => {
  await page.viewport(1280, 800);
  const { container } = await render(ContextRail, props());
  const button = container.querySelector('.railc') as HTMLElement;
  expect(rotation(button.querySelector('.ic')!)).toBe('180deg');
  expect(button.getAttribute('aria-pressed')).toBe('false');
});

test('peeked, the header control points the expand direction', async () => {
  await page.viewport(1280, 800);
  const { container } = await render(ContextRail, props({ peeking: true }));
  const button = container.querySelector('.railc') as HTMLElement;
  expect(rotation(button.querySelector('.ic')!)).toBe('0deg');
  expect(button.getAttribute('aria-pressed')).toBe('true');
});

test('dragging the resize handle sets the rail width, and a double-click resets it', async () => {
  await page.viewport(1280, 800);
  const { railWidth } = await import('$lib/stores/railWidth.svelte');
  railWidth.reset();
  const { container } = await render(ContextRail, props({ resizable: true }));
  const rail = container.querySelector('.work-rail') as HTMLElement;
  rail.style.width = '262px';
  const handle = page.getByRole('separator', { name: 'Resize sidebar' }).element() as HTMLElement;
  const at = (type: string, clientX: number) =>
    handle.dispatchEvent(new PointerEvent(type, { clientX, pointerId: 1, bubbles: true }));

  at('pointerdown', 262);
  at('pointermove', 362);
  at('pointerup', 362);
  expect(railWidth.px).toBe(362);
  expect(localStorage.getItem('tsugite_rail_width')).toBe('362');

  handle.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
  expect(railWidth.px).toBeNull();
  expect(localStorage.getItem('tsugite_rail_width')).toBeNull();
});

test('arrow keys on the resize handle step the width, never below the minimum', async () => {
  const { railWidth } = await import('$lib/stores/railWidth.svelte');
  railWidth.reset();
  const { container } = await render(ContextRail, props({ resizable: true }));
  (container.querySelector('.work-rail') as HTMLElement).style.width = '210px';
  const handle = page.getByRole('separator', { name: 'Resize sidebar' });

  await handle.element().focus();
  await userEvent.keyboard('{ArrowRight}');
  expect(railWidth.px).toBe(226);
  (container.querySelector('.work-rail') as HTMLElement).style.width = '200px';
  await userEvent.keyboard('{ArrowLeft}');
  expect(railWidth.px).toBe(200);
  railWidth.reset();
});

test('without resizable there is no resize handle', async () => {
  await render(ContextRail, props());
  expect(page.getByRole('separator', { name: 'Resize sidebar' }).elements()).toHaveLength(0);
});
