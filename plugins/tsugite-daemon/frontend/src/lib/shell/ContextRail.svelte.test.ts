/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import ContextRail from './ContextRail.svelte';

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
