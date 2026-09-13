/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import NavRail from './NavRail.svelte';
import type { ViewDef } from '../../views';
import { chatsNavBadge } from './navBadges';
// The peek test measures widths.
import '../../styles/tokens.css';

const views: ViewDef[] = [
  { id: 'chats', label: 'Chats', icon: 'chat', mode: 'workspace' },
  { id: 'jobs', label: 'Jobs', icon: 'jobs', mode: 'full' },
];

const base = { views, activeId: 'chats', onOpenSettings: vi.fn() };

const jobBadges = {
  jobs: [
    { count: 2, variant: 'info' as const, label: '2 jobs running' },
    { count: 1, variant: 'action' as const, label: '1 job needs you' },
  ],
};

test('a view with no live counts renders no badge', async () => {
  const { container } = await render(NavRail, base);
  expect(container.querySelector('.bdg')).toBeNull();
});

test("live counts render on their own view's row, named for a screen reader", async () => {
  const { container } = await render(NavRail, { ...base, badges: jobBadges });
  await expect.element(page.getByLabelText('2 jobs running')).toBeInTheDocument();
  await expect.element(page.getByLabelText('1 job needs you')).toBeInTheDocument();
  expect(container.querySelectorAll('[data-testid="nav-jobs"] .bdg .t-badge')).toHaveLength(2);
  expect(container.querySelector('[data-testid="nav-chats"] .bdg')).toBeNull();
});

test('the needs-you count is a different badge shape from the running count', async () => {
  const { container } = await render(NavRail, { ...base, badges: jobBadges });
  const [running, needsYou] = container.querySelectorAll('[data-testid="nav-jobs"] .t-badge');
  expect(running!.className).not.toContain('t-badge--act');
  expect(needsYou!.className).toContain('t-badge--act');
});

test('a chat waiting on you badges the chats row from any view', async () => {
  const { container } = await render(NavRail, {
    ...base,
    activeId: 'jobs',
    badges: { chats: chatsNavBadge(2) },
  });
  const badge = container.querySelector('[data-testid="nav-chats"] .t-badge');
  expect(badge!.textContent!.trim()).toBe('2');
  expect(badge!.className).toContain('t-badge--act');
  await expect.element(page.getByLabelText('2 chats need you')).toBeInTheDocument();
});

test('a collapsed rail still signals the rows that need you', async () => {
  const { container } = await render(NavRail, {
    ...base,
    collapsed: true,
    badges: { ...jobBadges, chats: chatsNavBadge(2) },
  });
  expect(container.querySelectorAll('.t-badge--dot')).toHaveLength(2);
  await expect.element(page.getByLabelText('2 jobs running, 1 job needs you')).toBeInTheDocument();
  await expect.element(page.getByLabelText('2 chats need you')).toBeInTheDocument();
});

test('a peeked collapsed rail shows its labels without widening its slot in the shell', async () => {
  // Collapse only exists above the phone breakpoint, where the rail is a column.
  await page.viewport(1280, 800);
  const { container } = await render(NavRail, {
    ...base,
    collapsed: true,
    peeking: true,
    onToggleCollapsed: vi.fn(),
  });
  const nav = container.querySelector('[data-testid="nav-rail"]') as HTMLElement;
  const body = container.querySelector('.rail-body') as HTMLElement;
  await expect.element(page.getByTestId('nav-chats').getByText('Chats')).toBeVisible();
  expect(nav.getBoundingClientRect().width).toBe(52);
  expect(body.getBoundingClientRect().width).toBe(198);
});

test('the pointer arriving on a collapsed rail asks for a peek', async () => {
  const onHoverStart = vi.fn();
  const { container } = await render(NavRail, { ...base, collapsed: true, onHoverStart });
  await userEvent.hover(container.querySelector('[data-testid="nav-rail"]') as HTMLElement);
  expect(onHoverStart).toHaveBeenCalled();
});

test('keyboard focus reaching a collapsed rail asks for a peek', async () => {
  const onHoverStart = vi.fn();
  const { container } = await render(NavRail, {
    ...base,
    collapsed: true,
    onToggleCollapsed: vi.fn(),
    onHoverStart,
  });
  (container.querySelector('.rail-collapse') as HTMLElement).focus();
  expect(onHoverStart).toHaveBeenCalled();
});

test('an expanded rail has nothing to peek, so hovering it asks for nothing', async () => {
  const onHoverStart = vi.fn();
  const { container } = await render(NavRail, { ...base, onHoverStart });
  await userEvent.hover(container.querySelector('[data-testid="nav-rail"]') as HTMLElement);
  expect(onHoverStart).not.toHaveBeenCalled();
});
