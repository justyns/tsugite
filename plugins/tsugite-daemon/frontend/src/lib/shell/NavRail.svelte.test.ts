/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import NavRail from './NavRail.svelte';
import type { ViewDef } from '../../views';
import { chatsNavBadge } from './navBadges';

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
