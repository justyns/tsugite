/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { beforeEach, expect, test, vi } from 'vitest';
import SessionsRail from './SessionsRail.svelte';
import { spaces } from '$lib/stores/spaces.svelte';
import { collectLeaves } from '$lib/shell/mux/layout';
import { TESTID } from '$lib/testids';
import type { SessionRow } from '$lib/stores/sessions.svelte';

// The ended section persists its open/closed choice; reset it so each test
// starts from the collapsed default.
beforeEach(() => localStorage.removeItem('tsugite_rail_ended_open'));

function row(id: string, extra: Partial<SessionRow> = {}): SessionRow {
  return {
    id,
    user_id: 'u',
    label: id,
    source: 'web',
    status: 'active',
    state: 'idle',
    created_at: '2026-07-17T00:00:00Z',
    last_active: '2026-07-17T00:00:00Z',
    parent_id: null,
    prompt: '',
    model: null,
    error: null,
    result: null,
    title: id,
    is_default: false,
    metadata: {},
    pinned: false,
    pin_position: null,
    last_viewed_at: null,
    superseded_by: null,
    unread: false,
    is_primary: false,
    busy: false,
    ...extra,
  };
}

const base = {
  agent: 'smokeagent',
  selectedId: null,
  attn: new Set<string>(),
  onSelect: vi.fn(),
  onNew: vi.fn(),
  onServerSearch: vi.fn(),
};

test('right-clicking a session row opens the actions menu', async () => {
  render(SessionsRail, { ...base, rows: [row('s1', { title: 'sse backoff' })] });
  await page.getByRole('button', { name: /sse backoff/ }).click({ button: 'right' });
  const menu = page.getByRole('menu', { name: 'Session actions' });
  await expect.element(menu).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Open in new tab' })).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Pin' })).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Copy session id' })).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Mark complete' })).toBeInTheDocument();

  // Ended sessions offer no lifecycle action; pinned rows offer Unpin.
  await userEvent.keyboard('{Escape}');
  await expect.element(menu).not.toBeInTheDocument();
});

test('"Open in new tab" docks the session as a fresh chat tab', async () => {
  render(SessionsRail, { ...base, rows: [row('s2', { title: 'backup prune' })] });
  const before = collectLeaves(spaces.active.layout.root).flatMap((l) => l.tabs).length;

  await page.getByRole('button', { name: /backup prune/ }).click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Open in new tab' }).click();

  const tabs = collectLeaves(spaces.active.layout.root).flatMap((l) => l.tabs);
  expect(tabs.length).toBe(before + 1);
  const docked = tabs[tabs.length - 1]!;
  expect(docked.kind).toBe('chat');
  expect(docked.params.sessionId).toBe('s2');
});

test('finished sessions tuck into a collapsed "ended" section with a count', async () => {
  render(SessionsRail, {
    ...base,
    rows: [
      row('l', { title: 'live draft' }),
      row('c', { title: 'wrapped up sync', status: 'completed' }),
      row('f', { title: 'blew up mid run', status: 'failed' }),
    ],
  });
  // the live row stays in the main recency flow
  await expect.element(page.getByRole('button', { name: /live draft/ })).toBeInTheDocument();
  // finished rows leave that flow for a collapsed section that counts them
  const toggle = page.getByRole('button', { name: 'ended 2' });
  await expect.element(toggle).toBeInTheDocument();
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect.element(page.getByText('wrapped up sync')).not.toBeInTheDocument();
  await expect.element(page.getByText('blew up mid run')).not.toBeInTheDocument();
});

test('expanding "ended" reveals the finished rows, visibly muted', async () => {
  render(SessionsRail, {
    ...base,
    rows: [row('c', { title: 'wrapped up sync', status: 'completed' })],
  });
  await page.getByRole('button', { name: 'ended 1' }).click();
  const rowBtn = page.getByRole('button', { name: /wrapped up sync/ });
  await expect.element(rowBtn).toBeInTheDocument();
  await expect.element(rowBtn).toHaveClass(/is-ended/);
});

test('a finished session is never counted as needs-you', async () => {
  render(SessionsRail, {
    ...base,
    attn: new Set(['c']),
    rows: [row('c', { title: 'wrapped up sync', status: 'completed' })],
  });
  await expect.element(page.getByTestId(TESTID.chatNeedsYou)).toHaveTextContent('needs you 0');
});

test('a status: filter surfaces finished sessions even though ended is collapsed by default', async () => {
  render(SessionsRail, {
    ...base,
    rows: [
      row('l', { title: 'live draft' }),
      row('c', { title: 'wrapped up sync', status: 'completed' }),
    ],
  });
  await page.getByLabelText('Filter sessions').fill('status:completed');
  await expect.element(page.getByText('wrapped up sync')).toBeInTheDocument();
  await expect.element(page.getByText('live draft')).not.toBeInTheDocument();
});
