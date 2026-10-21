/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import ChatsRail from './ChatsRail.svelte';
import { sessions, type SessionRow } from '$lib/stores/sessions.svelte';
import { agent, sessionRow as row, attentionRecord as rec } from './__fixtures__/sessionRow';
import { jobs, type Job } from '$lib/stores/jobs.svelte';
import { chatsNavBadge } from '$lib/shell/navBadges';
import { chatsNeedingAnswer } from './sessionModel';
import { TESTID } from '$lib/testids';
import { api } from '$lib/api/client';
import { agents } from '$lib/stores/agents.svelte';

// The ended section persists its open/closed choice across tests.
beforeEach(() => {
  localStorage.removeItem('tsugite_rail_ended_open');
  vi.spyOn(agents, 'load').mockResolvedValue();
});

afterEach(() => {
  sessions.rows = [];
  agents.list = [];
  agents.error = null;
  agents['missed'].clear();
  jobs.jobs = [];
  vi.restoreAllMocks();
});

function job(id: string, parent: string, state: string): Job {
  return { job_id: id, parent_session_id: parent, state } as Job;
}

const base = { focusedSessionId: null, onOpenChat: vi.fn() };

test('a session with jobs still in flight shows the count on its tile', async () => {
  sessions.rows = [row('s1', { title: 'deploy pipeline' })];
  jobs.jobs = [job('j1', 's1', 'running'), job('j2', 's1', 'queued')];
  await render(ChatsRail, base);
  await expect.element(page.getByLabelText('2 active jobs')).toBeInTheDocument();
});

test('a resolved job leaves no count behind', async () => {
  sessions.rows = [row('s1', { title: 'deploy pipeline' })];
  jobs.jobs = [job('j1', 's1', 'done')];
  const { container } = await render(ChatsRail, base);
  await expect.element(page.getByText('deploy pipeline')).toBeInTheDocument();
  expect(container.querySelector('.mk .t-badge')).toBeNull();
});

test('a job parked on the person makes its session need you', async () => {
  sessions.rows = [
    row('s1', { title: 'backup prune', attention: [rec('job', { kind: 'awaiting_input' })] }),
    row('s2', { title: 'quiet chat' }),
  ];
  jobs.jobs = [job('j1', 's1', 'awaiting_input')];
  const { container } = await render(ChatsRail, base);
  await expect.element(page.getByTestId(TESTID.chatNeedsYou)).toHaveTextContent('needs you 1');
  expect(container.querySelectorAll('.t-srow.is-attn')).toHaveLength(1);
});

test('the chats nav badge counts chats waiting on an answer; the rail pill also counts parked-job chats', async () => {
  sessions.rows = [
    row('s1', { title: 'rent', attention: [rec('delivery', { kind: 'needs_ack' })] }),
    row('s2', { title: 'release', attention: [rec('job', { kind: 'awaiting_input' })] }),
    row('s3', { title: 'quiet chat' }),
    row('s4', {
      title: 'old news',
      status: 'completed',
      attention: [rec('delivery', { kind: 'needs_ack' })],
    }),
    row('s5', { title: 'pick one', attention: [rec('ask')] }),
  ];
  jobs.jobs = [job('j1', 's2', 'awaiting_input')];
  const { container } = await render(ChatsRail, base);

  // s1 holds an unanswered card, s5 an outstanding question, and s2 owns the
  // parked job. s4 has ended, so its card is nobody's obligation any more.
  const badge = chatsNavBadge(chatsNeedingAnswer(sessions.rows).length);
  // The badge counts what the chat itself waits on - the parked job is reported
  // on Jobs.
  expect(badge[0]!.count).toBe(2);
  // The pill is a filter chip for the list under it, so it also counts the chat
  // the parked job sits in. All three of those rows carry the alert glyph, which
  // is what keeps the badge from ever outrunning what the rail shows.
  await expect.element(page.getByTestId(TESTID.chatNeedsYou)).toHaveTextContent('needs you 3');
  expect(container.querySelectorAll('.t-srow.is-attn')).toHaveLength(3);
});

test('a compacted-away chat leaves its card to the successor the rail lists', async () => {
  sessions.rows = [
    row('old', {
      title: 'long thread',
      attention: [rec('delivery', { kind: 'needs_ack' })],
      superseded_by: 'new',
    }),
    row('new', { title: 'long thread', attention: [rec('delivery', { kind: 'needs_ack' })] }),
  ];
  const { container } = await render(ChatsRail, base);

  // The rail never lists the superseded row, so the badge must not count it.
  expect(chatsNavBadge(chatsNeedingAnswer(sessions.rows).length)[0]!.count).toBe(1);
  expect(container.querySelectorAll('.t-srow.is-attn')).toHaveLength(1);
});

test('clicking the chat already on screen marks it read', async () => {
  sessions.rows = [row('s1', { title: 'open chat', unread: true })];
  const post = vi.spyOn(api, 'post').mockResolvedValue({ ok: true } as never);
  const onOpenChat = vi.fn();
  await render(ChatsRail, { focusedSessionId: 's1', onOpenChat });

  await page.getByText('open chat').click();

  expect(post).toHaveBeenCalledWith('/api/sessions/s1/mark-viewed');
  expect(sessions.rows[0]!.unread).toBe(false);
  expect(onOpenChat).toHaveBeenCalledWith('s1');
});

test('clicking a read chat posts nothing', async () => {
  sessions.rows = [row('s1', { title: 'read chat' })];
  const post = vi.spyOn(api, 'post');
  await render(ChatsRail, { focusedSessionId: 's1', onOpenChat: vi.fn() });

  await page.getByText('read chat').click();

  expect(post).not.toHaveBeenCalled();
});

function run(id: string, scheduleId: string, minute: number, extra: Partial<SessionRow> = {}) {
  return row(id, {
    source: 'schedule',
    parent_id: scheduleId,
    status: 'completed',
    last_active: `2026-07-17T00:${String(minute).padStart(2, '0')}:00Z`,
    ...extra,
  });
}

test('every run of one schedule collapses to a single rail row with its run count', async () => {
  sessions.rows = [
    run('r1', 'dev-land', 10),
    run('r2', 'dev-land', 20),
    run('r3', 'dev-land', 30),
    run('k1', 'kuma-ingest', 15),
  ];
  const { container } = await render(ChatsRail, base);
  await page.getByRole('button', { name: /ended 2/ }).click();
  expect(container.querySelectorAll('.t-srow')).toHaveLength(2);
  await expect.element(page.getByLabelText('3 runs')).toBeInTheDocument();
  await expect.element(page.getByLabelText('1 run', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('kuma-ingest')).toBeInTheDocument();
});

test('clicking a collapsed schedule row opens its latest run', async () => {
  sessions.rows = [run('r1', 'dev-land', 10), run('r2', 'dev-land', 20)];
  const onOpenChat = vi.fn();
  vi.spyOn(api, 'post').mockResolvedValue({ ok: true } as never);
  await render(ChatsRail, { focusedSessionId: null, onOpenChat });
  await page.getByRole('button', { name: /ended 1/ }).click();
  await page.getByRole('button', { name: /^dev-land,/ }).click();
  expect(onOpenChat).toHaveBeenCalledWith('r2');
});

test("a collapsed row's schedule link opens the Schedules view on that schedule", async () => {
  sessions.rows = [run('r1', 'dev-land', 10)];
  await render(ChatsRail, base);
  await page.getByRole('button', { name: /ended 1/ }).click();
  await page.getByTestId(TESTID.sessionOpenSchedule).click();
  expect(location.hash).toBe('#schedules?id=dev-land');
});

test('a run with open attention flags the collapsed row and counts once toward needs you', async () => {
  sessions.rows = [
    run('r1', 'dev-land', 10, { status: 'active', attention: [rec('ask')] }),
    run('r2', 'dev-land', 20, { status: 'active', attention: [rec('ask')] }),
    run('r3', 'dev-land', 30, { status: 'active' }),
  ];
  const { container } = await render(ChatsRail, base);
  expect(container.querySelectorAll('.t-srow')).toHaveLength(1);
  expect(container.querySelectorAll('.t-srow.is-attn')).toHaveLength(1);
  await expect.element(page.getByTestId(TESTID.chatNeedsYou)).toHaveTextContent('needs you 1');
  await page.getByLabelText('Filter sessions').fill('is:needs-you');
  await expect.element(page.getByText('dev-land')).toBeInTheDocument();
});

test('a collapsed row stays live and opens the waiting run when an older run needs you', async () => {
  sessions.rows = [
    run('r1', 'dev-land', 10, { status: 'active', attention: [rec('ask')] }),
    run('r2', 'dev-land', 20),
  ];
  const onOpenChat = vi.fn();
  vi.spyOn(api, 'post').mockResolvedValue({ ok: true } as never);
  const { container } = await render(ChatsRail, { focusedSessionId: null, onOpenChat });
  await expect.element(page.getByRole('button', { name: /^dev-land,/ })).toBeVisible();
  expect(container.querySelectorAll('.t-srow.is-attn')).toHaveLength(1);
  await page.getByRole('button', { name: /^dev-land,/ }).click();
  expect(onOpenChat).toHaveBeenCalledWith('r1');
});

test('a new run arriving over the live feed updates the existing row', async () => {
  let served = [run('r1', 'dev-land', 10)];
  vi.spyOn(api, 'get').mockImplementation(async () => ({ sessions: served }) as never);
  const { container } = await render(ChatsRail, base);
  await page.getByRole('button', { name: /ended 1/ }).click();
  await expect.element(page.getByLabelText('1 run', { exact: true })).toBeInTheDocument();

  served = [run('r2', 'dev-land', 20, { status: 'active', busy: true }), ...served];
  sessions.applySessionUpdate({ action: 'created', id: 'r2' });

  await expect.element(page.getByLabelText('2 runs')).toBeInTheDocument();
  expect(container.querySelectorAll('.t-srow')).toHaveLength(1);
});

test('two agents nest the sessions under agent headers', async () => {
  agents.list = [agent('main', 'atlas', { is_default: true }), agent('ops', 'homelab')];
  sessions.rows = [row('s1', { title: 'deploy pipeline' })];
  await render(ChatsRail, base);
  await expect.element(page.getByRole('button', { name: /^atlas/ })).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: /^homelab/ })).toBeInTheDocument();
});

test('a session of an agent the list lacks refetches the agents', async () => {
  agents.list = [agent('main', 'atlas', { is_default: true }), agent('ops', 'homelab')];
  sessions.rows = [row('s1', { title: 'deploy pipeline', agent_id: 'coding' })];
  await render(ChatsRail, base);
  await expect.poll(() => vi.mocked(agents.load).mock.calls.length).toBe(2);
});

test('a failed agent list shows a note whose Retry reloads it', async () => {
  agents.error = 'daemon down';
  sessions.rows = [row('s1', { title: 'deploy pipeline' })];
  await render(ChatsRail, base);
  await expect.element(page.getByText("Couldn't load agents")).toBeInTheDocument();
  const calls = vi.mocked(agents.load).mock.calls.length;
  await page.getByRole('button', { name: 'Retry' }).click();
  expect(vi.mocked(agents.load).mock.calls.length).toBe(calls + 1);
});
