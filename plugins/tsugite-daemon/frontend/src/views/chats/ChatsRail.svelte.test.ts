/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import ChatsRail from './ChatsRail.svelte';
import { sessions } from '$lib/stores/sessions.svelte';
import { sessionRow as row } from './__fixtures__/sessionRow';
import { jobs, type Job } from '$lib/stores/jobs.svelte';
import { chatsNavBadge } from '$lib/shell/navBadges';
import { needsYouSessions } from './sessionModel';
import { TESTID } from '$lib/testids';

afterEach(() => {
  sessions.rows = [];
  jobs.jobs = [];
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
  sessions.rows = [row('s1', { title: 'backup prune' }), row('s2', { title: 'quiet chat' })];
  jobs.jobs = [job('j1', 's1', 'awaiting_input')];
  const { container } = await render(ChatsRail, base);
  await expect.element(page.getByTestId(TESTID.chatNeedsYou)).toHaveTextContent('needs you 1');
  expect(container.querySelectorAll('.t-srow.is-attn')).toHaveLength(1);
});

test('the chats nav badge counts chats waiting on an answer; the rail pill also counts parked-job chats', async () => {
  sessions.rows = [
    row('s1', { title: 'rent', needs_attention: true }),
    row('s2', { title: 'release' }),
    row('s3', { title: 'quiet chat' }),
    row('s4', { title: 'old news', status: 'completed', needs_attention: true }),
    row('s5', { title: 'pick one', progress: { status_text: 'Awaiting your answer' } as never }),
  ];
  jobs.jobs = [job('j1', 's2', 'awaiting_input')];
  const { container } = await render(ChatsRail, base);

  // s1 holds an unanswered card, s5 an outstanding question, and s2 owns the
  // parked job. s4 has ended, so its card is nobody's obligation any more.
  const badge = chatsNavBadge(needsYouSessions(sessions.rows).length);
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
    row('old', { title: 'long thread', needs_attention: true, superseded_by: 'new' }),
    row('new', { title: 'long thread', needs_attention: true }),
  ];
  const { container } = await render(ChatsRail, base);

  // The rail never lists the superseded row, so the badge must not count it.
  expect(chatsNavBadge(needsYouSessions(sessions.rows).length)[0]!.count).toBe(1);
  expect(container.querySelectorAll('.t-srow.is-attn')).toHaveLength(1);
});
