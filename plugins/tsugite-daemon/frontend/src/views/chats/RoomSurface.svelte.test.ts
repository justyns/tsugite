/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { cleanup, render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import RoomSurface from './RoomSurface.svelte';
import { api } from '$lib/api/client';
import { rooms, type Room } from '$lib/stores/rooms.svelte';
import { agents } from '$lib/stores/agents.svelte';
import { sessions } from '$lib/stores/sessions.svelte';
import { jobs } from '$lib/stores/jobs.svelte';
import { jobDrawerRequest } from '../jobs/jobDrawerSignal.svelte';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { agent, sessionRow } from './__fixtures__/sessionRow';
import '../../styles/tokens.css';

afterEach(async () => {
  await cleanup();
  vi.restoreAllMocks();
  rooms.list = [];
  rooms.loaded = false;
  rooms.error = null;
  sessions.rows = [];
});

const room: Room = {
  id: 'r1',
  title: 'PROJ-123',
  participants: [
    { agent_id: 'main', session_id: 'p-main', lead: true, muted: false },
    { agent_id: 'ops', session_id: 'p-ops', lead: false, muted: false },
  ],
  max_rounds: 12,
  rounds_used: 3,
  queued: [],
  job_id: null,
  needs_attention: false,
  attention: [],
  created_at: '2026-10-22T00:00:00Z',
  status: 'active',
};
const room2: Room = { ...room, id: 'r2', title: 'second room' };

const at = '2026-10-22T12:00:00Z';
const message = (id: number, author: string, content: string, extra = {}) => ({
  id,
  type: 'room_message',
  timestamp: at,
  author,
  participant_session_id: author === 'user' ? null : `p-${author}`,
  content,
  private_ref: null,
  ...extra,
});
const events = [
  { id: 1, type: 'room_member', timestamp: at, action: 'join', agent_id: 'ops' },
  message(2, 'user', 'check the disks'),
  message(3, 'ops', 'all disks healthy', { private_ref: 7 }),
  { id: 4, type: 'room_card', timestamp: at, kind: 'turn_failed', agent_id: 'ops', text: 'boom' },
];

type Page = { events: unknown[]; has_more?: boolean; oldest_id?: number | null };

function stubApi(eventsFor: (path: string) => Page | Promise<Page> = () => ({ events })) {
  const rows = [sessionRow('p-main'), sessionRow('p-ops', { busy: true })];
  return vi.spyOn(api, 'get').mockImplementation(async (path: string) => {
    if (path.startsWith('/api/rooms')) return { rooms: [room, room2] } as never;
    if (path.startsWith('/api/agents'))
      return {
        agents: [
          agent('main', 'atlas', { is_default: true, color: 'blue', model: 'opus' }),
          agent('ops', 'homelab', { color: 'green' }),
        ],
      } as never;
    if (path.startsWith('/api/chat/sessions')) return { sessions: rows } as never;
    return (await eventsFor(path)) as never;
  });
}

async function mount(eventsFor?: (path: string) => Page | Promise<Page>) {
  const get = stubApi(eventsFor);
  await rooms.load();
  await agents.load();
  const screen = await render(RoomSurface, { params: { roomId: 'r1' } });
  return { get, screen };
}

const transcript = () => page.getByRole('region', { name: 'Room' });

test('room messages show their author, and system rows show joins and failures', async () => {
  await mount();
  await expect.element(page.getByText('all disks healthy')).toBeInTheDocument();
  await expect
    .element(page.getByRole('article').filter({ hasText: 'check the disks' }))
    .toHaveTextContent(/^you/);
  await expect
    .element(page.getByRole('article').filter({ hasText: 'all disks healthy' }))
    .toHaveTextContent(/^homelab/);
  expect(document.querySelector('.room-scroll [data-color="green"]')).not.toBeNull();
  await expect.element(page.getByText('homelab joined')).toBeInTheDocument();
  await expect.element(page.getByText('homelab: boom')).toBeInTheDocument();
  const link = page.getByRole('link', { name: 'worked in homelab ›' });
  await expect.element(link).toHaveAttribute('href', '#chats?sessionId=p-ops');
});

test("an opener's post shows its agent as the opener", async () => {
  await mount(() => ({ events: [message(1, 'main', 'how is it going?', { role: 'opener' })] }));
  await expect
    .element(page.getByRole('article').filter({ hasText: 'how is it going?' }))
    .toHaveTextContent(/^atlas \(opener\)/);
});

test('participant chips show name, model, lead, busy and a mute toggle for non-leads', async () => {
  await mount();
  const chips = page.getByRole('group', { name: 'Participants' });
  await expect.element(chips).toHaveTextContent(/atlas\s*opus\s*lead/);
  await expect.element(page.getByTitle('@ops')).toHaveTextContent('homelab');
  expect(page.getByTitle('working').elements()).toHaveLength(1);
  const patch = vi.spyOn(api, 'patch').mockResolvedValue({
    ...room,
    participants: [room.participants[0], { ...room.participants[1], muted: true }],
  } as never);
  await page.getByRole('button', { name: 'Mute homelab' }).click();
  expect(patch).toHaveBeenCalledWith('/api/rooms/r1/participants/ops', { muted: true });
  await expect
    .element(page.getByRole('button', { name: 'Mute homelab' }))
    .toHaveAttribute('aria-pressed', 'true');
});

test('a working line above the composer lists running and queued participants', async () => {
  await mount();
  const line = page.getByRole('status').filter({ hasText: 'working' });
  await expect.element(line).toHaveTextContent('homelab is working…');
  rooms.applySessionUpdate({ action: 'room_updated', room: { ...room, queued: ['main'] } });
  await expect.element(line).toHaveTextContent('atlas and homelab are working…');
});

test('tapping a participant chip mentions it, and its open link opens its session', async () => {
  await mount();
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.type(box, 'hi');
  await page.getByRole('button', { name: 'Mention homelab' }).click();
  await expect.element(box).toHaveValue('hi @ops ');
  await expect.element(box).toHaveFocus();
  await expect
    .element(page.getByRole('link', { name: 'Open homelab session' }))
    .toHaveAttribute('href', '#chats?sessionId=p-ops');
});

test('the lead chip opens the lead session instead of mentioning it', async () => {
  await mount();
  await expect
    .element(page.getByRole('link', { name: 'Open atlas session' }))
    .toHaveAttribute('href', '#chats?sessionId=p-main');
  expect(page.getByRole('button', { name: 'Mention atlas' }).elements()).toHaveLength(0);
});

test('the target line follows the mentions in the draft', async () => {
  await mount();
  await expect.element(page.getByText('to lead atlas')).toBeInTheDocument();
  await userEvent.type(page.getByRole('textbox', { name: 'Message' }), '@ops look');
  await expect.element(page.getByText('to homelab, then lead atlas')).toBeInTheDocument();
});

test('sending posts the draft to the room and clears it', async () => {
  await mount();
  const post = vi.spyOn(api, 'post').mockResolvedValue(room as never);
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.type(box, 'hello room{Enter}');
  expect(post).toHaveBeenCalledWith('/api/rooms/r1/messages', { content: 'hello room' });
  await expect.element(box).toHaveValue('');
});

test('a failed send puts the message back in the composer', async () => {
  await mount();
  const push = vi.spyOn(toasts, 'push');
  vi.spyOn(api, 'post').mockRejectedValue(new Error('Room r1 is closed'));
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.type(box, 'hello room{Enter}');
  await expect.element(box).toHaveValue('hello room');
  expect(push).toHaveBeenCalledWith('err', 'Could not post to the room', {
    body: 'Room r1 is closed',
  });
});

test('a live room frame appends the event it carries', async () => {
  const { get } = await mount();
  await expect.element(page.getByText('all disks healthy')).toBeInTheDocument();
  const calls = get.mock.calls.length;
  sessions.applySessionEvent({
    session_id: 'r1',
    event_type: 'room_message',
    ...message(5, 'ops', 'pool trimmed'),
  });
  await expect.element(page.getByText('pool trimmed')).toBeInTheDocument();
  expect(get.mock.calls.length).toBe(calls);
});

test('a late response for the previous room never lands in the next one', async () => {
  let release!: (page: Page) => void;
  const { screen } = await mount((path) =>
    path.includes('/r1/')
      ? new Promise<Page>((resolve) => (release = resolve))
      : { events: [message(9, 'user', 'second room hello')] },
  );
  await screen.rerender({ params: { roomId: 'r2' } });
  await expect.element(page.getByText('second room hello')).toBeInTheDocument();
  release({ events });
  await new Promise((r) => setTimeout(r, 50));
  await expect.element(page.getByText('all disks healthy')).not.toBeInTheDocument();
});

test('a failed load shows the error with a retry', async () => {
  let fail = true;
  await mount(() => {
    if (fail) throw new Error('daemon down');
    return { events };
  });
  await expect.element(page.getByText('daemon down')).toBeInTheDocument();
  fail = false;
  await page.getByRole('button', { name: 'Retry' }).click();
  await expect.element(page.getByText('all disks healthy')).toBeInTheDocument();
});

test('a room list that fails to load shows the error with a retry', async () => {
  stubApi();
  rooms.error = 'daemon down';
  rooms.loaded = true;
  render(RoomSurface, { params: { roomId: 'r1' } });
  await expect.element(page.getByText("Couldn't load rooms: daemon down")).toBeInTheDocument();
  await page.getByRole('button', { name: 'Retry' }).click();
  await expect.element(page.getByRole('group', { name: 'Participants' })).toBeInTheDocument();
});

test('older events load on demand', async () => {
  await mount((path) =>
    path.includes('before_id=2')
      ? { events: [message(1, 'user', 'the first message')], has_more: false, oldest_id: 1 }
      : { events: events.slice(1), has_more: true, oldest_id: 2 },
  );
  await expect.element(page.getByText('the first message')).not.toBeInTheDocument();
  await page.getByRole('button', { name: 'load earlier messages' }).click();
  await expect.element(page.getByText('the first message')).toBeInTheDocument();
  await expect
    .element(page.getByRole('button', { name: 'load earlier messages' }))
    .not.toBeInTheDocument();
});

test('a closed room hides the composer and the mute toggles', async () => {
  stubApi();
  vi.mocked(api.get).mockImplementation(async (path: string) =>
    path.startsWith('/api/rooms')
      ? ({ rooms: [{ ...room, status: 'completed' }] } as never)
      : ({ events, sessions: [], agents: [] } as never),
  );
  await rooms.load();
  render(RoomSurface, { params: { roomId: 'r1' } });
  await expect.element(transcript()).toHaveTextContent(/closed/);
  await expect.element(page.getByRole('textbox', { name: 'Message' })).not.toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: /^Mute/ })).not.toBeInTheDocument();
});

test('a round-cap card offers to raise the cap', async () => {
  await mount(() => ({
    events: [
      ...events,
      { id: 5, type: 'room_card', timestamp: at, kind: 'round_cap', text: 'Round cap reached' },
    ],
  }));
  const patch = vi.spyOn(api, 'patch').mockResolvedValue({ ...room, max_rounds: 24 } as never);
  await page.getByRole('button', { name: '+12 rounds' }).click();
  expect(patch).toHaveBeenCalledWith('/api/rooms/r1', { max_rounds: 24 });
  await expect.element(page.getByText('rounds 3/24')).toBeInTheDocument();
  sessions.applySessionEvent({
    session_id: 'r1',
    event_type: 'room_card',
    id: 6,
    kind: 'cap_raised',
    max_rounds: 24,
    text: 'Round cap raised to 24.',
  });
  await expect.element(page.getByText('Round cap raised to 24.')).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: '+12 rounds' })).not.toBeInTheDocument();
});

test('a room_updated broadcast refreshes the header and chips', async () => {
  await mount();
  await expect.element(page.getByText('rounds 3/12')).toBeInTheDocument();
  rooms.applySessionUpdate({
    action: 'room_updated',
    id: 'r1',
    room: {
      ...room,
      rounds_used: 4,
      participants: [room.participants[0], { ...room.participants[1], muted: true }],
    },
  });
  await expect.element(page.getByText('rounds 4/12')).toBeInTheDocument();
  await expect
    .element(page.getByRole('button', { name: 'Mute homelab' }))
    .toHaveAttribute('aria-pressed', 'true');
});

test('skipped and interrupted turns show as system rows', async () => {
  await mount(() => ({
    events: [
      ...events,
      {
        id: 5,
        type: 'room_card',
        timestamp: at,
        kind: 'turn_skipped',
        agent_id: 'ops',
        text: 'muted',
      },
      {
        id: 6,
        type: 'room_card',
        timestamp: at,
        kind: 'interrupted',
        text: 'Stopped by the user.',
      },
    ],
  }));
  await expect.element(page.getByText('homelab: muted')).toBeInTheDocument();
  await expect.element(page.getByText('Stopped by the user.')).toBeInTheDocument();
});

test('a used-up round count offers to raise the cap without a card', async () => {
  stubApi();
  vi.spyOn(api, 'get').mockImplementation(async (path: string) =>
    path.startsWith('/api/rooms')
      ? ({ rooms: [{ ...room, rounds_used: 12 }] } as never)
      : ({ events, sessions: [], agents: [] } as never),
  );
  await rooms.load();
  render(RoomSurface, { params: { roomId: 'r1' } });
  await expect.element(page.getByRole('button', { name: '+12 rounds' })).toBeInTheDocument();
});

test('the header shows the linked job, and cards link to the job or session they reference', async () => {
  const job = { job_id: 'job-1', state: 'verifying', prompt: 'free up NAS disk' };
  const card = (id: number, extra: object) => ({ id, type: 'room_card', timestamp: at, ...extra });
  stubApi((path) =>
    path === '/api/jobs/job-1'
      ? (job as never)
      : {
          events: [
            card(5, { kind: 'job_failed', job_id: 'job-1', state: 'stuck', text: 'Job stuck.' }),
            card(6, { kind: 'job_done', job_id: 'job-1', state: 'done', text: 'Job done.' }),
            card(7, {
              kind: 'attention',
              session_id: 'p-ops',
              agent_id: 'ops',
              text: 'Needs you (ask).',
            }),
          ],
        },
  );
  await rooms.load();
  rooms.list = [{ ...room, job_id: 'job-1' }];
  await agents.load();
  render(RoomSurface, { params: { roomId: 'r1' } });

  try {
    const header = page.getByRole('link', { name: /free up NAS disk/ });
    await expect.element(header).toHaveAttribute('href', '#jobs');
    await expect.element(header).toHaveTextContent('verifying');
    await header.click();
    expect(jobDrawerRequest.pending).toBe('job-1');
    jobDrawerRequest.consume();
    const done = page.getByRole('link', { name: 'Job done.' });
    await expect.element(done).toHaveAttribute('href', '#jobs');
    expect(done.element().closest('.sys')!.classList).toContain('is-ok');
    await done.click();
    expect(jobDrawerRequest.pending).toBe('job-1');
    const failed = page.getByRole('link', { name: 'Job stuck.' });
    expect(failed.element().closest('.sys')!.classList).toContain('is-err');
    await expect
      .element(page.getByRole('link', { name: 'homelab: Needs you (ask).' }))
      .toHaveAttribute('href', '#chats?sessionId=p-ops');
  } finally {
    jobs.jobs = [];
    jobDrawerRequest.consume();
    window.location.hash = '';
  }
});

test('a failed job fetch is not retried on every room update', async () => {
  const get = stubApi((path) =>
    path === '/api/jobs/gone' ? Promise.reject(new Error('404')) : { events },
  );
  await rooms.load();
  rooms.list = [{ ...room, job_id: 'gone' }];
  await agents.load();
  render(RoomSurface, { params: { roomId: 'r1' } });
  await expect.element(page.getByText('all disks healthy')).toBeInTheDocument();

  rooms.applySessionUpdate({
    action: 'room_updated',
    room: { ...room, job_id: 'gone', rounds_used: 4 },
  });
  rooms.applySessionUpdate({
    action: 'room_updated',
    room: { ...room, job_id: 'gone', rounds_used: 5 },
  });
  await expect.element(page.getByText('rounds 5/12')).toBeInTheDocument();

  expect(get.mock.calls.filter(([path]) => path === '/api/jobs/gone')).toHaveLength(1);
});
