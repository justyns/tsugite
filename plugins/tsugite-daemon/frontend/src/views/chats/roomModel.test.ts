import { describe, expect, it } from 'vitest';
import type { Room } from '$lib/stores/rooms.svelte';
import { sessionRow } from './__fixtures__/sessionRow';
import {
  cardHref,
  messageTargets,
  pendingCapCard,
  roomBusy,
  workingParticipants,
  type RoomCard,
  type RoomEvent,
} from './roomModel';

function room(extra: Partial<Room> = {}): Room {
  return {
    id: 'r1',
    title: 'PROJ-123',
    participants: [
      { agent_id: 'atlas', session_id: 's-atlas', lead: true, muted: false },
      { agent_id: 'homelab', session_id: 's-homelab', lead: false, muted: false },
      { agent_id: 'coder', session_id: 's-coder', lead: false, muted: true },
    ],
    max_rounds: 12,
    rounds_used: 0,
    queued: [],
    job_id: null,
    needs_attention: false,
    attention: [],
    created_at: '2026-10-22T00:00:00Z',
    status: 'active',
    ...extra,
  };
}

describe('roomBusy', () => {
  it('is busy while any participant session runs', () => {
    expect(roomBusy(room(), [sessionRow('s-homelab', { busy: true })])).toBe(true);
    expect(roomBusy(room(), [sessionRow('s-homelab')])).toBe(false);
  });

  it('follows a compacted participant session to its successor', () => {
    const rows = [
      sessionRow('s-homelab', { superseded_by: 's-homelab-2' }),
      sessionRow('s-homelab-2', { busy: true }),
    ];
    expect(roomBusy(room(), rows)).toBe(true);
  });
});

describe('workingParticipants', () => {
  it('lists running and queued participants in roster order', () => {
    const rows = [sessionRow('s-homelab', { busy: true })];
    const ids = (r: Room) => workingParticipants(r, rows).map((p) => p.agent_id);
    expect(ids(room())).toEqual(['homelab']);
    expect(ids(room({ queued: ['homelab', 'atlas'] }))).toEqual(['atlas', 'homelab']);
  });

  it('makes the room busy while a turn is only queued', () => {
    expect(roomBusy(room({ queued: ['atlas'] }), [])).toBe(true);
  });
});

describe('messageTargets', () => {
  it('sends an unaddressed message to the lead', () => {
    expect(messageTargets('status?', room())).toEqual({ mentioned: [], lead: 'atlas' });
  });

  it('lists each mentioned participant once, in order, and skips the lead and strangers', () => {
    const text = '@coder then @homelab, cc @atlas @nobody and @coder again';
    expect(messageTargets(text, room()).mentioned).toEqual(['coder', 'homelab']);
  });

  it('ignores an @ inside a word, as the daemon does', () => {
    expect(messageTargets('mail ops@homelab', room()).mentioned).toEqual([]);
  });
});

describe('pendingCapCard', () => {
  const msg = (id: number): RoomEvent => ({
    type: 'room_message',
    id,
    timestamp: '',
    author: 'user',
    participant_session_id: null,
    content: 'hi',
    private_ref: null,
  });
  const cap: RoomEvent = { type: 'room_card', id: 2, timestamp: '', kind: 'round_cap', text: '' };
  const join: RoomEvent = {
    type: 'room_member',
    id: 3,
    timestamp: '',
    action: 'join',
    agent_id: 'x',
  };

  it('returns the cap card while it is the latest message or card', () => {
    expect(pendingCapCard([msg(1), cap, join])).toBe(cap);
  });

  it('clears once a message follows the card', () => {
    expect(pendingCapCard([cap, msg(4)])).toBeUndefined();
  });
});

describe('cardHref', () => {
  const card = (extra: Partial<RoomCard>): RoomCard => ({
    type: 'room_card',
    id: 1,
    timestamp: '2026-10-22T00:00:00Z',
    kind: 'job_status',
    text: 'Job running.',
    ...extra,
  });

  it('links a job card to the jobs board and a session card to its chat', () => {
    expect(cardHref(card({ job_id: 'job-1', agent_id: 'ops' }))).toBe('#jobs');
    expect(cardHref(card({ kind: 'attention', session_id: 's-ops' }))).toBe(
      '#chats?sessionId=s-ops',
    );
    expect(cardHref(card({ kind: 'round_cap' }))).toBeNull();
  });
});
