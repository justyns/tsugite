import { describe, expect, it } from 'vitest';
import type { Room } from '$lib/stores/rooms.svelte';
import { sessionRow } from './__fixtures__/sessionRow';
import { messageTargets, pendingCapCard, roomBusy, type RoomEvent } from './roomModel';

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
