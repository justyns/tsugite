import type { Room, RoomParticipant } from '$lib/stores/rooms.svelte';
import type { SessionRow } from '$lib/stores/sessions.svelte';
import { liveSessionId } from './sessionModel';

export const USER_AUTHOR = 'user';

/** The daemon's room mention pattern (rooms.py _MENTION_RE). */
const MENTION_RE = /(?<![\w@])@([a-z0-9][a-z0-9_-]{0,63})/g;

export function participantBusy(p: RoomParticipant, rows: SessionRow[]): boolean {
  const id = liveSessionId(rows, p.session_id);
  return rows.some((row) => row.id === id && row.busy);
}

export function roomBusy(room: Room, rows: SessionRow[]): boolean {
  return room.participants.some((p) => participantBusy(p, rows));
}

/** The participants a user message runs: each @mentioned non-lead participant,
 *  then the lead. */
export function messageTargets(content: string, room: Room): { mentioned: string[]; lead: string } {
  const ids = new Set([...content.matchAll(MENTION_RE)].map((m) => m[1]!));
  const mentioned = [...ids].filter((id) =>
    room.participants.some((p) => p.agent_id === id && !p.lead),
  );
  return { mentioned, lead: room.participants.find((p) => p.lead)?.agent_id ?? '' };
}

export interface RoomMessage {
  type: 'room_message';
  id: number;
  timestamp: string;
  author: string;
  participant_session_id: string | null;
  content: string;
  private_ref: number | null;
}

export interface RoomMember {
  type: 'room_member';
  id: number;
  timestamp: string;
  action: 'join' | 'leave' | 'mute' | 'unmute';
  agent_id: string;
}

export interface RoomCard {
  type: 'room_card';
  id: number;
  timestamp: string;
  kind: string;
  text: string;
  agent_id?: string;
}

export type RoomEvent = RoomMessage | RoomMember | RoomCard;

const ROOM_EVENT_TYPES = new Set(['room_message', 'room_member', 'room_card']);

export function roomEvents(events: { type?: unknown }[]): RoomEvent[] {
  return events.filter((e) => ROOM_EVENT_TYPES.has(String(e.type))) as RoomEvent[];
}

/** The round_cap card when it is the latest message or card. The daemon posts
 *  one when a batch would pass the cap, which can happen before rounds_used
 *  reaches max_rounds. */
export function pendingCapCard(events: RoomEvent[]): RoomCard | undefined {
  const last = events.findLast((e) => e.type !== 'room_member');
  return last?.type === 'room_card' && last.kind === 'round_cap' ? last : undefined;
}
