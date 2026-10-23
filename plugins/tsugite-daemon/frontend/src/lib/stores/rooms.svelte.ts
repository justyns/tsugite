/** Rooms store (GET /api/rooms). A room's id is also its transcript session's id. */
import { api } from '$lib/api/client';
import type { SurfaceRef } from '$lib/shell/mux/layout';
import type { AttentionRecord } from './sessions.svelte';

export interface RoomParticipant {
  agent_id: string;
  session_id: string;
  lead: boolean;
  muted: boolean;
}

export interface Room {
  id: string;
  title: string;
  participants: RoomParticipant[];
  max_rounds: number;
  rounds_used: number;
  created_at: string;
  /** The transcript session's status. A finished status means the room is closed. */
  status: string;
  /** The room closes when this Job's acceptance criteria verify. */
  job_id: string | null;
  /** Open attention records on the room and its participant sessions. */
  needs_attention: boolean;
  attention: AttentionRecord[];
}

function roomPath(id: string, rest = ''): string {
  return `/api/rooms/${encodeURIComponent(id)}${rest}`;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export class RoomsStore {
  list = $state<Room[]>([]);
  loaded = $state(false);
  error = $state<string | null>(null);

  private inflight: Promise<void> | null = null;

  /** Fetches the list, or joins the fetch already in flight. Rejects when it fails. */
  load(): Promise<void> {
    this.inflight ??= this.fetchList();
    return this.inflight;
  }

  private async fetchList(): Promise<void> {
    try {
      const res = await api.get<{ rooms: Room[] }>('/api/rooms');
      this.list = res.rooms;
      this.error = null;
      this.loaded = true;
    } catch (err) {
      this.error = message(err);
      throw err;
    } finally {
      this.inflight = null;
    }
  }

  byId(id: string): Room | undefined {
    return this.list.find((r) => r.id === id);
  }

  /** The room surface for `id`, or null when `id` is not a room. Rejects when
   *  the list cannot be loaded. */
  async surfaceRef(id: string): Promise<SurfaceRef | null> {
    if (!this.loaded) await this.load();
    const room = this.byId(id);
    return room ? { kind: 'room', params: { roomId: room.id }, title: room.title } : null;
  }

  /** Applies a `room_updated` broadcast, whose payload is the whole room. */
  applySessionUpdate(data: Record<string, unknown>): void {
    if (data.action === 'room_updated') this.upsert(data.room as Room);
  }

  async post(id: string, content: string): Promise<void> {
    await api.post(roomPath(id, '/messages'), { content });
  }

  async stop(id: string): Promise<void> {
    await this.replace(api.post<Room>(roomPath(id, '/stop')));
  }

  async setMuted(id: string, agentId: string, muted: boolean): Promise<void> {
    await this.replace(
      api.patch<Room>(roomPath(id, `/participants/${encodeURIComponent(agentId)}`), { muted }),
    );
  }

  async setMaxRounds(id: string, maxRounds: number): Promise<void> {
    await this.replace(api.patch<Room>(roomPath(id), { max_rounds: maxRounds }));
  }

  private async replace(request: Promise<Room>): Promise<void> {
    this.upsert(await request);
  }

  private upsert(room: Room): void {
    this.list = this.byId(room.id)
      ? this.list.map((r) => (r.id === room.id ? room : r))
      : [...this.list, room];
  }
}

export const rooms = new RoomsStore();
