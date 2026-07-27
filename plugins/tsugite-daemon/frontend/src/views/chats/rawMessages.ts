/**
 * Raw-messages debug fetch: GET /api/agents/{agent}/raw-messages?session_id=…
 * The daemon reconstructs, per turn, the request messages the model saw and its
 * raw response from the event log on demand - nothing durable is stored, so the
 * view is replay-safe and always matches what resume would rebuild.
 */
import { api } from '$lib/api/client';

export interface RawMessage {
  role: string;
  /** Reconstruction yields a string; typed loose so a future multimodal block
   *  array still renders instead of throwing. */
  content: unknown;
}

export interface RawTurn {
  /** Monotonic 1-based id for this model call: its stable identity, since `turn`
   *  is a per-run step counter that resets to 0 each user message. */
  index: number;
  turn: number | null;
  provider: string | null;
  model: string | null;
  /** The whole prompt the model saw this call (the conversation up to it). */
  request: RawMessage[];
  /** What this call added over the previous one, so the view can show the delta
   *  instead of re-rendering the full history every entry. */
  new_messages: RawMessage[];
  /** True when a compaction dropped the prior prefix before this call, so the
   *  prompt reset to a summary and `new_messages` is the whole request. */
  reset_before: boolean;
  response: { raw_content: string } | null;
}

export interface RawMessages {
  system_prompt: string | null;
  turns: RawTurn[];
}

export async function fetchRawMessages(
  agent: string,
  sessionId: string,
  userId?: string,
): Promise<RawMessages | null> {
  const params = new URLSearchParams({ session_id: sessionId });
  if (userId) params.set('user_id', userId);
  const res = await api.get<{ raw_messages: RawMessages | null }>(
    `/api/agents/${encodeURIComponent(agent)}/raw-messages?${params.toString()}`,
  );
  return res.raw_messages;
}
