/**
 * Pure mappings from a backend session row (stores/sessions.svelte) to the
 * SessionRow component's display vocabulary, plus the rail's pinned/active/recent
 * grouping and the relative-time stamp. Kept framework-free so it runs in the
 * fast unit vitest project and the rail stays a thin render of these.
 */
import type { SessionRow } from '$lib/stores/sessions.svelte';
import type { SessionSourceType, SessionState } from '$lib/components/rows/rowState';

/** metadata.type carries the semantic category badge (code/ops/res/chat). The
 *  backend `source` (web/interactive/schedule/...) is the origin, a different
 *  axis; the row badge is the category, defaulting to chat. */
export function sessionSourceType(row: SessionRow): SessionSourceType {
  const type = String(row.metadata?.type ?? '').toLowerCase();
  if (type === 'code' || type === 'ops' || type === 'research') return type;
  return 'chat';
}

const DONE_STATUSES = new Set(['completed', 'cancelled']);

/** The daemon's FINISHED_STATUSES (session_store.py): a session that has ended,
 *  by any outcome. Kept in hand-sync with the daemon, which owns the list. */
const FINISHED_STATUSES = new Set(['completed', 'failed', 'cancelled']);

/** Whether the session has ended (done, failed, or cancelled) and so leaves the
 *  rail's live recency flow for its own bucket. */
export function isFinishedSession(row: SessionRow): boolean {
  return FINISHED_STATUSES.has(row.status);
}

/** A busy session is "thinking" while the daemon reports it waiting on the LLM,
 *  else "running" (a tool/turn is active). Read from the progress status_text
 *  the daemon already computes (session_store._progress_status_text). */
function busyState(row: SessionRow): SessionState {
  const text = String(row.progress?.status_text ?? '').toLowerCase();
  return text.includes('waiting on llm') || text.includes('thinking') ? 'thinking' : 'running';
}

export interface RowStateHints {
  /** A blocking ask_user is outstanding for this session (tracked from the
   *  session_event broadcast; never expressible by status alone). */
  pendingAsk?: boolean;
}

export function sessionRowState(row: SessionRow, hints: RowStateHints = {}): SessionState {
  if (row.status === 'failed') return 'failed';
  if (DONE_STATUSES.has(row.status)) return 'done';
  if (hints.pendingAsk) return 'needs-you';
  if (row.busy) return busyState(row);
  return 'idle';
}

export interface SessionGroups {
  pinned: SessionRow[];
  active: SessionRow[];
  recent: SessionRow[];
  ended: SessionRow[];
}

export interface GroupHints {
  /** Session ids with an outstanding ask_user (they sort into active). */
  attn: Set<string>;
}

/** Four rail buckets: pinned (user-anchored), active (live or needs-you), recent
 *  (everything else still live), and ended (finished, pulled out of the recency
 *  flow). A row appears in exactly one bucket. Pinned wins over ended, so a pinned
 *  finished session stays pinned; a finished row never lands in active even with a
 *  pending ask. Order within a bucket is preserved from the store's sorted input. */
export function groupSessions(rows: SessionRow[], hints: GroupHints): SessionGroups {
  const pinned: SessionRow[] = [];
  const active: SessionRow[] = [];
  const recent: SessionRow[] = [];
  const ended: SessionRow[] = [];
  for (const row of rows) {
    if (row.pinned) pinned.push(row);
    else if (isFinishedSession(row)) ended.push(row);
    else if (row.busy || hints.attn.has(row.id)) active.push(row);
    else recent.push(row);
  }
  return { pinned, active, recent, ended };
}

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

/** Compact relative stamp for the row's `when` slot: now / Nm / Nh / "jul 12". */
export function formatWhen(iso: string | null, now: number = Date.now()): string {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const secs = Math.max(0, Math.round((now - t) / 1000));
  if (secs < 45) return 'now';
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h`;
  const d = new Date(t);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}`;
}

/** The row's sub-line description: the topic chip text, falling back to the
 *  user/source label so the row is never bare. */
export function sessionTopic(row: SessionRow): string {
  const topic = row.metadata?.topic;
  if (typeof topic === 'string' && topic.trim()) return topic;
  return row.label ?? '';
}
