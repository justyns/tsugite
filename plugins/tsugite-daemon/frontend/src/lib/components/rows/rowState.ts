/** Pure state → display metadata mappings for the rows components (kept out
 * of the .svelte files so they're directly unit-testable). */
import type { IconName } from '$lib/components/icon/icons';

// ---------- SessionRow ----------

export type SessionState = 'running' | 'thinking' | 'idle' | 'done' | 'failed' | 'needs-you';
export type SessionSourceType = 'ops' | 'code' | 'research' | 'chat';

export type SessionStateMeta = {
  /** Word shown in the composed aria-label and used as the state-language text. */
  label: string;
  color: string;
  /** Ambient-activity glyph (braille spinner) instead of a static icon. */
  spin: boolean;
  icon?: IconName;
};

const SESSION_STATE_META: Record<SessionState, SessionStateMeta> = {
  running: { label: 'running', color: 'var(--st-ok)', spin: true },
  thinking: { label: 'thinking', color: 'var(--st-info)', spin: true },
  idle: { label: 'idle', color: 'var(--tx3)', spin: false, icon: 'ring' },
  done: { label: 'done', color: 'var(--st-mute)', spin: false, icon: 'check' },
  failed: { label: 'failed', color: 'var(--st-err)', spin: false, icon: 'x' },
  'needs-you': { label: 'awaiting your input', color: 'var(--st-warn)', spin: false, icon: 'q' },
};

export function sessionStateMeta(state: SessionState): SessionStateMeta {
  return SESSION_STATE_META[state];
}

const SOURCE_TYPE_LABEL: Record<SessionSourceType, string> = {
  ops: 'ops',
  code: 'code',
  research: 'res',
  chat: 'chat',
};

/** Session-row label abbreviations (research -> "res"). */
export function sourceTypeLabel(type: SessionSourceType): string {
  return SOURCE_TYPE_LABEL[type];
}

/** A row's status is otherwise conveyed by icon + color alone (the `.ind` glyph
 * has no visible text), so this composes the state word into an aria-label,
 * per the state-language rule that nothing may rely on color alone. */
export function buildSessionRowAriaLabel(opts: {
  title: string;
  state: SessionState;
  isUnread?: boolean;
}): string {
  const parts = [opts.title, SESSION_STATE_META[opts.state].label];
  if (opts.isUnread) parts.push('unread');
  return parts.join(', ');
}

// ---------- SpacesRow ----------

export type SpaceState = 'working' | 'blocked' | 'idle' | 'done';

export type SpaceStateMeta = {
  label: string;
  spin: boolean;
  icon?: IconName;
};

const SPACE_STATE_META: Record<SpaceState, SpaceStateMeta> = {
  working: { label: 'working', spin: true },
  blocked: { label: 'blocked', spin: false, icon: 'q' },
  idle: { label: 'idle', spin: false, icon: 'ring' },
  done: { label: 'done', spin: false, icon: 'check' },
};

export function spaceStateMeta(state: SpaceState): SpaceStateMeta {
  return SPACE_STATE_META[state];
}

/** Defensive clamp so a bad/out-of-range backend value can't blow out the
 * meter bar width or render `NaN%`. */
export function clampPct(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.min(100, Math.max(0, value));
}

// ---------- CheckItem ----------

export type CheckState = 'pending' | 'active' | 'pass' | 'fail';

const CHECK_STATE_PREFIX: Record<CheckState, string> = {
  pending: 'Pending',
  active: 'Verifying',
  pass: 'Passed',
  fail: 'Failed',
};

/** Screen-reader-only prefix - the box icon (or blank box) is the only other
 * signal of state, so this keeps state readable without relying on color. */
export function checkStatePrefix(state: CheckState): string {
  return CHECK_STATE_PREFIX[state];
}
