/** Pure state → display metadata mappings for the rows components (kept out
 * of the .svelte files so they're directly unit-testable). */
import type { IconName } from '$lib/components/icon/icons';

// ---------- SessionRow ----------

export type SessionState = 'running' | 'thinking' | 'idle' | 'done' | 'failed';
export type SessionSourceType = 'ops' | 'code' | 'research' | 'chat' | 'room';

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
};

export const NEEDS_YOU_META: { label: string; color: string; icon: IconName } = {
  label: 'awaiting your input',
  color: 'var(--st-warn)',
  icon: 'q',
};

export function sessionStateMeta(state: SessionState): SessionStateMeta {
  return SESSION_STATE_META[state];
}

const SOURCE_TYPE_LABEL: Record<SessionSourceType, string> = {
  ops: 'ops',
  code: 'code',
  research: 'res',
  chat: 'chat',
  room: 'room',
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
  agent?: string;
  state: SessionState;
  needsYou?: boolean;
  isUnread?: boolean;
}): string {
  const parts = [opts.title];
  if (opts.agent) parts.push(opts.agent);
  parts.push(SESSION_STATE_META[opts.state].label);
  if (opts.needsYou) parts.push(NEEDS_YOU_META.label);
  if (opts.isUnread) parts.push('unread');
  return parts.join(', ');
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
