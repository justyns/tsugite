/**
 * Job-state pill metadata. The shared `Pill` component only models session
 * states (idle/busy/streaming/...); the job state-language is a distinct set,
 * so `JobPill.svelte` renders it locally off this map. Each state has its own
 * `data-st` token and colour.
 *
 * `awaiting_input` maps to the shorter `awaiting` token.
 */
import type { IconName } from '$lib/components/icon/icons';

export type JobPillState =
  'queued' | 'running' | 'verifying' | 'awaiting' | 'stuck' | 'errored' | 'done' | 'cancelled';

export function jobPillState(state: string): JobPillState {
  return state === 'awaiting_input' ? 'awaiting' : (state as JobPillState);
}

export interface JobPillMeta {
  label: string;
  icon: IconName;
  /** Render the braille spinner instead of a static icon (live work). */
  spin: boolean;
}

const META: Record<JobPillState, JobPillMeta> = {
  queued: { label: 'queued', icon: 'clock', spin: false },
  running: { label: 'running', icon: 'play', spin: true },
  verifying: { label: 'verifying', icon: 'search', spin: false },
  awaiting: { label: 'awaiting input', icon: 'q', spin: false },
  stuck: { label: 'stuck', icon: 'alert', spin: false },
  errored: { label: 'errored', icon: 'x', spin: false },
  done: { label: 'done', icon: 'check', spin: false },
  cancelled: { label: 'cancelled', icon: 'cancel', spin: false },
};

export function jobPillMeta(state: string): JobPillMeta {
  return META[jobPillState(state)] ?? META.queued;
}
