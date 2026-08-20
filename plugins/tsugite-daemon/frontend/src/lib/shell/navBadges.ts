import { groupForState, type JobGroup, type JobLike } from '$lib/stores/jobsFilter';
import type { SessionRow } from '$lib/stores/sessions.svelte';

export interface NavBadge {
  count: number;
  /** `info` is ambient (work is happening), `action` demands a person. */
  variant: 'info' | 'action';
  label: string;
}

function needsYouBadge(count: number, noun: string): NavBadge {
  return {
    count,
    variant: 'action',
    label: count === 1 ? `1 ${noun} needs you` : `${count} ${noun}s need you`,
  };
}

export function jobsNavBadges(counts: Record<JobGroup, number>): NavBadge[] {
  const badges: NavBadge[] = [];
  if (counts.active > 0) {
    badges.push({
      count: counts.active,
      variant: 'info',
      label: `${counts.active} job${counts.active === 1 ? '' : 's'} running`,
    });
  }
  if (counts.stuck > 0) badges.push(needsYouBadge(counts.stuck, 'job'));
  return badges;
}

export function chatsNavBadge(count: number): NavBadge[] {
  return count > 0 ? [needsYouBadge(count, 'chat')] : [];
}

export function needsYouTotal(chats: SessionRow[], jobs: JobLike[]): number {
  const ids = new Set(chats.map((r) => r.id));
  const elsewhere = jobs.filter(
    (j) => groupForState(j.state) === 'stuck' && !ids.has(j.parent_session_id ?? ''),
  );
  return ids.size + elsewhere.length;
}
