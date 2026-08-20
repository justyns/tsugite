/**
 * Jobs filter grammar + state grouping, ported 1:1 from the Alpine jobs.js
 * (parseFilterText / filterJobsByText / GROUPS). Pure and node-testable; the
 * jobs store applies them over the list GET /api/jobs already returns.
 *
 * Grammar: whitespace-separated tokens.
 *   state:running   -> restrict to that job state (repeatable, OR within the axis)
 *   agent:odyn      -> restrict to that agent   (repeatable, OR within the axis)
 *   session:abc123  -> restrict to jobs spawned by that chat (parent_session_id)
 *   #job-abc123     -> free-text term (the '#' is stripped)
 *   anything else   -> free-text term
 * Axes combine with AND; every free-text term must substring-match the job's
 * id/prompt/agent/parent_session_id haystack.
 */

export interface JobLike {
  job_id?: string;
  state?: string;
  agent?: string;
  prompt?: string;
  verify_attempts?: number;
  max_attempts?: number;
  // Nullable to match the full Job payload (worker/parent ids are string|null).
  parent_session_id?: string | null;
}

export interface JobFilter {
  states: string[];
  agents: string[];
  sessions: string[];
  terms: string[];
}

/** The three board groupings the tab badges count; mirrors the backend's
 *  ?state= aliases in jobs.py. */
export const JOB_GROUPS = {
  stuck: ['stuck', 'errored', 'awaiting_input'],
  active: ['queued', 'running', 'verifying'],
  resolved: ['done', 'cancelled'],
} as const;

export type JobGroup = keyof typeof JOB_GROUPS;

export function parseJobFilter(text: string): JobFilter {
  const states: string[] = [];
  const agents: string[] = [];
  const sessions: string[] = [];
  const terms: string[] = [];
  for (const token of text.trim().split(/\s+/)) {
    if (!token) continue;
    if (token.startsWith('state:')) states.push(token.slice(6).toLowerCase());
    else if (token.startsWith('agent:')) agents.push(token.slice(6).toLowerCase());
    else if (token.startsWith('session:')) sessions.push(token.slice(8).toLowerCase());
    else if (token.startsWith('#')) terms.push(token.slice(1).toLowerCase());
    else terms.push(token.toLowerCase());
  }
  return { states, agents, sessions, terms };
}

function haystack(job: JobLike): string {
  return [job.job_id, job.prompt, job.agent, job.parent_session_id]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}

export function jobMatchesFilter(job: JobLike, filter: JobFilter): boolean {
  if (filter.states.length && !filter.states.includes((job.state ?? '').toLowerCase()))
    return false;
  if (filter.agents.length && !filter.agents.includes((job.agent ?? '').toLowerCase()))
    return false;
  if (
    filter.sessions.length &&
    !filter.sessions.includes((job.parent_session_id ?? '').toLowerCase())
  )
    return false;
  if (filter.terms.length) {
    const hay = haystack(job);
    if (!filter.terms.every((t) => hay.includes(t))) return false;
  }
  return true;
}

export function filterJobs<T extends JobLike>(jobs: T[], text: string): T[] {
  const filter = parseJobFilter(text);
  return jobs.filter((j) => jobMatchesFilter(j, filter));
}

/** Which board group a job state belongs to (or null for none). */
export function groupForState(state: string | undefined): JobGroup | null {
  if (!state) return null;
  for (const group of Object.keys(JOB_GROUPS) as JobGroup[]) {
    if ((JOB_GROUPS[group] as readonly string[]).includes(state)) return group;
  }
  return null;
}

export interface SessionJobTally {
  open: number;
  parked: number;
}

export function jobTallyBySession(jobs: JobLike[]): Map<string, SessionJobTally> {
  const tallies = new Map<string, SessionJobTally>();
  for (const job of jobs) {
    const id = job.parent_session_id;
    const group = groupForState(job.state);
    if (!id || group === 'resolved') continue;
    const tally = tallies.get(id) ?? { open: 0, parked: 0 };
    tally.open += 1;
    if (group === 'stuck') tally.parked += 1;
    tallies.set(id, tally);
  }
  return tallies;
}

export function groupCounts(jobs: JobLike[]): Record<JobGroup, number> {
  const counts: Record<JobGroup, number> = { stuck: 0, active: 0, resolved: 0 };
  for (const job of jobs) {
    const group = groupForState(job.state);
    if (group) counts[group] += 1;
  }
  return counts;
}
