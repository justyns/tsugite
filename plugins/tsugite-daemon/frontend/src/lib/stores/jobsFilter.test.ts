import { describe, expect, it } from 'vitest';
import {
  filterJobs,
  groupCounts,
  groupForState,
  jobMatchesFilter,
  jobTallyBySession,
  parseJobFilter,
} from './jobsFilter';

describe('parseJobFilter', () => {
  it('splits state:/agent:/# and free-text tokens', () => {
    expect(parseJobFilter('state:running agent:Odyn #job-abc deploy')).toEqual({
      states: ['running'],
      agents: ['odyn'],
      sessions: [],
      terms: ['job-abc', 'deploy'],
    });
  });

  it('is empty for blank input', () => {
    expect(parseJobFilter('   ')).toEqual({ states: [], agents: [], sessions: [], terms: [] });
  });
});

describe('jobMatchesFilter / filterJobs', () => {
  const jobs = [
    { job_id: 'job-1', state: 'running', agent: 'odyn', prompt: 'deploy the site' },
    { job_id: 'job-2', state: 'done', agent: 'scout', prompt: 'research pricing' },
    { job_id: 'job-3', state: 'running', agent: 'scout', prompt: 'deploy staging' },
  ];

  it('ANDs the state, agent, and term axes', () => {
    expect(filterJobs(jobs, 'state:running agent:scout').map((j) => j.job_id)).toEqual(['job-3']);
  });

  it('matches free-text against id/prompt/agent haystack', () => {
    expect(filterJobs(jobs, 'deploy').map((j) => j.job_id)).toEqual(['job-1', 'job-3']);
    expect(filterJobs(jobs, '#job-2').map((j) => j.job_id)).toEqual(['job-2']);
  });

  it('requires every term to match', () => {
    expect(jobMatchesFilter(jobs[0]!, parseJobFilter('deploy site'))).toBe(true);
    expect(jobMatchesFilter(jobs[0]!, parseJobFilter('deploy pricing'))).toBe(false);
  });
});

describe('session: axis', () => {
  const jobs = [
    { job_id: 'job-1', state: 'running', agent: 'odyn', parent_session_id: 'sess-a' },
    { job_id: 'job-2', state: 'done', agent: 'odyn', parent_session_id: 'sess-b' },
    { job_id: 'job-3', state: 'running', agent: 'scout', parent_session_id: 'sess-a' },
    { job_id: 'job-4', state: 'running', agent: 'scout', parent_session_id: null },
  ];

  it('parses session: into its own axis', () => {
    expect(parseJobFilter('session:sess-a')).toEqual({
      states: [],
      agents: [],
      sessions: ['sess-a'],
      terms: [],
    });
  });

  it('restricts to jobs spawned by that session', () => {
    expect(filterJobs(jobs, 'session:sess-a').map((j) => j.job_id)).toEqual(['job-1', 'job-3']);
  });

  it('ANDs with the other axes', () => {
    expect(filterJobs(jobs, 'session:sess-a agent:scout').map((j) => j.job_id)).toEqual(['job-3']);
  });

  it('excludes parentless jobs and matches session ids case-insensitively', () => {
    expect(filterJobs(jobs, 'session:SESS-B').map((j) => j.job_id)).toEqual(['job-2']);
    expect(filterJobs(jobs, 'session:sess-c')).toEqual([]);
  });
});

describe('groupForState / groupCounts', () => {
  it('maps states into the stuck/active/resolved board groups', () => {
    expect(groupForState('errored')).toBe('stuck');
    expect(groupForState('awaiting_input')).toBe('stuck');
    expect(groupForState('verifying')).toBe('active');
    expect(groupForState('cancelled')).toBe('resolved');
    expect(groupForState('spawning')).toBeNull();
  });

  it('reads a queued job as active - it is live work, not resolved', () => {
    expect(groupForState('queued')).toBe('active');
  });

  it('counts jobs per group', () => {
    expect(
      groupCounts([
        { state: 'queued' },
        { state: 'running' },
        { state: 'verifying' },
        { state: 'stuck' },
        { state: 'done' },
      ]),
    ).toEqual({ stuck: 1, active: 3, resolved: 1 });
  });
});

describe('jobTallyBySession', () => {
  it("tallies each session's outstanding jobs, and which of them are parked", () => {
    const tally = jobTallyBySession([
      { job_id: '1', state: 'running', parent_session_id: 'a' },
      { job_id: '2', state: 'queued', parent_session_id: 'a' },
      { job_id: '3', state: 'stuck', parent_session_id: 'a' },
      { job_id: '4', state: 'verifying', parent_session_id: 'b' },
    ]);
    expect(tally.get('a')).toEqual({ open: 3, parked: 1 });
    expect(tally.get('b')).toEqual({ open: 1, parked: 0 });
  });

  it('drops resolved jobs and jobs with no owning session', () => {
    const tally = jobTallyBySession([
      { job_id: '1', state: 'done', parent_session_id: 'a' },
      { job_id: '2', state: 'cancelled', parent_session_id: 'a' },
      { job_id: '3', state: 'running', parent_session_id: null },
    ]);
    expect(tally.size).toBe(0);
  });
});
