import { describe, expect, it } from 'vitest';
import { chatsNavBadge, jobsNavBadges, needsYouTotal } from './navBadges';
import { sessionRow as row } from '../../views/chats/__fixtures__/sessionRow';

describe('jobsNavBadges', () => {
  it('shows nothing while no job is outstanding', () => {
    expect(jobsNavBadges({ active: 0, stuck: 0, resolved: 7 })).toEqual([]);
  });

  it('reads running work as ambient and parked work as action-required', () => {
    expect(jobsNavBadges({ active: 2, stuck: 3, resolved: 0 })).toEqual([
      { count: 2, variant: 'info', label: '2 jobs running' },
      { count: 3, variant: 'action', label: '3 jobs need you' },
    ]);
  });

  it('names a single job in the singular', () => {
    expect(jobsNavBadges({ active: 1, stuck: 1, resolved: 0 })).toEqual([
      { count: 1, variant: 'info', label: '1 job running' },
      { count: 1, variant: 'action', label: '1 job needs you' },
    ]);
  });
});

describe('chatsNavBadge', () => {
  it('shows nothing while no chat is waiting on you', () => {
    expect(chatsNavBadge(0)).toEqual([]);
  });

  it('reads a waiting chat as action-required, named in the singular', () => {
    expect(chatsNavBadge(1)).toEqual([{ count: 1, variant: 'action', label: '1 chat needs you' }]);
  });

  it('names several waiting chats in the plural', () => {
    expect(chatsNavBadge(4)).toEqual([{ count: 4, variant: 'action', label: '4 chats need you' }]);
  });
});

describe('needsYouTotal', () => {
  it('adds the parked jobs to the chats already waiting on you', () => {
    const jobs = [{ job_id: 'j1', state: 'awaiting_input', parent_session_id: 's2' }];
    expect(needsYouTotal([row('s1')], jobs)).toBe(2);
  });

  it('counts a parked job in a chat that already needs you once', () => {
    const jobs = [{ job_id: 'j1', state: 'errored', parent_session_id: 's1' }];
    expect(needsYouTotal([row('s1')], jobs)).toBe(1);
  });

  it('counts a parked job whose chat is gone', () => {
    const jobs = [{ job_id: 'j1', state: 'stuck', parent_session_id: null }];
    expect(needsYouTotal([], jobs)).toBe(1);
  });

  it('leaves running and resolved jobs out', () => {
    const jobs = [
      { job_id: 'j1', state: 'running', parent_session_id: 's1' },
      { job_id: 'j2', state: 'done', parent_session_id: 's1' },
    ];
    expect(needsYouTotal([], jobs)).toBe(0);
  });
});
