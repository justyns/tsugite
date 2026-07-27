import { describe, expect, test } from 'vitest';
import { resolveChatAgent, isJobArtifact } from './chatAgent';

describe('resolveChatAgent', () => {
  test("heals a wrong agent param to the session's true agent (jobs deep-link bug)", () => {
    // The jobs board opened the parent chat with agent=job_worker - the job's
    // worker agent file, which is NOT a chat adapter. The session actually runs
    // under `smoke`; trusting the param 404'd every agent-scoped call. The
    // session's own agent must win over the deep-link hint.
    expect(resolveChatAgent({ sessionAgent: 'smoke', paramAgent: 'job_worker' })).toBe('smoke');
  });

  test('falls back param -> fallback -> empty while the session agent is unresolved', () => {
    expect(resolveChatAgent({ sessionAgent: null, paramAgent: 'alpha' })).toBe('alpha');
    expect(resolveChatAgent({ sessionAgent: null, fallbackAgent: 'beta' })).toBe('beta');
    expect(resolveChatAgent({ sessionAgent: null })).toBe('');
  });

  test('the param is only a fallback - a resolved session agent always overrides it', () => {
    // A legitimate cross-agent deep link (param already correct) and a wrong one
    // both resolve to the session's true agent, so neither breaks.
    expect(resolveChatAgent({ sessionAgent: 'beta', paramAgent: 'beta' })).toBe('beta');
    expect(resolveChatAgent({ sessionAgent: 'gamma', paramAgent: 'beta' })).toBe('gamma');
  });
});

describe('isJobArtifact', () => {
  test('a worker/verifier session (metadata.job_id) is a job artifact', () => {
    expect(isJobArtifact({ job_id: 'job-a08493b1' })).toBe(true);
    expect(isJobArtifact({ job_id: 'job-x', verifier_for: 'session-abc' })).toBe(true);
  });

  test('the parent host session (job_host, no job_id) is not an artifact', () => {
    expect(isJobArtifact({ job_host: true })).toBe(false);
    expect(isJobArtifact({})).toBe(false);
    expect(isJobArtifact(null)).toBe(false);
    expect(isJobArtifact(undefined)).toBe(false);
  });
});
