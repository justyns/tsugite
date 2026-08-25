import { describe, expect, test } from 'vitest';
import { isJobArtifact } from './jobArtifact';

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
