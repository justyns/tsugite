import { describe, expect, it } from 'vitest';
import { jobPillMeta, jobPillState } from './jobPill';

describe('jobPillState', () => {
  it('shortens awaiting_input to its short label', () => {
    expect(jobPillState('awaiting_input')).toBe('awaiting');
    expect(jobPillState('running')).toBe('running');
  });
});

describe('jobPillMeta', () => {
  it('gives running a spinner and a state word', () => {
    const m = jobPillMeta('running');
    expect(m.spin).toBe(true);
    expect(m.label).toBe('running');
  });
  it('maps every backend state to an icon + non-empty label', () => {
    for (const s of [
      'queued',
      'running',
      'verifying',
      'awaiting_input',
      'stuck',
      'errored',
      'done',
      'cancelled',
    ]) {
      const m = jobPillMeta(s);
      expect(m.icon).toBeTruthy();
      expect(m.label.length).toBeGreaterThan(0);
    }
  });
  it('never returns the spinner for a resolved state', () => {
    expect(jobPillMeta('done').spin).toBe(false);
    expect(jobPillMeta('cancelled').spin).toBe(false);
  });
});
