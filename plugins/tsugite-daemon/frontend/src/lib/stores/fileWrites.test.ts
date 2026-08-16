import { describe, expect, it } from 'vitest';
import { writeTargetsDoc, writtenPath } from './fileWrites';

describe('writtenPath', () => {
  it('reads the path off a file_write frame', () => {
    expect(
      writtenPath({ event_type: 'file_write', path: '/ws/smoke/ops/alpha.md', line_count: 2 }),
    ).toBe('/ws/smoke/ops/alpha.md');
  });

  it('is null for other frames and for a missing path', () => {
    expect(writtenPath({ event_type: 'file_read', path: 'a.md' })).toBeNull();
    expect(
      writtenPath({ event_type: 'tool_result_audit', tool: 'write_file', success: true }),
    ).toBeNull();
    expect(writtenPath({ event_type: 'file_write' })).toBeNull();
    expect(writtenPath({ event_type: 'file_write', path: '' })).toBeNull();
  });
});

describe('writeTargetsDoc', () => {
  it('matches a workspace-relative path the agent passed', () => {
    expect(writeTargetsDoc('ops/alpha.md', 'ops/alpha.md', '/ws/smoke')).toBe(true);
    expect(writeTargetsDoc('./ops/alpha.md', 'ops/alpha.md', '/ws/smoke')).toBe(true);
  });

  it('matches an absolute path against the workspace the tab is browsing', () => {
    expect(writeTargetsDoc('/ws/smoke/ops/alpha.md', 'ops/alpha.md', '/ws/smoke')).toBe(true);
    expect(writeTargetsDoc('/ws/smoke//ops/alpha.md', 'ops/alpha.md', '/ws/smoke/')).toBe(true);
    expect(writeTargetsDoc('/ws/other/ops/alpha.md', 'ops/alpha.md', '/ws/smoke')).toBe(false);
    expect(writeTargetsDoc('/ws/smoke/ops/alpha.md', 'ops/alpha.md', '')).toBe(false);
  });

  it('rejects another file, a prefix, and an unopened tab', () => {
    expect(writeTargetsDoc('ops/beta.md', 'ops/alpha.md', '/ws/smoke')).toBe(false);
    expect(writeTargetsDoc('ops/alpha.md.bak', 'ops/alpha.md', '/ws/smoke')).toBe(false);
    expect(writeTargetsDoc('ops/alpha.md', '', '/ws/smoke')).toBe(false);
  });
});
