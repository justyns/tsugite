import { describe, expect, it } from 'vitest';
import { parseSSEFrame, reconcileHello, resumeQuery, splitFrames } from '$lib/api/sse';

describe('parseSSEFrame', () => {
  it('parses a single data line', () => {
    expect(parseSSEFrame('data: {"type":"ping"}')).toEqual({
      event: undefined,
      id: undefined,
      data: '{"type":"ping"}',
    });
  });

  it('strips exactly one leading space after the colon', () => {
    expect(parseSSEFrame('data:  two-spaces')?.data).toBe(' two-spaces');
    expect(parseSSEFrame('data:nospace')?.data).toBe('nospace');
  });

  it('joins multi-line data with newlines', () => {
    expect(parseSSEFrame('data: line1\ndata: line2')?.data).toBe('line1\nline2');
  });

  it('captures event and id fields', () => {
    const frame = parseSSEFrame('event: job_update\nid: 42\ndata: {}');
    expect(frame).toEqual({ event: 'job_update', id: '42', data: '{}' });
  });

  it('returns null for comment/keepalive frames', () => {
    expect(parseSSEFrame(': keepalive')).toBeNull();
  });

  it('returns null for empty frames', () => {
    expect(parseSSEFrame('')).toBeNull();
  });
});

describe('splitFrames', () => {
  it('splits complete frames and keeps a trailing partial as rest', () => {
    const { frames, rest } = splitFrames('data: a\n\ndata: b\n\ndata: par');
    expect(frames).toEqual(['data: a', 'data: b']);
    expect(rest).toBe('data: par');
  });

  it('yields no frames when there is no boundary yet', () => {
    const { frames, rest } = splitFrames('data: incomplete');
    expect(frames).toEqual([]);
    expect(rest).toBe('data: incomplete');
  });

  it('produces an empty rest when the buffer ends on a boundary', () => {
    const { frames, rest } = splitFrames('data: a\n\n');
    expect(frames).toEqual(['data: a']);
    expect(rest).toBe('');
  });
});

describe('resumeQuery', () => {
  it('is empty before the first hello (no epoch yet)', () => {
    expect(resumeQuery(null, 0)).toBe('');
  });

  it('carries epoch + last_seq once an epoch is known', () => {
    expect(resumeQuery('abc123', 7)).toBe('?epoch=abc123&last_seq=7');
  });

  it('url-encodes the epoch', () => {
    expect(resumeQuery('a b/c', 1)).toBe('?epoch=a%20b%2Fc&last_seq=1');
  });
});

describe('reconcileHello', () => {
  it('adopts the epoch and seq on the first hello without forcing a reload', () => {
    expect(reconcileHello({ epoch: null, lastSeq: 0 }, { epoch: 'e1', seq: 5 })).toEqual({
      epoch: 'e1',
      lastSeq: 5,
      reload: false,
    });
  });

  it('keeps the cursor and does not reload when the same daemon replays a gap', () => {
    expect(reconcileHello({ epoch: 'e1', lastSeq: 9 }, { epoch: 'e1', seq: 3 })).toEqual({
      epoch: 'e1',
      lastSeq: 9,
      reload: false,
    });
  });

  it('resets the cursor and reloads when the daemon epoch changed (restart)', () => {
    expect(reconcileHello({ epoch: 'e1', lastSeq: 9 }, { epoch: 'e2', seq: 0 })).toEqual({
      epoch: 'e2',
      lastSeq: 0,
      reload: true,
    });
  });

  it('resets the cursor and reloads when the server demands a resync', () => {
    expect(
      reconcileHello({ epoch: 'e1', lastSeq: 9 }, { epoch: 'e1', seq: 2, resync: true }),
    ).toEqual({
      epoch: 'e1',
      lastSeq: 2,
      reload: true,
    });
  });

  it('defaults a missing seq to 0 when resetting', () => {
    expect(reconcileHello({ epoch: null, lastSeq: 0 }, { epoch: 'e1' }).lastSeq).toBe(0);
  });
});
