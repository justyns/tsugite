import { describe, expect, it } from 'vitest';
import { sessionRow } from '../../views/chats/__fixtures__/sessionRow';
import { resolveSolo } from './solo';

const rows = [
  sessionRow('s1'),
  sessionRow('old', { superseded_by: 'new', alias: 'assistant' }),
  sessionRow('new', { alias: 'assistant' }),
];

describe('resolveSolo', () => {
  it('uses sessionId and ignores alias when both are present', () => {
    expect(resolveSolo({ sessionId: 's1', alias: 'assistant' }, rows, 'pending', true)).toEqual({
      state: 'ready',
      sessionId: 's1',
    });
  });

  it('resolves an alias to the row holding it', () => {
    expect(
      resolveSolo(
        { alias: 'assistant' },
        [sessionRow('s2', { alias: 'assistant' })],
        'pending',
        true,
      ),
    ).toEqual({
      state: 'ready',
      sessionId: 's2',
    });
  });

  it('follows the alias to the successor after compaction', () => {
    expect(resolveSolo({ alias: 'assistant' }, rows, 'pending', true)).toEqual({
      state: 'ready',
      sessionId: 'new',
    });
  });

  it('reports an alias as loading until the sessions list has loaded', () => {
    expect(resolveSolo({ alias: 'ghost' }, [], 'pending', false)).toEqual({ state: 'loading' });
  });

  it('waits on the fetch for a sessionId missing from the list', () => {
    expect(resolveSolo({ sessionId: 'fresh' }, rows, 'pending', true)).toEqual({
      state: 'loading',
      sessionId: 'fresh',
    });
  });

  it('resolves a fetched sessionId the list does not hold', () => {
    expect(resolveSolo({ sessionId: 'fresh' }, rows, 'found', true)).toEqual({
      state: 'ready',
      sessionId: 'fresh',
    });
  });

  it('reports a fetch failure other than 404 as an error', () => {
    expect(resolveSolo({ sessionId: 'x' }, rows, 'error', true)).toEqual({
      state: 'error',
      sessionId: 'x',
    });
  });

  it('reports no target when neither param is present', () => {
    expect(resolveSolo({}, rows, 'pending', true)).toEqual({ state: 'none' });
  });
});
