import { describe, expect, it } from 'vitest';
import { parseRefToken } from './refToken';

describe('parseRefToken', () => {
  it('returns null for an empty string', () => {
    expect(parseRefToken('', 0)).toBeNull();
  });

  it('returns null when the caret is at position 0', () => {
    expect(parseRefToken('@file', 0)).toBeNull();
  });

  it('opens an empty @ token the moment the trigger is typed at the start', () => {
    expect(parseRefToken('@', 1)).toEqual({ trigger: '@', query: '', start: 0, end: 1 });
  });

  it('opens an empty @ token after whitespace', () => {
    expect(parseRefToken('hello @', 7)).toEqual({ trigger: '@', query: '', start: 6, end: 7 });
  });

  it('captures the query typed after an @ at the start of input', () => {
    expect(parseRefToken('@abc', 4)).toEqual({ trigger: '@', query: 'abc', start: 0, end: 4 });
  });

  it('captures the query typed after an @ mid-string', () => {
    expect(parseRefToken('hello @wor', 10)).toEqual({
      trigger: '@',
      query: 'wor',
      start: 6,
      end: 10,
    });
  });

  it('supports the # trigger', () => {
    expect(parseRefToken('#tag', 4)).toEqual({ trigger: '#', query: 'tag', start: 0, end: 4 });
  });

  it('keeps file-path characters (slash, dot, dash) in the query', () => {
    expect(parseRefToken('@kb/ops/sse-reconnect.md', 24)).toEqual({
      trigger: '@',
      query: 'kb/ops/sse-reconnect.md',
      start: 0,
      end: 24,
    });
  });

  it('does not trigger on an @ that is not at a word boundary (email)', () => {
    expect(parseRefToken('user@host', 9)).toBeNull();
  });

  it('does not trigger on a # that is not at a word boundary (hex color)', () => {
    expect(parseRefToken('color a#b', 9)).toBeNull();
  });

  it('closes once whitespace is typed after the query', () => {
    expect(parseRefToken('@a b', 4)).toBeNull();
  });

  it('reads the token the caret sits inside, ignoring text to the right', () => {
    expect(parseRefToken('see @foo bar', 8)).toEqual({
      trigger: '@',
      query: 'foo',
      start: 4,
      end: 8,
    });
  });

  it('re-opens for a caret still inside a partially typed token', () => {
    expect(parseRefToken('@ab', 2)).toEqual({ trigger: '@', query: 'a', start: 0, end: 2 });
  });

  it('scopes the token to a single line (newline is a boundary)', () => {
    expect(parseRefToken('@foo\n@bar', 9)).toEqual({
      trigger: '@',
      query: 'bar',
      start: 5,
      end: 9,
    });
  });

  it('returns null when a second trigger sits inside the run', () => {
    expect(parseRefToken('@a#b', 4)).toBeNull();
  });

  it('clamps a caret past the end of the string', () => {
    expect(parseRefToken('@abc', 99)).toEqual({ trigger: '@', query: 'abc', start: 0, end: 4 });
  });

  it('clamps a negative caret to null', () => {
    expect(parseRefToken('@abc', -3)).toBeNull();
  });

  describe('prefix-scoped queries', () => {
    it('keeps the space when the token opens with a known prefix', () => {
      expect(parseRefToken('@jira auth', 10, ['jira'])).toEqual({
        trigger: '@',
        query: 'jira auth',
        start: 0,
        end: 10,
      });
    });

    it('opens the scoped token the moment the prefix gets its space', () => {
      expect(parseRefToken('@jira ', 6, ['jira'])).toEqual({
        trigger: '@',
        query: 'jira ',
        start: 0,
        end: 6,
      });
    });

    it('keeps multiple spaces inside the subquery', () => {
      expect(parseRefToken('see @jira auth flow', 19, ['jira'])).toEqual({
        trigger: '@',
        query: 'jira auth flow',
        start: 4,
        end: 19,
      });
    });

    it('still closes on a space when the first word is not a known prefix', () => {
      expect(parseRefToken('@a b', 4, ['jira'])).toBeNull();
    });

    it('does not keep spaces without a prefix list (plain token behavior)', () => {
      expect(parseRefToken('@jira auth', 10)).toBeNull();
    });
  });
});
