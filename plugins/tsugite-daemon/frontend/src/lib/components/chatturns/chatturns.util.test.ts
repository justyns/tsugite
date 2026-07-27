import { describe, expect, it } from 'vitest';
import { formatTokens, splitCommand } from './chatturns.util';

describe('splitCommand', () => {
  it('splits the program from its arguments', () => {
    expect(splitCommand('rg -n "EventSource" src/')).toEqual({
      program: 'rg',
      rest: ' -n "EventSource" src/',
    });
  });

  it('handles a bare program with no arguments', () => {
    expect(splitCommand('ls')).toEqual({ program: 'ls', rest: '' });
  });

  it('ignores leading whitespace', () => {
    expect(splitCommand('  npm test')).toEqual({ program: 'npm', rest: ' test' });
  });

  it('keeps interior whitespace in the argument tail', () => {
    expect(splitCommand('npm test -w @tsugite/sse --watch')).toEqual({
      program: 'npm',
      rest: ' test -w @tsugite/sse --watch',
    });
  });

  it('is empty-safe', () => {
    expect(splitCommand('')).toEqual({ program: '', rest: '' });
  });
});

describe('formatTokens', () => {
  it('shows raw counts below 1000', () => {
    expect(formatTokens(0)).toBe('0');
    expect(formatTokens(6)).toBe('6');
    expect(formatTokens(999)).toBe('999');
  });

  it('abbreviates thousands with one decimal', () => {
    expect(formatTokens(1200)).toBe('1.2k');
    expect(formatTokens(9200)).toBe('9.2k');
  });

  it('drops a trailing .0', () => {
    expect(formatTokens(2000)).toBe('2k');
  });

  it('drops the decimal past 100k', () => {
    expect(formatTokens(128000)).toBe('128k');
  });
});
