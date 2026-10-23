import { describe, expect, it } from 'vitest';
import { formatTokens, parseMarkdown, splitCommand } from './chatturns.util';

describe('parseMarkdown', () => {
  const links = 'https://example.test/a\nhttps://example.test/b\nhttps://example.test/c';

  it('escapes raw HTML so agent output mounts no element', () => {
    for (const src of [
      '<img src=x onerror=alert(1)>',
      '<script>alert(1)</script>',
      'a <b onclick=x>b</b>',
    ]) {
      const html = parseMarkdown(src);
      expect(html).not.toMatch(/<(img|script|b)\b/);
      expect(html).toContain('&lt;');
    }
  });

  it('renders links with a script scheme as plain text', () => {
    for (const src of [
      '[x](javascript:alert(1))',
      '[x](JaVaScRiPt:alert(1))',
      '[x](&#106;avascript:alert(1))',
      '[x](data:text/html,hi)',
      '<javascript:alert(1)>',
    ]) {
      expect(parseMarkdown(src)).not.toContain('<a');
    }
  });

  it('keeps http, mailto and relative links', () => {
    for (const href of [
      'https://example.test/a',
      'mailto:a@example.test',
      '#gallery',
      './a/b:c',
      '/x?y=1',
    ]) {
      expect(parseMarkdown(`[x](${href})`)).toContain('<a href=');
    }
  });

  it('keeps raw HTML inside code spans and fences as literal text', () => {
    expect(parseMarkdown('`<b>x</b>`')).toContain('<code>&lt;b&gt;x&lt;/b&gt;</code>');
    expect(parseMarkdown('```\n<img src=x>\n```')).toContain('&lt;img src=x&gt;');
  });

  it('collapses single newlines into one paragraph by default', () => {
    const html = parseMarkdown(links);
    expect(html).not.toContain('<br');
    expect(html.match(/<p>/g)).toHaveLength(1);
  });

  it('renders each line of a link list separately when breaks are on', () => {
    const html = parseMarkdown(links, true);
    expect(html).toContain('<br');
    expect(html.match(/<br/g)).toHaveLength(2);
    expect(html.match(/<p>/g)).toHaveLength(1);
  });

  it('renders a backticked context key as code, and the bare form as no link', () => {
    // The composer writes `<file:x>` for an attached @ reference.
    expect(parseMarkdown('see <file:index.md>')).not.toContain('<a ');
    const html = parseMarkdown('see `<file:index.md>`');
    expect(html).toContain('<code>');
    expect(html).not.toContain('<a ');
  });

  it('leaves a blank-line paragraph split alone either way', () => {
    for (const breaks of [false, true]) {
      expect(parseMarkdown('one\n\ntwo', breaks).match(/<p>/g)).toHaveLength(2);
    }
  });

  it('keeps fenced code intact either way', () => {
    const fenced = '```\nfirst\nsecond\n```';
    for (const breaks of [false, true]) {
      const html = parseMarkdown(fenced, breaks);
      expect(html).toContain('<code>');
      expect(html).not.toContain('<br');
      expect(html).toContain('first\nsecond');
    }
  });

  it('honours an explicit two-space hard break either way', () => {
    for (const breaks of [false, true]) {
      expect(parseMarkdown('one  \ntwo', breaks)).toContain('<br');
    }
  });
});

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
