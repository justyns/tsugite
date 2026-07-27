import { describe, it, expect } from 'vitest';
import { splitFrontmatter, parseFrontmatterEntries, renderFrontmatterHtml } from './frontmatter';

describe('splitFrontmatter', () => {
  it('separates a fenced frontmatter block from the body', () => {
    const src = '---\ntitle: Backup Retention\n---\n# Backup Retention\n\nbody text\n';
    const r = splitFrontmatter(src);
    expect(r.hasFrontmatter).toBe(true);
    expect(r.fm).toBe('title: Backup Retention');
    expect(r.body).toBe('# Backup Retention\n\nbody text\n');
  });

  it('reports no frontmatter when the document does not open with a fence', () => {
    const src = '# Just a document\n\nno frontmatter here';
    const r = splitFrontmatter(src);
    expect(r.hasFrontmatter).toBe(false);
    expect(r.fm).toBe('');
    expect(r.body).toBe(src);
  });
});

describe('parseFrontmatterEntries', () => {
  it('parses simple scalar entries', () => {
    expect(
      parseFrontmatterEntries('title: Backup Retention\nowner: platform-team\nreviewed: true'),
    ).toEqual([
      { key: 'title', value: 'Backup Retention', nested: false },
      { key: 'owner', value: 'platform-team', nested: false },
      { key: 'reviewed', value: 'true', nested: false },
    ]);
  });

  it('captures a block sequence as raw dedented text, marked nested', () => {
    const entries = parseFrontmatterEntries(
      'links:\n  - runbook\n  - sse-reconnect\nowner: platform-team',
    );
    expect(entries).toEqual([
      { key: 'links', value: '- runbook\n- sse-reconnect', nested: true },
      { key: 'owner', value: 'platform-team', nested: false },
    ]);
  });

  it('captures a block scalar body as raw dedented text, marked nested', () => {
    const entries = parseFrontmatterEntries('instructions: |\n  line one\n  line two\nmodel: x');
    expect(entries).toEqual([
      { key: 'instructions', value: 'line one\nline two', nested: true },
      { key: 'model', value: 'x', nested: false },
    ]);
  });

  it('captures a nested map as raw dedented text', () => {
    const entries = parseFrontmatterEntries('sandbox:\n  enabled: true\n  no_network: true');
    expect(entries).toEqual([
      { key: 'sandbox', value: 'enabled: true\nno_network: true', nested: true },
    ]);
  });

  it('skips blank lines and comment lines between entries', () => {
    const entries = parseFrontmatterEntries('# a comment\nname: x\n\n# another\nmodel: y');
    expect(entries).toEqual([
      { key: 'name', value: 'x', nested: false },
      { key: 'model', value: 'y', nested: false },
    ]);
  });

  it('trims trailing blank lines out of a captured block', () => {
    const entries = parseFrontmatterEntries('tools:\n  - a\n  - b\n\n\nowner: x');
    expect(entries).toEqual([
      { key: 'tools', value: '- a\n- b', nested: true },
      { key: 'owner', value: 'x', nested: false },
    ]);
  });

  it('returns an empty array for an empty block', () => {
    expect(parseFrontmatterEntries('')).toEqual([]);
    expect(parseFrontmatterEntries('   \n  ')).toEqual([]);
  });

  it('bails out to an empty array when a top-level line is not key-shaped (malformed)', () => {
    expect(parseFrontmatterEntries('just some\nprose text\nnot frontmatter at all')).toEqual([]);
  });

  it('bails out entirely rather than dropping surrounding prose around one stray colon', () => {
    // A single line that happens to look key-shaped should not cause the rest of a
    // non-frontmatter block to be silently discarded.
    const entries = parseFrontmatterEntries('Some intro text\nNote: see below\nmore prose here');
    expect(entries).toEqual([]);
  });
});

describe('renderFrontmatterHtml', () => {
  it('renders scalar entries as table rows with escaped text', () => {
    const html = renderFrontmatterHtml([{ key: 'owner', value: '<b>x</b>', nested: false }]);
    expect(html).toContain('<blockquote');
    expect(html).toContain('<table>');
    expect(html).toContain('<th>owner</th>');
    expect(html).toContain('<td>&lt;b&gt;x&lt;/b&gt;</td>');
    expect(html).not.toContain('<b>x</b>');
  });

  it('renders a nested entry value inside a code block', () => {
    const html = renderFrontmatterHtml([
      { key: 'tools', value: '- read_file\n- run', nested: true },
    ]);
    expect(html).toContain('<th>tools</th>');
    expect(html).toContain('<td><pre><code>- read_file\n- run</code></pre></td>');
  });

  it('returns an empty string for no entries', () => {
    expect(renderFrontmatterHtml([])).toBe('');
  });
});
