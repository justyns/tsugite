import { describe, it, expect } from 'vitest';
import type { WorkspaceEntry } from '$lib/stores/files.svelte';
import {
  parseWikiLinks,
  parseTags,
  parseHeadings,
  resolveTarget,
  buildIndex,
  relatedNotes,
  stripTagsLine,
  formatSize,
  breadcrumbs,
  buildTree,
  renderMarkdown,
} from './wiki';

describe('parseWikiLinks', () => {
  it('extracts bare and aliased links', () => {
    const refs = parseWikiLinks('see [[backup-retention]] and [[ops/runbook|the runbook]].');
    expect(refs).toEqual([
      { target: 'backup-retention', alias: 'backup-retention' },
      { target: 'ops/runbook', alias: 'the runbook' },
    ]);
  });

  it('returns nothing for single brackets or empty input', () => {
    expect(parseWikiLinks('a [single] bracket, and [x](y) a link')).toEqual([]);
    expect(parseWikiLinks('')).toEqual([]);
  });

  it('ignores links inside code spans and fences', () => {
    const md = 'inline `[[not-a-link]]` and\n```\n[[also-not]]\n```\nbut [[real]] counts.';
    expect(parseWikiLinks(md)).toEqual([{ target: 'real', alias: 'real' }]);
  });
});

describe('parseTags', () => {
  it('reads inline hashtags in order, de-duplicated and lowercased', () => {
    expect(parseTags('tags: #Ops #sse #ops\n\nbody #Daemon')).toEqual(['ops', 'sse', 'daemon']);
  });

  it('reads a yaml-style tags list', () => {
    expect(parseTags('tags: [ops, backup]')).toEqual(['ops', 'backup']);
  });

  it('does not treat markdown headings as tags', () => {
    expect(parseTags('# Heading\n\n## Sub heading')).toEqual([]);
  });

  it('ignores hashes inside code', () => {
    expect(parseTags('color `#ffffff` here, tag #real')).toEqual(['real']);
  });
});

describe('parseHeadings', () => {
  it('captures depth and text, skipping fenced content', () => {
    const md =
      '# Title\n\ntext\n\n## Backoff table\n\n```\n## not a heading\n```\n\n## Verification';
    expect(parseHeadings(md)).toEqual([
      { depth: 1, text: 'Title' },
      { depth: 2, text: 'Backoff table' },
      { depth: 2, text: 'Verification' },
    ]);
  });
});

describe('resolveTarget', () => {
  const files = ['index.md', 'ops/sse-reconnect.md', 'ops/backup-retention.md', 'concepts/5s.md'];

  it('matches by basename regardless of case or .md suffix', () => {
    expect(resolveTarget('backup-retention', files)).toBe('ops/backup-retention.md');
    expect(resolveTarget('SSE-Reconnect.md', files)).toBe('ops/sse-reconnect.md');
    expect(resolveTarget('5s', files)).toBe('concepts/5s.md');
  });

  it('matches a full relative path', () => {
    expect(resolveTarget('ops/sse-reconnect', files)).toBe('ops/sse-reconnect.md');
  });

  it('returns null for an unknown page', () => {
    expect(resolveTarget('household-systems', files)).toBeNull();
  });
});

describe('buildIndex', () => {
  const docs = [
    {
      path: 'ops/sse-reconnect.md',
      content: 'tags: #ops #sse\n\nsee [[backup-retention]] and [[household-systems]].',
    },
    {
      path: 'ops/backup-retention.md',
      content: 'tags: #ops #backup\n\nresolved in [[sse-reconnect]] review.',
    },
    { path: 'ops/runbook.md', content: 'tags: #ops\n\nsee [[sse-reconnect]] for backoff.' },
  ];

  it('inverts wikilinks into backlinks with a snippet, skipping unresolved and self links', () => {
    const idx = buildIndex(docs);
    const back = idx.backlinks.get('ops/sse-reconnect.md') ?? [];
    expect(back.map((b) => b.file).sort()).toEqual(['ops/backup-retention.md', 'ops/runbook.md']);
    expect(back[0]!.snippet).toContain('sse-reconnect');
    // household-systems is unresolved, so no backlink bucket is created for it
    expect(idx.backlinks.has('household-systems')).toBe(false);
  });

  it('counts tags across the corpus and per file', () => {
    const idx = buildIndex(docs);
    expect(idx.tagCounts.get('ops')).toBe(3);
    expect(idx.tagCounts.get('sse')).toBe(1);
    expect(idx.tagsByFile.get('ops/backup-retention.md')).toEqual(['ops', 'backup']);
  });

  it('exposes a resolver bound to the corpus file list', () => {
    const idx = buildIndex(docs);
    expect(idx.resolve('backup-retention')).toBe('ops/backup-retention.md');
    expect(idx.resolve('nope')).toBeNull();
  });
});

describe('relatedNotes', () => {
  it('returns notes sharing at least the threshold number of tags, most-shared first', () => {
    const tagsByFile = new Map<string, string[]>([
      ['a.md', ['ops', 'sse', 'daemon']],
      ['b.md', ['ops', 'sse']],
      ['c.md', ['ops']],
      ['d.md', ['home']],
    ]);
    expect(relatedNotes('a.md', tagsByFile, 2)).toEqual([{ path: 'b.md', shared: 2 }]);
  });
});

describe('stripTagsLine', () => {
  it('removes the first tags: line only', () => {
    expect(stripTagsLine('# T\n\ntags: #ops #sse\n\nbody')).toBe('# T\n\nbody');
  });
});

describe('formatSize', () => {
  it('scales bytes to a human unit', () => {
    expect(formatSize(512)).toBe('512 B');
    expect(formatSize(2048)).toBe('2.0 KB');
    expect(formatSize(5 * 1024 * 1024)).toBe('5.0 MB');
  });
});

describe('breadcrumbs', () => {
  it('splits a path into non-empty segments', () => {
    expect(breadcrumbs('ops/sse-reconnect.md')).toEqual(['ops', 'sse-reconnect.md']);
    expect(breadcrumbs('')).toEqual([]);
  });
});

describe('buildTree', () => {
  const entries: WorkspaceEntry[] = [
    { path: 'ops', name: 'ops', is_dir: true },
    { path: 'index.md', name: 'index.md', is_dir: false, size: 10, modified: 't' },
    { path: 'ops/runbook.md', name: 'runbook.md', is_dir: false, size: 20, modified: 't' },
    {
      path: 'ops/sse-reconnect.md',
      name: 'sse-reconnect.md',
      is_dir: false,
      size: 30,
      modified: 't',
    },
  ];

  it('nests files under their directory with directories sorted first', () => {
    const tree = buildTree(entries);
    expect(tree.map((n) => n.name)).toEqual(['ops', 'index.md']);
    const ops = tree[0]!;
    expect(ops.isDir).toBe(true);
    expect(ops.children.map((n) => n.name)).toEqual(['runbook.md', 'sse-reconnect.md']);
  });

  it('synthesizes intermediate directories not listed explicitly', () => {
    const tree = buildTree([
      { path: 'a/b/deep.md', name: 'deep.md', is_dir: false, size: 1, modified: 't' },
    ]);
    expect(tree[0]!.name).toBe('a');
    expect(tree[0]!.children[0]!.name).toBe('b');
    expect(tree[0]!.children[0]!.children[0]!.name).toBe('deep.md');
  });
});

describe('renderMarkdown', () => {
  const resolve = (t: string) => (t === 'backup-retention' ? 'ops/backup-retention.md' : null);

  it('renders a resolved wikilink as a navigable anchor', () => {
    const html = renderMarkdown('see [[backup-retention]] here', resolve);
    expect(html).toContain('class="wikilink"');
    expect(html).toContain('data-wk-nav="ops/backup-retention.md"');
    expect(html).toContain('[[backup-retention]]');
  });

  it('marks an unresolved wikilink missing with a hidden label', () => {
    const html = renderMarkdown('see [[household-systems]] here', resolve);
    expect(html).toContain('wikilink is-missing');
    expect(html).toContain('(missing page)');
    expect(html).not.toContain('data-wk-nav');
  });

  it('honours a pipe alias for the visible text', () => {
    const html = renderMarkdown('see [[backup-retention|retention policy]]', resolve);
    expect(html).toContain('retention policy');
    expect(html).toContain('data-wk-nav="ops/backup-retention.md"');
  });

  it('does not linkify wikilinks inside code and renders block markdown', () => {
    const html = renderMarkdown('`[[backup-retention]]`\n\n## Heading', resolve);
    expect(html).not.toContain('data-wk-nav');
    expect(html).toContain('<h2');
  });

  it('renders a leading frontmatter block as a kv panel ahead of the markdown body', () => {
    const src =
      '---\ntitle: Backup Retention\nowner: platform-team\n---\n# Backup Retention\n\nbody text';
    const html = renderMarkdown(src, resolve);
    expect(html).toContain('<blockquote');
    expect(html).toContain('<th>title</th><td>Backup Retention</td>');
    expect(html).toContain('<th>owner</th><td>platform-team</td>');
    expect(html).toContain('<h1');
    expect(html).toContain('body text');
    // the raw fence markers never leak into the rendered body
    expect(html).not.toMatch(/<p>\s*---/);
  });

  it('renders a nested frontmatter value in a code block, followed by the body', () => {
    const src = '---\ntools:\n  - read_file\n  - run\n---\nbody';
    const html = renderMarkdown(src, resolve);
    expect(html).toContain('<th>tools</th><td><pre><code>- read_file\n- run</code></pre></td>');
    expect(html).toContain('<p>body</p>');
  });

  it('renders normally when there is no frontmatter', () => {
    const html = renderMarkdown('# Just a heading\n\nsome text', resolve);
    expect(html).not.toContain('<blockquote');
    expect(html).toContain('<h1');
  });

  it('falls back to rendering the original content as-is for malformed frontmatter, without crashing', () => {
    const src = '---\nnot really frontmatter\njust prose\n---\nbody';
    expect(() => renderMarkdown(src, resolve)).not.toThrow();
    const html = renderMarkdown(src, resolve);
    expect(html).not.toContain('<blockquote');
    expect(html).toContain('not really frontmatter');
  });

  it('never throws on an unterminated frontmatter fence', () => {
    expect(() => renderMarkdown('---\ntitle: x\nno closing fence', resolve)).not.toThrow();
  });
});
