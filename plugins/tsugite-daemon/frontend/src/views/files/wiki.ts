/**
 * Client-side wiki logic for the workspace view: parsing `[[wikilinks]]` and
 * `#tags`, resolving link targets against the workspace file list, inverting the
 * link graph into backlinks, and rendering markdown with those affordances.
 *
 * The daemon exposes no server-side wiki index,
 * so the index is computed here over the file list + contents and cached by the
 * view. Everything except `renderMarkdown` is pure and unit-tested.
 *
 * `renderMarkdown` also strips a leading `---`-fenced frontmatter block (see
 * `./frontmatter`) and renders it as its own kv panel ahead of the markdown
 * body, instead of letting it fall into `marked` as broken paragraph/list soup.
 */
import { Marked } from 'marked';
import type { WorkspaceEntry } from '$lib/stores/files.svelte';
import {
  splitFrontmatter,
  parseFrontmatterEntries,
  renderFrontmatterHtml,
  escapeHtml,
} from './frontmatter';

export interface WikiRef {
  target: string;
  alias: string;
}

export interface Backlink {
  file: string;
  snippet: string;
}

export interface Heading {
  depth: number;
  text: string;
}

export interface IndexDoc {
  path: string;
  content: string;
}

export interface WikiIndex {
  files: string[];
  backlinks: Map<string, Backlink[]>;
  tagCounts: Map<string, number>;
  tagsByFile: Map<string, string[]>;
  resolve(target: string): string | null;
}

export interface TreeNode {
  name: string;
  path: string;
  isDir: boolean;
  size?: number;
  modified?: string;
  children: TreeNode[];
}

/** Blank out fenced and inline code so link/tag scanners ignore their contents. */
function stripCode(md: string): string {
  return md.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '');
}

export function parseWikiLinks(md: string): WikiRef[] {
  const out: WikiRef[] = [];
  const re = /\[\[([^\]|\n]+?)(?:\|([^\]\n]+?))?\]\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(stripCode(md)))) {
    const target = (m[1] ?? '').trim();
    out.push({ target, alias: (m[2] ?? m[1] ?? '').trim() });
  }
  return out;
}

export function parseTags(md: string): string[] {
  const tags: string[] = [];
  const seen = new Set<string>();
  const add = (raw: string) => {
    const t = raw.toLowerCase();
    if (t && !seen.has(t)) {
      seen.add(t);
      tags.push(t);
    }
  };
  const src = stripCode(md);
  // `tags: [a, b]` yaml-style list (bare words, no leading #)
  const listMatch = /^tags:\s*\[([^\]]*)\]/im.exec(src);
  if (listMatch) {
    (listMatch[1] ?? '')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .forEach(add);
  }
  // inline hashtags: preceded by start/whitespace/paren, must begin with a letter
  const re = /(?:^|[\s(])#([A-Za-z][\w-]*)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src))) add(m[1] ?? '');
  return tags;
}

export function parseHeadings(md: string): Heading[] {
  const out: Heading[] = [];
  let inFence = false;
  for (const line of md.split('\n')) {
    if (/^\s*```/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const m = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    if (m) out.push({ depth: (m[1] ?? '').length, text: (m[2] ?? '').trim() });
  }
  return out;
}

const normPage = (s: string): string => s.replace(/\.md$/i, '').toLowerCase();

export function resolveTarget(target: string, files: string[]): string | null {
  const t = normPage(target.trim());
  if (!t) return null;
  for (const f of files) if (normPage(f) === t) return f;
  const base = t.split('/').pop();
  for (const f of files) if (normPage(f).split('/').pop() === base) return f;
  return null;
}

function snippetFor(content: string, ref: WikiRef): string {
  const needle = `[[${ref.target}`;
  const line = content.split('\n').find((l) => l.includes(needle)) ?? '';
  const clean = line.replace(/\s+/g, ' ').trim();
  return clean.length > 120 ? `${clean.slice(0, 117)}…` : clean;
}

/** A paths-only index: wikilink resolution needs just the file list, so it
 *  works without reading a single file. The content-derived maps (backlinks,
 *  tags) stay empty until an explicit workspace scan builds the full index. */
export function buildPathsIndex(paths: string[]): WikiIndex {
  return {
    files: paths,
    backlinks: new Map(),
    tagCounts: new Map(),
    tagsByFile: new Map(),
    resolve: (t) => resolveTarget(t, paths),
  };
}

export function buildIndex(docs: IndexDoc[]): WikiIndex {
  const files = docs.map((d) => d.path);
  const backlinks = new Map<string, Backlink[]>();
  const tagCounts = new Map<string, number>();
  const tagsByFile = new Map<string, string[]>();
  for (const doc of docs) {
    for (const ref of parseWikiLinks(doc.content)) {
      const targetPath = resolveTarget(ref.target, files);
      if (!targetPath || targetPath === doc.path) continue;
      const list = backlinks.get(targetPath) ?? [];
      list.push({ file: doc.path, snippet: snippetFor(doc.content, ref) });
      backlinks.set(targetPath, list);
    }
    const tags = parseTags(doc.content);
    tagsByFile.set(doc.path, tags);
    for (const t of tags) tagCounts.set(t, (tagCounts.get(t) ?? 0) + 1);
  }
  return { files, backlinks, tagCounts, tagsByFile, resolve: (t) => resolveTarget(t, files) };
}

export function relatedNotes(
  path: string,
  tagsByFile: Map<string, string[]>,
  minShared = 2,
): { path: string; shared: number }[] {
  const mine = new Set(tagsByFile.get(path) ?? []);
  const out: { path: string; shared: number }[] = [];
  for (const [p, tags] of tagsByFile) {
    if (p === path) continue;
    const shared = tags.filter((t) => mine.has(t)).length;
    if (shared >= minShared) out.push({ path: p, shared });
  }
  return out.sort((a, b) => b.shared - a.shared || a.path.localeCompare(b.path));
}

export function stripTagsLine(md: string): string {
  return md.replace(/^[ \t]*tags:.*(?:\r?\n+|$)/im, '');
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatMtime(iso: string | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function breadcrumbs(path: string): string[] {
  return path.split('/').filter(Boolean);
}

export function buildTree(entries: WorkspaceEntry[]): TreeNode[] {
  const root: TreeNode = { name: '', path: '', isDir: true, children: [] };
  const dirs = new Map<string, TreeNode>([['', root]]);
  const ensureDir = (path: string): TreeNode => {
    const existing = dirs.get(path);
    if (existing) return existing;
    const parts = path.split('/');
    const name = parts.pop() ?? path;
    const parent = ensureDir(parts.join('/'));
    const node: TreeNode = { name, path, isDir: true, children: [] };
    parent.children.push(node);
    dirs.set(path, node);
    return node;
  };
  for (const e of entries) {
    if (e.is_dir) {
      ensureDir(e.path);
    } else {
      const parts = e.path.split('/');
      parts.pop();
      ensureDir(parts.join('/')).children.push({
        name: e.name,
        path: e.path,
        isDir: false,
        size: e.size,
        modified: e.modified,
        children: [],
      });
    }
  }
  const sort = (node: TreeNode) => {
    node.children.sort((a, b) => Number(b.isDir) - Number(a.isDir) || a.name.localeCompare(b.name));
    node.children.forEach(sort);
  };
  sort(root);
  return root.children;
}

// Isolated marked instance so the wikilink extension never leaks into the chat
// Prose renderer (which shares the package-level `marked` singleton). A
// module-scoped resolver is set for the duration of each synchronous parse.
let activeResolver: (target: string) => string | null = () => null;
const md = new Marked();
md.use({
  extensions: [
    {
      name: 'wikilink',
      level: 'inline',
      start(src: string) {
        const i = src.indexOf('[[');
        return i < 0 ? undefined : i;
      },
      tokenizer(src: string) {
        const m = /^\[\[([^\]|\n]+?)(?:\|([^\]\n]+?))?\]\]/.exec(src);
        if (!m) return undefined;
        return {
          type: 'wikilink',
          raw: m[0] ?? '',
          target: (m[1] ?? '').trim(),
          alias: (m[2] ?? m[1] ?? '').trim(),
        };
      },
      renderer(token) {
        const wl = token as unknown as { target: string; alias: string };
        const path = activeResolver(wl.target);
        const missing = path == null;
        const label = `[[${escapeHtml(wl.alias)}]]`;
        if (missing) {
          return `<a class="wikilink is-missing" role="link" tabindex="0" data-wk-missing="${escapeHtml(wl.target)}" title="Missing page">${label}<span class="vh"> (missing page)</span></a>`;
        }
        return `<a class="wikilink" role="link" tabindex="0" data-wk-nav="${escapeHtml(path)}" title="${escapeHtml(path)}">${label}</a>`;
      },
    },
  ],
});

/**
 * Render workspace markdown to HTML with `[[wikilinks]]` turned into navigable
 * anchors (`data-wk-nav="<path>"`) or missing-page markers. `resolve` maps a
 * link target to a workspace path, or null when no such file exists.
 *
 * Content originates from the trusted daemon workspace; like the chat `Prose`
 * component this renderer owns no sanitization policy.
 */
export function renderMarkdown(
  content: string,
  resolve: (target: string) => string | null,
): string {
  activeResolver = resolve;
  try {
    const { fm, body, hasFrontmatter } = splitFrontmatter(content);
    const entries = hasFrontmatter ? parseFrontmatterEntries(fm) : [];
    // Zero parsed entries means either no frontmatter, or a leading `---` that
    // isn't really a frontmatter block (malformed) - render the original
    // content untouched rather than guess.
    if (entries.length === 0) return md.parse(content, { async: false }) as string;
    return renderFrontmatterHtml(entries) + (md.parse(body, { async: false }) as string);
  } finally {
    activeResolver = () => null;
  }
}
