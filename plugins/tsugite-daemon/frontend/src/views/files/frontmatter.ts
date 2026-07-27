/**
 * Frontmatter helpers shared by the two views that render a document's leading
 * `---`-fenced block: the files wiki (`wiki.ts`'s `renderMarkdown`, over arbitrary
 * workspace docs) and the agent builder (`agents/agentFrontmatter.ts`, over `.md`
 * agent files). `splitFrontmatter` lives here - both need the exact same
 * fence-detection (BOM/CRLF tolerance, unterminated-fence handling) and it would
 * drift if duplicated. It lives in `files/` rather than `lib/` because `wiki.ts`,
 * its primary consumer, already sets the precedent of view-local pure-function
 * modules; `agentFrontmatter.ts` imports it from here instead of redefining it.
 *
 * `parseFrontmatterEntries` is display-only and deliberately NOT a YAML parser:
 * it splits the block into top-level `key:` entries and tells a one-line scalar
 * from a multi-line/nested value (sequence, map, or block scalar), which is all a
 * "render this frontmatter as a kv panel" view needs. The agent builder's Form
 * tab needs real typed values (booleans, numbers, lists) for its structured
 * display and keeps using its own recursive `parseYamlSubset` for that.
 */

/** Split a `---`-fenced frontmatter block from the markdown body below it. */
export function splitFrontmatter(src: string): {
  fm: string;
  body: string;
  hasFrontmatter: boolean;
} {
  // A frontmatter block must open on the very first line. Tolerate a UTF-8 BOM
  // and trailing whitespace on the fence line, and both LF and CRLF newlines.
  const text = src.replace(/^﻿/, '');
  const lines = text.split('\n');
  if (lines[0]?.replace(/\r$/, '').trim() !== '---') {
    return { fm: '', body: src, hasFrontmatter: false };
  }
  for (let i = 1; i < lines.length; i++) {
    if (lines[i]!.replace(/\r$/, '').trim() === '---') {
      const fm = lines.slice(1, i).join('\n');
      const body = lines
        .slice(i + 1)
        .join('\n')
        .replace(/^\n+/, '');
      return { fm, body, hasFrontmatter: true };
    }
  }
  // Unterminated fence: treat the whole thing as frontmatter, no body.
  return { fm: lines.slice(1).join('\n'), body: '', hasFrontmatter: true };
}

export interface FrontmatterEntry {
  key: string;
  /** Display text: the inline scalar, or the dedented raw block for a nested value. */
  value: string;
  /** True when `value` is raw YAML text (a block sequence, map, or scalar) rather than one scalar. */
  nested: boolean;
}

const TOP_LEVEL_KEY = /^([A-Za-z0-9_.-]+):(?:[ \t](.*))?$/;
const BLOCK_SCALAR_INDICATORS = new Set(['|', '|-', '|+', '>', '>-', '>+']);

function indentOf(line: string): number {
  return line.length - line.trimStart().length;
}

/**
 * Split a frontmatter block into top-level entries for display. Any top-level
 * line that isn't shaped like `key:` or `key: value` means this isn't really a
 * frontmatter block (a stray `---`-delimited paragraph, for instance) - bail out
 * to an empty array rather than guess, so the caller can fall back to rendering
 * the original content untouched.
 */
export function parseFrontmatterEntries(fm: string): FrontmatterEntry[] {
  const lines = fm.replace(/\r\n/g, '\n').split('\n');
  const entries: FrontmatterEntry[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    const trimmed = line.trim();
    if (trimmed === '' || trimmed.startsWith('#')) {
      i++;
      continue;
    }
    const m = TOP_LEVEL_KEY.exec(line);
    if (!m) return [];
    const key = m[1]!;
    const inline = (m[2] ?? '').trim();
    i++;

    const expectsBlock = inline === '' || BLOCK_SCALAR_INDICATORS.has(inline);
    if (!expectsBlock) {
      entries.push({ key, value: inline, nested: false });
      continue;
    }

    const block: string[] = [];
    while (i < lines.length && (lines[i] === '' || /^[ \t]/.test(lines[i]!))) {
      block.push(lines[i]!);
      i++;
    }
    while (block.length && block[block.length - 1]!.trim() === '') block.pop();

    if (block.length === 0) {
      entries.push({ key, value: '', nested: false });
      continue;
    }
    const indent = Math.min(...block.filter((l) => l.trim() !== '').map((l) => indentOf(l)));
    const dedented = block.map((l) => (l.trim() === '' ? '' : l.slice(indent))).join('\n');
    entries.push({ key, value: dedented, nested: true });
  }
  return entries;
}

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Render frontmatter entries as a compact, visually distinct kv panel: a
 * `<blockquote>` (the callout-style box both consumers already token) wrapping a
 * `<table>` of `key: value` rows, nested values as a code block. Deliberately
 * plain tags with no bespoke classes, so it renders correctly wherever a
 * markdown document's `blockquote`/`table`/`pre` are already themed - the files
 * wiki's `.doc-md` and the agent builder's document prose treatment alike -
 * without either view needing new CSS for it.
 */
export function renderFrontmatterHtml(entries: FrontmatterEntry[]): string {
  if (entries.length === 0) return '';
  const rows = entries
    .map((e) => {
      const val = e.nested ? `<pre><code>${escapeHtml(e.value)}</code></pre>` : escapeHtml(e.value);
      return `<tr><th>${escapeHtml(e.key)}</th><td>${val}</td></tr>`;
    })
    .join('');
  return `<blockquote class="tsu-fm"><table><tbody>${rows}</tbody></table></blockquote>`;
}
