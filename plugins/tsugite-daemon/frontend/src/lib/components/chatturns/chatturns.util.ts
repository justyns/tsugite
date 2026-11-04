// Pure helpers shared by the chat-turn components. Kept framework-free so they
// run in the fast `unit` vitest project.

import { Marked } from 'marked';
import type { LinkBadge, LinkPreview } from '$lib/stores/linkPreviews.svelte';
import { TESTID } from '$lib/testids';

// `&` is rejected before the first delimiter: the browser decodes entities
// such as `&#106;avascript:` in the href.
const SAFE_HREF = /^(https?:|mailto:|[^:/?#&]*([/?#]|$))/i;

type LinkChip = (href: string) => LinkPreview | undefined;

const GLYPHS: Record<LinkBadge['tone'], string> = {
  ok: '✓',
  warn: '!',
  err: '✕',
  info: '●',
  mute: '○',
};

const escapeHtml = (text: string) =>
  text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

function chipHtml(href: string, { label, status, icon }: LinkPreview): string {
  const tone = status ? ` data-tone="${status.tone}"` : '';
  const aria = escapeHtml(status ? `${label}, ${status.text}` : label);
  // The daemon keeps only an icon path that is safe inside url('...').
  const mask = icon
    ? `<span class="ic-mask" aria-hidden="true" style="--icon: url('${escapeHtml(icon)}')"></span>`
    : '';
  const glyph = status ? `<span class="st" aria-hidden="true">${GLYPHS[status.tone]}</span> ` : '';
  return (
    `<a class="t-chip t-linkchip" data-testid="${TESTID.chatLinkChip}"${tone} href="${escapeHtml(href)}"` +
    ` target="_blank" rel="noopener" aria-label="${aria}">${mask}${glyph}${escapeHtml(label)}</a>`
  );
}

// Set for the duration of one synchronous parse.
let linkChip: LinkChip | undefined;

const md = new Marked({
  renderer: {
    html: ({ text }) => escapeHtml(text),
    link({ href, tokens }) {
      if (!SAFE_HREF.test(href)) return this.parser.parseInline(tokens);
      const preview = linkChip?.(href);
      return preview ? chipHtml(href, preview) : false;
    },
  },
});

/** Render markdown to an HTML string synchronously. Shared by the reasoning
 *  (Think) and prose bubbles so both parse with identical options. Raw HTML in
 *  the source is escaped to visible text, since agent output is untrusted.
 *  `breaks` renders soft line breaks as hard ones. `chip` returns the preview to
 *  render a link as a chip, or undefined to keep it plain. */
export function parseMarkdown(src: string, breaks = false, chip?: LinkChip): string {
  linkChip = chip;
  try {
    return md.parse(src, { async: false, breaks }) as string;
  } finally {
    linkChip = undefined;
  }
}

/** Split a command line into its program (bolded in exec headers) and the rest. */
export function splitCommand(command: string): { program: string; rest: string } {
  const trimmed = command.trimStart();
  const gap = trimmed.indexOf(' ');
  if (gap === -1) return { program: trimmed, rest: '' };
  return { program: trimmed.slice(0, gap), rest: trimmed.slice(gap) };
}

/** Compact token count for the reasoning meta (`· N tokens`): 1200 -> "1.2k". */
export function formatTokens(n: number): string {
  if (n < 1000) return String(n);
  const k = n / 1000;
  const rounded = k >= 100 ? Math.round(k) : Math.round(k * 10) / 10;
  return `${rounded}k`;
}
