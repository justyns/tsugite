// Pure helpers shared by the chat-turn components. Kept framework-free so they
// run in the fast `unit` vitest project.

import { Marked } from 'marked';

// `&` is rejected before the first delimiter: the browser decodes entities
// such as `&#106;avascript:` in the href.
const SAFE_HREF = /^(https?:|mailto:|[^:/?#&]*([/?#]|$))/i;

const md = new Marked({
  renderer: {
    html: ({ text }) => text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'),
    link({ href, tokens }) {
      return SAFE_HREF.test(href) ? false : this.parser.parseInline(tokens);
    },
  },
});

/** Render markdown to an HTML string synchronously. Shared by the reasoning
 *  (Think) and prose bubbles so both parse with identical options. Raw HTML in
 *  the source is escaped to visible text, since agent output is untrusted.
 *  `breaks` renders soft line breaks as hard ones. */
export function parseMarkdown(src: string, breaks = false): string {
  return md.parse(src, { async: false, breaks }) as string;
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
