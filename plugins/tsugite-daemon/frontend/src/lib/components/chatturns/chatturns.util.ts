// Pure helpers shared by the chat-turn components. Kept framework-free so they
// run in the fast `unit` vitest project.

import { marked } from 'marked';

/** Render trusted markdown to an HTML string synchronously. Shared by the
 *  reasoning (Think) and prose bubbles so both parse with identical options. */
export function parseMarkdown(src: string): string {
  return marked.parse(src, { async: false }) as string;
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
