/**
 * Free-text filter over the tool registry. Whitespace-separated terms use AND
 * semantics against one haystack per tool (name + category + description +
 * source) - same simple grammar as the jobs board's plain free-text terms,
 * minus the `key:value` tokens (this list has no fields worth a mini-grammar).
 */
import type { ToolInfo } from '$lib/stores/tools.svelte';

function haystackOf(tool: ToolInfo): string {
  return [tool.name, tool.category, tool.description, tool.source]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}

export function filterTools(tools: ToolInfo[], query: string): ToolInfo[] {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return tools;
  return tools.filter((tool) => {
    const haystack = haystackOf(tool);
    return terms.every((term) => haystack.includes(term));
  });
}
