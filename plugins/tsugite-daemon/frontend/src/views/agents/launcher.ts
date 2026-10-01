import type { MdFile } from '$lib/stores/agentsMeta.svelte';
import type { ChatSettingsPatch } from '$lib/stores/sessions.svelte';

export const chatAgents = (files: MdFile[]): MdFile[] => files.filter((f) => f.chat);

/** Settings a new chat needs applied before its first turn; empty when everything is a default. */
export function chatSettings(opts: {
  agent: MdFile;
  model: string | null;
  effort: string;
}): ChatSettingsPatch {
  const patch: ChatSettingsPatch = {};
  if (!opts.agent.default) patch.agent_file = opts.agent.path;
  if (opts.model) patch.model = opts.model;
  if (opts.effort !== 'default') patch.reasoning_effort = opts.effort;
  return patch;
}

/** Picker label: the agent name, with its source when two files share a name. */
export const agentLabel = (f: MdFile, all: MdFile[]): string =>
  all.filter((o) => o.name === f.name).length > 1 ? `${f.name} (${f.source})` : f.name;
