import { describe, expect, test } from 'vitest';
import type { MdFile } from '$lib/stores/agentsMeta.svelte';
import { agentLabel, chatAgents, chatSettings } from './launcher';

const file = (name: string, chat: boolean, source = 'project', isDefault = false): MdFile => ({
  path: `/${source}/${name}.md`,
  name,
  source,
  readonly: false,
  description: '',
  chat,
  default: isDefault,
});

describe('launcher', () => {
  const files = [
    file('default', true, 'builtin', true),
    file('hollis', true),
    file('nightly', false),
  ];

  test('chatAgents keeps only chat-capable files', () => {
    expect(chatAgents(files).map((f) => f.name)).toEqual(['default', 'hollis']);
  });

  test('chatSettings is empty when every field is a default', () => {
    expect(
      chatSettings({
        agent: files[0]!,
        model: null,
        effort: 'default',
      }),
    ).toEqual({});
  });

  test('chatSettings carries a non-default agent path, model and effort', () => {
    expect(
      chatSettings({
        agent: files[1]!,
        model: 'openai:gpt-5.4',
        effort: 'low',
      }),
    ).toEqual({
      agent_file: '/project/hollis.md',
      model: 'openai:gpt-5.4',
      reasoning_effort: 'low',
    });
  });

  test('agentLabel adds the source only when two files share a name', () => {
    const dup = [file('helper', true, 'builtin'), file('helper', true, 'project')];
    expect(agentLabel(dup[0]!, dup)).toBe('helper (builtin)');
    expect(agentLabel(files[1]!, files)).toBe('hollis');
  });
});
