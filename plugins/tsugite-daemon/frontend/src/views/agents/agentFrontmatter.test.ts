import { describe, it, expect } from 'vitest';
import {
  splitFrontmatter,
  parseYamlSubset,
  coerceScalar,
  parseAgentFile,
  summarizeAgent,
} from './agentFrontmatter';

describe('splitFrontmatter', () => {
  it('separates a fenced frontmatter block from the body', () => {
    const src = '---\nname: odyn\n---\nYou are odyn.\n\n- rule one\n';
    const r = splitFrontmatter(src);
    expect(r.hasFrontmatter).toBe(true);
    expect(r.fm).toBe('name: odyn');
    expect(r.body).toBe('You are odyn.\n\n- rule one\n');
  });

  it('reports no frontmatter when the file does not open with a fence', () => {
    const src = 'just a prompt body\nno frontmatter';
    const r = splitFrontmatter(src);
    expect(r.hasFrontmatter).toBe(false);
    expect(r.fm).toBe('');
    expect(r.body).toBe(src);
  });

  it('tolerates CRLF newlines and a BOM', () => {
    const src = '﻿---\r\nname: x\r\n---\r\nbody\r\n';
    const r = splitFrontmatter(src);
    expect(r.hasFrontmatter).toBe(true);
    expect(r.fm.replace(/\r/g, '')).toBe('name: x');
  });

  it('treats an unterminated fence as all-frontmatter', () => {
    const r = splitFrontmatter('---\nname: x\nmodel: y');
    expect(r.hasFrontmatter).toBe(true);
    expect(r.body).toBe('');
  });
});

describe('coerceScalar', () => {
  it('coerces bools, null, ints, floats', () => {
    expect(coerceScalar('true')).toBe(true);
    expect(coerceScalar('false')).toBe(false);
    expect(coerceScalar('null')).toBe(null);
    expect(coerceScalar('~')).toBe(null);
    expect(coerceScalar('40')).toBe(40);
    expect(coerceScalar('1.5')).toBe(1.5);
  });

  it('strips single and double quotes, keeping inner # and colons', () => {
    expect(coerceScalar('"memory/{{ today() }}.md"')).toBe('memory/{{ today() }}.md');
    expect(coerceScalar("'a: b # c'")).toBe('a: b # c');
  });

  it('drops an unquoted trailing comment', () => {
    expect(coerceScalar('public # visibility')).toBe('public');
  });

  it('parses inline flow lists and empty collections', () => {
    expect(coerceScalar('[]')).toEqual([]);
    expect(coerceScalar('{}')).toEqual({});
    expect(coerceScalar('[a, b, c]')).toEqual(['a', 'b', 'c']);
  });
});

describe('parseYamlSubset', () => {
  it('parses scalars with type coercion', () => {
    const fm = parseYamlSubset('name: odyn\nmax_turns: 40\nspawnable: false\nmodel: null');
    expect(fm).toEqual({ name: 'odyn', max_turns: 40, spawnable: false, model: null });
  });

  it('parses a block sequence of scalars', () => {
    const fm = parseYamlSubset('tools:\n  - read_file\n  - "@terminal"\n  - run');
    expect(fm.tools).toEqual(['read_file', '@terminal', 'run']);
  });

  it('parses a block sequence of maps (prefetch)', () => {
    const fm = parseYamlSubset(
      'prefetch:\n  - tool: get_skills\n    args: {}\n    assign: available_skills\n  - tool: other\n    assign: x',
    );
    expect(fm.prefetch).toEqual([
      { tool: 'get_skills', args: {}, assign: 'available_skills' },
      { tool: 'other', assign: 'x' },
    ]);
  });

  it('parses a one-level nested map (sandbox)', () => {
    const fm = parseYamlSubset('sandbox:\n  enabled: true\n  no_network: true');
    expect(fm.sandbox).toEqual({ enabled: true, no_network: true });
  });

  it('parses a block scalar (instructions |) preserving line breaks', () => {
    const fm = parseYamlSubset('instructions: |\n  line one\n  line two\nmodel: x');
    expect(fm.instructions).toBe('line one\nline two');
    expect(fm.model).toBe('x');
  });

  it('folds a > block scalar onto one line', () => {
    const fm = parseYamlSubset('description: >\n  a long\n  folded value');
    expect(fm.description).toBe('a long folded value');
  });

  it('ignores comment lines and blank lines between keys', () => {
    const fm = parseYamlSubset('# header comment\nname: x\n\n# another\nmodel: y');
    expect(fm).toEqual({ name: 'x', model: 'y' });
  });

  it('does not throw on an empty or whitespace-only block', () => {
    expect(parseYamlSubset('')).toEqual({});
    expect(parseYamlSubset('   \n  ')).toEqual({});
  });
});

const ODYN = `---
name: odyn
description: Primary interactive operator agent for the console.
extends: default
model: claude_code:claude-opus-4-8
reasoning_effort: medium
max_turns: 40
visibility: public
spawnable: true
tools:
  - read_file
  - run
  - "@secrets"
  - "@terminal"
attachments:
  - MEMORY.md
  - "memory/{{ today() }}.md"
auto_load_skills:
  - response-patterns
allowed_secrets:
  - JIRA_TOKEN
  - GH_TOKEN
prefetch:
  - tool: get_skills_for_template
    args: {}
    assign: available_skills
run_if: mention || schedule
---
You are odyn, the primary interactive agent.

- keep session topics short
`;

describe('summarizeAgent over a realistic agent file', () => {
  const parsed = parseAgentFile(ODYN);
  const s = summarizeAgent(parsed.frontmatter);

  it('splits body from frontmatter', () => {
    expect(parsed.body.startsWith('You are odyn')).toBe(true);
    expect(parsed.hasFrontmatter).toBe(true);
  });

  it('extracts scalar fields', () => {
    expect(s.name).toBe('odyn');
    expect(s.model).toBe('claude_code:claude-opus-4-8');
    expect(s.extends).toBe('default');
    expect(s.effort).toBe('medium');
    expect(s.maxTurns).toBe(40);
    expect(s.visibility).toBe('public');
    expect(s.spawnable).toBe(true);
    expect(s.runIf).toBe('mention || schedule');
  });

  it('flags @namespace tokens in the tools list', () => {
    expect(s.tools).toEqual([
      { name: 'read_file', namespace: false },
      { name: 'run', namespace: false },
      { name: '@secrets', namespace: true },
      { name: '@terminal', namespace: true },
    ]);
  });

  it('separates string attachments from spec entries', () => {
    expect(s.attachments).toEqual(['MEMORY.md', 'memory/{{ today() }}.md']);
    expect(s.attachmentSpecs).toBe(0);
  });

  it('lists allowed secrets and prefetch entries', () => {
    expect(s.allowedSecrets).toEqual(['JIRA_TOKEN', 'GH_TOKEN']);
    expect(s.prefetch).toEqual([{ tool: 'get_skills_for_template', assign: 'available_skills' }]);
  });

  it('reports no unsurfaced extra keys', () => {
    expect(s.extraKeys).toEqual([]);
  });
});

describe('summarizeAgent edge cases', () => {
  it('counts dict-form attachment specs separately from string paths', () => {
    const fm = parseYamlSubset(
      'attachments:\n  - MEMORY.md\n  - path: notes/*.md\n    mode: index\n    assign: notes',
    );
    const s = summarizeAgent(fm);
    expect(s.attachments).toEqual(['MEMORY.md']);
    expect(s.attachmentSpecs).toBe(1);
  });

  it('surfaces nested sandbox config', () => {
    const s = summarizeAgent(parseYamlSubset('sandbox:\n  enabled: true\n  no_network: true'));
    expect(s.sandbox).toEqual({ enabled: true, no_network: true });
  });

  it('collects unknown frontmatter keys into extraKeys', () => {
    const s = summarizeAgent(parseYamlSubset('name: x\nmystery_field: 1\nanother: two'));
    expect(s.extraKeys.sort()).toEqual(['another', 'mystery_field']);
  });

  it('handles a minimal agent with an empty tools list', () => {
    const s = summarizeAgent(parseYamlSubset('name: scratch\nmax_turns: 5\ntools: []'));
    expect(s.tools).toEqual([]);
    expect(s.maxTurns).toBe(5);
  });
});
