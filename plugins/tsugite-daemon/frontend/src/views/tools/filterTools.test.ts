import { describe, expect, test } from 'vitest';
import { filterTools } from './filterTools';
import type { ToolInfo } from '$lib/stores/tools.svelte';

const registry: ToolInfo[] = [
  { name: 'exec', category: 'run', description: 'Run a shell command.', source: 'builtin' },
  {
    name: 'web.fetch',
    category: 'web',
    description: 'Fetch a URL over HTTP.',
    source: 'plugin',
  },
  {
    name: 'pdf_tables',
    category: 'pdf-extract',
    description: 'Extract tables from a PDF.',
    source: 'plugin',
  },
];

describe('filterTools', () => {
  test('a blank query returns every tool unchanged', () => {
    expect(filterTools(registry, '')).toEqual(registry);
    expect(filterTools(registry, '   ')).toEqual(registry);
  });

  test('matches on name', () => {
    expect(filterTools(registry, 'exec')).toEqual([registry[0]]);
  });

  test('matches on category, case-insensitively', () => {
    expect(filterTools(registry, 'WEB')).toEqual([registry[1]]);
  });

  test('matches on description', () => {
    expect(filterTools(registry, 'shell command')).toEqual([registry[0]]);
  });

  test('matches on source', () => {
    expect(filterTools(registry, 'builtin')).toEqual([registry[0]]);
    expect(filterTools(registry, 'plugin')).toEqual([registry[1], registry[2]]);
  });

  test('multiple whitespace-separated terms use AND semantics', () => {
    expect(filterTools(registry, 'pdf plugin')).toEqual([registry[2]]);
    expect(filterTools(registry, 'pdf builtin')).toEqual([]);
  });

  test('a dot in the query still matches a dotted tool name', () => {
    expect(filterTools(registry, 'web.fetch')).toEqual([registry[1]]);
  });

  test('no match returns an empty array', () => {
    expect(filterTools(registry, 'nonexistent-tool-xyz')).toEqual([]);
  });

  test('tolerates a tool missing optional fields', () => {
    const sparse: ToolInfo[] = [{ name: 'bare' }];
    expect(filterTools(sparse, 'bare')).toEqual(sparse);
    expect(filterTools(sparse, 'nomatch')).toEqual([]);
  });
});
