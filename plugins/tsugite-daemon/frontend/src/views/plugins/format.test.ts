import { describe, expect, test } from 'vitest';
import { groupLabel, pluginKey, sortPlugins } from './format';
import type { PluginInfo } from '$lib/stores/pluginsMeta.svelte';

function plugin(overrides: Partial<PluginInfo> = {}): PluginInfo {
  return {
    name: 'a',
    group: 'tsugite.tools',
    enabled: true,
    loaded: false,
    error: null,
    ...overrides,
  };
}

describe('groupLabel', () => {
  test('strips the tsugite. entry-point-group prefix', () => {
    expect(groupLabel('tsugite.adapters')).toBe('adapters');
  });

  test('leaves a group with no tsugite. prefix untouched', () => {
    expect(groupLabel('other.group')).toBe('other.group');
  });
});

describe('pluginKey', () => {
  test('combines group and name so the same name in two groups stays distinct', () => {
    expect(pluginKey({ group: 'tsugite.adapters', name: 'cc_driver' })).toBe(
      'tsugite.adapters:cc_driver',
    );
    expect(pluginKey({ group: 'tsugite.tools', name: 'cc_driver' })).toBe(
      'tsugite.tools:cc_driver',
    );
  });
});

describe('sortPlugins', () => {
  test('orders by group then name and returns a new array (does not mutate the input)', () => {
    const input = [
      plugin({ name: 'web', group: 'tsugite.plugins' }),
      plugin({ name: 'bwrap', group: 'tsugite.sandbox' }),
      plugin({ name: 'pty', group: 'tsugite.plugins' }),
    ];
    const sorted = sortPlugins(input);
    expect(sorted.map((p) => `${p.group}:${p.name}`)).toEqual([
      'tsugite.plugins:pty',
      'tsugite.plugins:web',
      'tsugite.sandbox:bwrap',
    ]);
    expect(input.map((p) => p.name)).toEqual(['web', 'bwrap', 'pty']);
  });

  test('is stable for an already-sorted or empty list', () => {
    expect(sortPlugins([])).toEqual([]);
    const a = plugin({ name: 'a', group: 'tsugite.plugins' });
    const b = plugin({ name: 'b', group: 'tsugite.plugins' });
    expect(sortPlugins([a, b])).toEqual([a, b]);
  });
});
