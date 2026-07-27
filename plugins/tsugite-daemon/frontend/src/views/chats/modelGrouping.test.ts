import { describe, expect, test } from 'vitest';
import {
  groupModelsByProvider,
  formatContext,
  formatPrice,
  type PickerModel,
} from './modelGrouping';

const m = (id: string, provider: string | null, extra: Partial<PickerModel> = {}): PickerModel => ({
  id,
  provider,
  ...extra,
});

describe('groupModelsByProvider', () => {
  test('buckets models under their provider', () => {
    const groups = groupModelsByProvider([
      m('anthropic:claude-opus-4-5', 'anthropic'),
      m('openai:gpt-5.4', 'openai'),
      m('anthropic:claude-sonnet-5', 'anthropic'),
    ]);
    expect(groups.map((g) => g.provider)).toEqual(['anthropic', 'openai']);
    expect(groups[0]!.models.map((x) => x.id)).toEqual([
      'anthropic:claude-opus-4-5',
      'anthropic:claude-sonnet-5',
    ]);
  });

  test('orders preferred providers first, the rest alphabetically', () => {
    const groups = groupModelsByProvider([
      m('zeta:z', 'zeta'),
      m('openai:o', 'openai'),
      m('acme:a', 'acme'),
      m('anthropic:c', 'anthropic'),
    ]);
    // anthropic + openai are preferred (in that order); acme, zeta trail A-Z.
    expect(groups.map((g) => g.provider)).toEqual(['anthropic', 'openai', 'acme', 'zeta']);
  });

  test('sorts a null-provider bucket last under "other"', () => {
    const groups = groupModelsByProvider([m('mystery', null), m('openai:o', 'openai')]);
    expect(groups.map((g) => g.provider)).toEqual(['openai', 'other']);
    expect(groups[1]!.models.map((x) => x.id)).toEqual(['mystery']);
  });

  test('flattening the groups preserves every model exactly once', () => {
    const input = [m('openai:b', 'openai'), m('anthropic:a', 'anthropic'), m('openai:a', 'openai')];
    const flat = groupModelsByProvider(input).flatMap((g) => g.models);
    expect(flat).toHaveLength(input.length);
    expect(new Set(flat.map((x) => x.id))).toEqual(new Set(input.map((x) => x.id)));
  });

  test('empty input yields no groups', () => {
    expect(groupModelsByProvider([])).toEqual([]);
  });

  test('grouping a pre-filtered subset only shows providers that survived the filter', () => {
    const all = [
      m('anthropic:claude-opus-4-5', 'anthropic'),
      m('openai:gpt-5.4', 'openai'),
      m('openai:gpt-5.4-mini', 'openai'),
    ];
    const filtered = all.filter((x) => x.id.includes('gpt'));
    const groups = groupModelsByProvider(filtered);
    expect(groups.map((g) => g.provider)).toEqual(['openai']);
  });
});

describe('formatContext', () => {
  test('renders thousands as k and millions as M', () => {
    expect(formatContext(200_000)).toBe('200k');
    expect(formatContext(128_000)).toBe('128k');
    expect(formatContext(1_000_000)).toBe('1M');
    expect(formatContext(1_500_000)).toBe('1.5M');
  });

  test('is empty for missing or non-positive values', () => {
    expect(formatContext(0)).toBe('');
    expect(formatContext(null)).toBe('');
    expect(formatContext(undefined)).toBe('');
  });
});

describe('formatPrice', () => {
  test('renders compact input / output dollars, trimming trailing zeros', () => {
    expect(formatPrice(5, 25)).toBe('$5 / $25');
    expect(formatPrice(3, 15)).toBe('$3 / $15');
    expect(formatPrice(0.15, 0.6)).toBe('$0.15 / $0.6');
  });

  test('is empty when either side is unpriced', () => {
    expect(formatPrice(null, null)).toBe('');
    expect(formatPrice(5, null)).toBe('');
    expect(formatPrice(null, 25)).toBe('');
    expect(formatPrice(undefined, undefined)).toBe('');
  });
});
