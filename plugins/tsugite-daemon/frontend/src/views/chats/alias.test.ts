import { describe, expect, it } from 'vitest';
import { isValidAlias, suggestAlias } from './alias';

describe('isValidAlias', () => {
  it('accepts what the daemon accepts', () => {
    expect(isValidAlias('daily')).toBe(true);
    expect(isValidAlias('a')).toBe(true);
    expect(isValidAlias('A_b-9')).toBe(true);
    expect(isValidAlias('a'.repeat(64))).toBe(true);
  });

  it('rejects what the daemon rejects', () => {
    expect(isValidAlias('')).toBe(false);
    expect(isValidAlias('_x')).toBe(false);
    expect(isValidAlias('-x')).toBe(false);
    expect(isValidAlias('has space')).toBe(false);
    // A dot is legal in a webhook source but not in an alias.
    expect(isValidAlias('a.b')).toBe(false);
    expect(isValidAlias('a'.repeat(65))).toBe(false);
  });
});

describe('suggestAlias', () => {
  it('slugs a prose title', () => {
    expect(suggestAlias('Fix the login bug')).toBe('fix-the-login-bug');
  });

  it('collapses the punctuation a raw user message carries', () => {
    // A title of 60 chars or fewer is the user's own text verbatim, so it can
    // hold anything they typed.
    expect(suggestAlias('Deploy: "v2" / hotfix\n')).toBe('deploy-v2-hotfix');
  });

  it('drops combining marks rather than splitting the word they sit in', () => {
    expect(suggestAlias('naïve approach')).toBe('naive-approach');
    expect(suggestAlias('Ångström')).toBe('angstrom');
  });

  it('strips leading and trailing dashes', () => {
    expect(suggestAlias('-- 2026 roadmap --')).toBe('2026-roadmap');
  });

  it('spends the length budget on content, not on leading punctuation', () => {
    const suggestion = suggestAlias(`!!!${'ab '.repeat(40)}`);
    expect(suggestion.startsWith('ab-ab')).toBe(true);
    expect(suggestion.length).toBeLessThanOrEqual(64);
  });

  it('gives nothing when the title holds nothing an alias can use', () => {
    expect(suggestAlias('???')).toBe('');
    expect(suggestAlias('日本語')).toBe('');
    expect(suggestAlias('')).toBe('');
  });
});
