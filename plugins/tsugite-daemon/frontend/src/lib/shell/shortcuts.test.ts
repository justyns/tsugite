import { expect, test } from 'vitest';
import { SHORTCUTS, keyLabel } from './shortcuts';

test('every shortcut carries keys, a label, and a known group', () => {
  const groups = new Set(['Global', 'Navigation', 'Chat']);
  for (const s of SHORTCUTS) {
    expect(s.keys.length).toBeGreaterThan(0);
    expect(s.label.length).toBeGreaterThan(0);
    expect(groups.has(s.group)).toBe(true);
  }
});

test('covers all three groups (the overlay renders them sectioned)', () => {
  const groups = new Set(SHORTCUTS.map((s) => s.group));
  expect(groups).toEqual(new Set(['Global', 'Navigation', 'Chat']));
});

test('keyLabel passes literal tokens through unchanged', () => {
  expect(keyLabel('Enter')).toBe('Enter');
  expect(keyLabel('?')).toBe('?');
  expect(keyLabel(']')).toBe(']');
});

test('keyLabel resolves the platform-specific Mod and Alt glyphs', () => {
  expect(['⌘', 'Ctrl']).toContain(keyLabel('Mod'));
  expect(['⌥', 'Alt']).toContain(keyLabel('Alt'));
});
