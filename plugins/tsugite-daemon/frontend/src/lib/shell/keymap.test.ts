import { expect, test } from 'vitest';
import { resolveShellShortcut } from './keymap';

const base = {
  key: '',
  metaKey: false,
  ctrlKey: false,
  shiftKey: false,
  altKey: false,
  typing: false,
};

test('cmd/ctrl + k toggles the palette', () => {
  expect(resolveShellShortcut({ ...base, key: 'k', metaKey: true })).toBe('toggle-palette');
  expect(resolveShellShortcut({ ...base, key: 'K', ctrlKey: true })).toBe('toggle-palette');
});

test('cmd/ctrl + shift + o starts a new chat', () => {
  expect(resolveShellShortcut({ ...base, key: 'o', metaKey: true, shiftKey: true })).toBe(
    'new-chat',
  );
  expect(resolveShellShortcut({ ...base, key: 'O', ctrlKey: true, shiftKey: true })).toBe(
    'new-chat',
  );
});

test('the new-chat chord fires even while a text field has focus', () => {
  expect(
    resolveShellShortcut({ ...base, key: 'o', metaKey: true, shiftKey: true, typing: true }),
  ).toBe('new-chat');
});

test('cmd/ctrl + o without shift is not new-chat', () => {
  expect(resolveShellShortcut({ ...base, key: 'o', metaKey: true })).toBeNull();
  expect(resolveShellShortcut({ ...base, key: 'o', ctrlKey: true })).toBeNull();
});

test('cmd/ctrl + shift + k still toggles the palette', () => {
  expect(resolveShellShortcut({ ...base, key: 'k', metaKey: true, shiftKey: true })).toBe(
    'toggle-palette',
  );
});

test('slash opens the palette, but only when focus is not in a field', () => {
  expect(resolveShellShortcut({ ...base, key: '/' })).toBe('open-palette');
  expect(resolveShellShortcut({ ...base, key: '/', typing: true })).toBeNull();
});

test('cmd/ctrl + comma opens settings', () => {
  expect(resolveShellShortcut({ ...base, key: ',', metaKey: true })).toBe('open-settings');
});

test('bare and unrelated keys are ignored', () => {
  expect(resolveShellShortcut({ ...base, key: 'k' })).toBeNull();
  expect(resolveShellShortcut({ ...base, key: 'a', metaKey: true })).toBeNull();
  expect(resolveShellShortcut({ ...base, key: '/', metaKey: true })).toBeNull();
});

test('? (shift+/) opens the shortcuts overlay, but yields to a text field', () => {
  expect(resolveShellShortcut({ ...base, key: '?' })).toBe('show-help');
  expect(resolveShellShortcut({ ...base, key: '?', typing: true })).toBeNull();
});

test('alt + arrow steps the chat list, even while a text field has focus', () => {
  expect(resolveShellShortcut({ ...base, key: 'ArrowDown', altKey: true })).toBe('next-chat');
  expect(resolveShellShortcut({ ...base, key: 'ArrowUp', altKey: true })).toBe('prev-chat');
  expect(resolveShellShortcut({ ...base, key: 'ArrowDown', altKey: true, typing: true })).toBe(
    'next-chat',
  );
});

test('a bare arrow (no alt) is not a chat step', () => {
  expect(resolveShellShortcut({ ...base, key: 'ArrowDown' })).toBeNull();
  expect(resolveShellShortcut({ ...base, key: 'ArrowUp' })).toBeNull();
});

test('cmd/ctrl + shift + bracket cycles mux tabs', () => {
  // Chromium reports the SHIFTED glyph for a bracket keypress: shift+] is '}',
  // shift+[ is '{' - so the resolver must match those, not the bare bracket.
  expect(resolveShellShortcut({ ...base, key: '}', metaKey: true, shiftKey: true })).toBe(
    'next-tab',
  );
  expect(resolveShellShortcut({ ...base, key: '{', ctrlKey: true, shiftKey: true })).toBe(
    'prev-tab',
  );
  // The bare-bracket form resolves too (defensive across layouts / key reports).
  expect(resolveShellShortcut({ ...base, key: ']', metaKey: true, shiftKey: true })).toBe(
    'next-tab',
  );
  expect(resolveShellShortcut({ ...base, key: '[', ctrlKey: true, shiftKey: true })).toBe(
    'prev-tab',
  );
});

test('a bracket without the mod chord is not a tab switch', () => {
  expect(resolveShellShortcut({ ...base, key: '}', shiftKey: true })).toBeNull();
  expect(resolveShellShortcut({ ...base, key: ']' })).toBeNull();
  expect(resolveShellShortcut({ ...base, key: '[' })).toBeNull();
});

test('the tab and new-chat mod+shift chords do not shadow each other', () => {
  expect(resolveShellShortcut({ ...base, key: 'o', metaKey: true, shiftKey: true })).toBe(
    'new-chat',
  );
  expect(resolveShellShortcut({ ...base, key: 'k', metaKey: true, shiftKey: true })).toBe(
    'toggle-palette',
  );
});
