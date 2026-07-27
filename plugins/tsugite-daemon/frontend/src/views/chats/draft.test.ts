import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { readDraft, writeDraft, clearDraft, readDraftStaged, writeDraftStaged } from './draft';

// draft.ts persists via $lib/storage, which no-ops unless `window` exists; the
// node unit env has none, so stub a minimal window.localStorage.
const store = new Map<string, string>();
beforeEach(() => {
  store.clear();
  vi.stubGlobal('window', {
    localStorage: {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => store.set(k, v),
      removeItem: (k: string) => store.delete(k),
    },
  });
});
afterEach(() => vi.unstubAllGlobals());

test('staged attachments + context items round-trip, isolated per session', () => {
  const staged = {
    attachments: [{ id: 'a', name: 'photo.jpg', size: '12 KB' }],
    contextItems: [{ key: 'session:x', label: 'chat', value: 'kind: tsugite session' }],
  };
  writeDraftStaged('s1', staged);
  expect(readDraftStaged('s1')).toEqual(staged);
  expect(readDraftStaged('s2')).toEqual({ attachments: [], contextItems: [] });
});

test('empty staged clears the key; corrupt/missing reads as empty', () => {
  writeDraftStaged('s1', { attachments: [{ id: 'a', name: 'x' }], contextItems: [] });
  writeDraftStaged('s1', { attachments: [], contextItems: [] });
  expect(readDraftStaged('s1')).toEqual({ attachments: [], contextItems: [] });
  store.set('tsugite_draft_staged_s1', 'not json');
  expect(readDraftStaged('s1')).toEqual({ attachments: [], contextItems: [] });
});

test('clearDraft removes both the text and the staged items', () => {
  writeDraft('s1', 'hi');
  writeDraftStaged('s1', { attachments: [{ id: 'a', name: 'x' }], contextItems: [] });
  clearDraft('s1');
  expect(readDraft('s1')).toBe('');
  expect(readDraftStaged('s1')).toEqual({ attachments: [], contextItems: [] });
});
