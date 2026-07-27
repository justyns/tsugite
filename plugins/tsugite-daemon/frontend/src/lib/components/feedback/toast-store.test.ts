import { describe, expect, test } from 'vitest';
import { isSticky, MAX_TOASTS, toasts } from './toast-store.svelte';

describe('isSticky', () => {
  test('err always persists regardless of the sticky flag', () => {
    expect(isSticky('err', false)).toBe(true);
    expect(isSticky('err', undefined)).toBe(true);
  });

  test('other variants respect the explicit sticky flag', () => {
    expect(isSticky('ok', true)).toBe(true);
    expect(isSticky('ok', false)).toBe(false);
    expect(isSticky('warn', undefined)).toBe(false);
  });
});

describe('toasts store', () => {
  test('push appends an entry and returns its id', () => {
    toasts.items.length = 0;
    const id = toasts.push('ok', 'Job done', { body: '5/5 criteria passed' });
    expect(toasts.items).toHaveLength(1);
    expect(toasts.items[0]).toMatchObject({
      id,
      variant: 'ok',
      title: 'Job done',
      body: '5/5 criteria passed',
    });
  });

  test('err entries resolve sticky=true even when not explicitly requested', () => {
    toasts.items.length = 0;
    toasts.push('err', 'Schedule failed');
    expect(toasts.items[0]!.sticky).toBe(true);
  });

  test('drops the oldest entry once more than MAX_TOASTS are queued', () => {
    toasts.items.length = 0;
    const ids = Array.from({ length: MAX_TOASTS + 2 }, (_, i) => toasts.push('info', `toast ${i}`));
    expect(toasts.items).toHaveLength(MAX_TOASTS);
    expect(toasts.items.map((t) => t.id)).toEqual(ids.slice(-MAX_TOASTS));
  });

  test('dismiss removes only the matching entry', () => {
    toasts.items.length = 0;
    const first = toasts.push('ok', 'first');
    const second = toasts.push('ok', 'second');
    toasts.dismiss(first);
    expect(toasts.items.map((t) => t.id)).toEqual([second]);
  });

  test('dismiss is a no-op for an id that is not queued', () => {
    toasts.items.length = 0;
    toasts.push('ok', 'first');
    expect(() => toasts.dismiss(999_999)).not.toThrow();
    expect(toasts.items).toHaveLength(1);
  });
});
