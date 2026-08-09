import { describe, expect, test } from 'vitest';
import {
  BRIDGE_VERSION,
  initMessage,
  parsePluginMessage,
  surfaceSrc,
  themeMessage,
} from './bridge';
import type { PluginSurface } from '$lib/stores/pluginsMeta.svelte';

const theme = { name: 'mocha', tokens: { '--bg0': '#1e1e2e' } };

const surface: PluginSurface = {
  plugin: 'demo',
  kind: 'plugin/demo/board',
  label: 'Board',
  icon: 'grid',
  entry: '/api/plugins/demo/ui/board.html',
  nav: false,
  params: ['path', 'session'],
};

describe('surfaceSrc', () => {
  test('forwards only the params the surface declared', () => {
    const src = surfaceSrc(surface, { path: 'notes.md', sessionId: 'leaks-otherwise' });
    expect(src).toBe('/api/plugins/demo/ui/board.html?path=notes.md');
  });

  test('a surface declaring no params gets a bare entry URL', () => {
    expect(surfaceSrc({ ...surface, params: [] }, { path: 'notes.md' })).toBe(
      '/api/plugins/demo/ui/board.html',
    );
  });

  test('param values are encoded, not pasted into the URL', () => {
    const src = surfaceSrc(surface, { path: 'a b/c&d.md' });
    expect(src).toBe('/api/plugins/demo/ui/board.html?path=a+b%2Fc%26d.md');
  });

  test('all declared params ride together', () => {
    expect(surfaceSrc(surface, { path: 'a.md', session: 's1' })).toBe(
      '/api/plugins/demo/ui/board.html?path=a.md&session=s1',
    );
  });
});

describe('initMessage', () => {
  test('carries the protocol version, the surface, and the resolved theme', () => {
    const msg = initMessage('plugin/demo/board', { path: 'a.md' }, theme);
    expect(msg).toEqual({
      type: 'tsugite:init',
      version: BRIDGE_VERSION,
      surface: { kind: 'plugin/demo/board', params: { path: 'a.md' } },
      theme,
    });
  });

  test('a theme switch pushes the same token shape on its own', () => {
    expect(themeMessage(theme)).toEqual({ type: 'tsugite:theme', theme });
  });

  test('survives postMessage when params are reactive state', () => {
    // A tab's params reach here as a Svelte $state proxy, and structured clone
    // rejects a proxy - passing one through makes postMessage throw outright.
    const reactive = new Proxy({ path: 'a.md' }, {});
    const msg = initMessage('plugin/demo/board', reactive, theme);
    expect(() => structuredClone(msg)).not.toThrow();
    expect(msg.surface.params).toEqual({ path: 'a.md' });
  });
});

describe('parsePluginMessage', () => {
  test('accepts the ready handshake', () => {
    expect(parsePluginMessage({ type: 'tsugite:ready' })).toEqual({ type: 'tsugite:ready' });
  });

  test('accepts a title and trims it', () => {
    expect(parsePluginMessage({ type: 'tsugite:title', title: '  report.docx  ' })).toEqual({
      type: 'tsugite:title',
      title: 'report.docx',
    });
  });

  test('caps a title that would push the tab strip around', () => {
    const parsed = parsePluginMessage({ type: 'tsugite:title', title: 'x'.repeat(500) });
    expect(parsed).toEqual({ type: 'tsugite:title', title: 'x'.repeat(120) });
  });

  test.each([
    ['a title that is not a string', { type: 'tsugite:title', title: 42 }],
    ['a message type this version does not speak', { type: 'tsugite:navigate', view: 'chats' }],
    ['an untyped object', { title: 'hello' }],
    ['a bare string, e.g. another library sharing the channel', 'hello'],
    ['null', null],
  ])('ignores %s', (_label, data) => {
    expect(parsePluginMessage(data)).toBeNull();
  });
});
