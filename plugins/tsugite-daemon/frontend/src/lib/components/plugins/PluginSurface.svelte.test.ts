/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import PluginSurface from './PluginSurface.svelte';
import { pluginsMeta, type PluginSurface as SurfaceDef } from '$lib/stores/pluginsMeta.svelte';
import { theme } from '$lib/stores/theme.svelte';
import { auth } from '$lib/stores/auth.svelte';
// The bridge ships resolved token values, so the test page needs the real sheet.
import '../../../styles/tokens.css';

// A plugin page is just HTML the daemon serves; a blob URL stands in for its
// route so the handshake runs against a real cross-document postMessage.
function pluginPage(body: string): string {
  return URL.createObjectURL(
    new Blob([`<!doctype html><meta charset="utf-8"><script>${body}<\/script>`], {
      type: 'text/html',
    }),
  );
}

const RECORD_AND_READY = `
  window.received = [];
  addEventListener('message', (e) => {
    window.received.push(e.data);
    if (e.data && e.data.type === 'tsugite:init') parent.postMessage({ type: 'tsugite:ready' }, '*');
  });
`;

const urls: string[] = [];

function seed(entry: string, over: Partial<SurfaceDef> = {}): SurfaceDef {
  urls.push(entry);
  const surface: SurfaceDef = {
    plugin: 'demo',
    kind: 'plugin/demo/board',
    label: 'Board',
    icon: 'grid',
    entry,
    nav: false,
    params: [],
    ...over,
  };
  pluginsMeta.surfaces = [surface];
  pluginsMeta.loaded = true;
  return surface;
}

/** The bridge lands a frame or two after the iframe's load event. */
async function handshake(): Promise<void> {
  await expect
    .poll(() => document.querySelector('[data-phase]')?.getAttribute('data-phase'))
    .toBe('ready');
}

beforeEach(() => {
  theme.set('mocha');
  auth.token = 'tsu_test-token';
});

afterEach(() => {
  pluginsMeta.surfaces = [];
  pluginsMeta.loaded = false;
  for (const url of urls.splice(0)) URL.revokeObjectURL(url);
});

test('init carries the protocol version and the resolved theme tokens', async () => {
  seed(pluginPage(RECORD_AND_READY));
  await render(PluginSurface, { kind: 'plugin/demo/board', params: { path: 'q4.docx' } });
  await handshake();

  const frame = document.querySelector('iframe')!;
  const init = (frame.contentWindow as unknown as { received: Record<string, never>[] })
    .received[0];
  expect(init).toMatchObject({
    type: 'tsugite:init',
    version: 1,
    surface: { kind: 'plugin/demo/board', params: { path: 'q4.docx' } },
    theme: { name: 'mocha' },
    // Handed over so the page never digs the token out of host storage itself.
    token: 'tsu_test-token',
  });
  // Resolved values, not names: a plugin styles itself from these alone.
  const tokens = (init as unknown as { theme: { tokens: Record<string, string> } }).theme.tokens;
  expect(tokens['--bg0']).toMatch(/^(#|rgb|oklch|hsl)/);
  expect(Object.keys(tokens).length).toBeGreaterThan(20);
});

test('a theme switch re-skins a live surface', async () => {
  seed(pluginPage(RECORD_AND_READY));
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {} });
  await handshake();

  theme.set('latte');

  const frame = document.querySelector('iframe')!;
  const themePushes = () =>
    (
      frame.contentWindow as unknown as { received: { type: string; theme?: { name: string } }[] }
    ).received.filter((m) => m.type === 'tsugite:theme');
  await expect.poll(() => themePushes().length).toBe(1);
  expect(themePushes()[0]?.theme?.name).toBe('latte');
});

test('a plugin can set its own tab title', async () => {
  seed(
    pluginPage(`
      addEventListener('message', (e) => {
        if (e.data && e.data.type === 'tsugite:init') {
          parent.postMessage({ type: 'tsugite:ready' }, '*');
          parent.postMessage({ type: 'tsugite:title', title: 'q4-report.docx' }, '*');
        }
      });
    `),
  );
  const setTitle = vi.fn();
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {}, setTitle });
  await handshake();

  await expect.poll(() => setTitle.mock.calls.length).toBeGreaterThan(0);
  expect(setTitle).toHaveBeenCalledWith('q4-report.docx');
});

test('a tab that outlived its plugin says so instead of framing nothing', async () => {
  pluginsMeta.surfaces = [];
  pluginsMeta.loaded = true;
  await render(PluginSurface, { kind: 'plugin/uninstalled/thing', params: {} });

  await expect
    .element(page.getByText("This tab's plugin isn't installed on this daemon."))
    .toBeInTheDocument();
  expect(document.querySelector('iframe')).toBeNull();
});

test('an unrecognized kind waits rather than accusing a plugin that may still load', async () => {
  pluginsMeta.surfaces = [];
  pluginsMeta.loaded = false;
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {} });

  expect(document.body.textContent).not.toContain("isn't installed");
  expect(document.querySelector('iframe')).toBeNull();
});
