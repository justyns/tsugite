/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import PluginSurface from './PluginSurface.svelte';
import { pluginsMeta, type PluginSurface as SurfaceDef } from '$lib/stores/pluginsMeta.svelte';
import { theme } from '$lib/stores/theme.svelte';
import { auth } from '$lib/stores/auth.svelte';
import { TESTID } from '$lib/testids';
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
    events: [],
    mode: 'full',
    ...over,
  };
  pluginsMeta.surfaces = [surface];
  pluginsMeta.loaded = true;
  return surface;
}

/** The bridge lands a frame or two after the iframe's load event. */
async function handshake(): Promise<void> {
  await expect
    .poll(() => page.getByTestId(TESTID.pluginSurface).element().getAttribute('data-phase'))
    .toBe('ready');
}

beforeEach(() => {
  theme.set('mocha');
  auth.token = 'tsu_test-token';
  auth.userId = 'desk-viewer';
});

afterEach(() => {
  pluginsMeta.surfaces = [];
  pluginsMeta.loaded = false;
  for (const url of urls.splice(0)) URL.revokeObjectURL(url);
});

test('init carries the viewing user, so a surface can attribute what the human does', async () => {
  seed(pluginPage(RECORD_AND_READY));
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {} });
  await handshake();

  const frame = document.querySelector<HTMLIFrameElement>(
    `[data-testid="${TESTID.pluginSurface}"] iframe`,
  )!;
  const init = (frame.contentWindow as unknown as { received: { user?: string }[] }).received[0];
  expect(init?.user).toBe('desk-viewer');
});

test('init carries the protocol version and the resolved theme tokens', async () => {
  seed(pluginPage(RECORD_AND_READY));
  await render(PluginSurface, { kind: 'plugin/demo/board', params: { path: 'q4.docx' } });
  await handshake();

  const frame = document.querySelector<HTMLIFrameElement>(
    `[data-testid="${TESTID.pluginSurface}"] iframe`,
  )!;
  const init = (frame.contentWindow as unknown as { received: Record<string, never>[] })
    .received[0];
  expect(init).toMatchObject({
    type: 'tsugite:init',
    version: 1,
    surface: { kind: 'plugin/demo/board', params: { path: 'q4.docx' } },
    theme: { name: 'mocha' },
    token: 'tsu_test-token',
  });
  const tokens = (init as unknown as { theme: { tokens: Record<string, string> } }).theme.tokens;
  expect(tokens['--bg0']).toMatch(/^(#|rgb|oklch|hsl)/);
  expect(Object.keys(tokens).length).toBeGreaterThan(20);
});

test('a theme switch re-skins a live surface', async () => {
  seed(pluginPage(RECORD_AND_READY));
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {} });
  await handshake();

  theme.set('latte');

  const frame = document.querySelector<HTMLIFrameElement>(
    `[data-testid="${TESTID.pluginSurface}"] iframe`,
  )!;
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

test('a click inside the frame claims pane focus for the surface', async () => {
  seed(
    pluginPage(`
      addEventListener('message', (e) => {
        if (e.data && e.data.type === 'tsugite:init') {
          parent.postMessage({ type: 'tsugite:ready' }, '*');
          parent.postMessage({ type: 'tsugite:focus' }, '*');
        }
      });
    `),
  );
  const focusPane = vi.fn();
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {}, focusPane });
  await handshake();

  await expect.poll(() => focusPane.mock.calls.length).toBeGreaterThan(0);
});

test('focus landing in a frame the plugin page cannot see still claims the pane', async () => {
  // A surface whose content is itself a cross-origin frame (a document editor)
  // never sees the click that lands in it, so it can send nothing. The host sees
  // its own window blur with focus sitting on the surface's iframe.
  seed(pluginPage(RECORD_AND_READY));
  const focusPane = vi.fn();
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {}, focusPane });
  await handshake();

  document
    .querySelector<HTMLIFrameElement>(`[data-testid="${TESTID.pluginSurface}"] iframe`)!
    .focus();
  // Whether the harness page itself holds focus is not ours to control, so the
  // blur a real click into the frame raises is delivered by hand; what is under
  // test is the host reading it as a claim.
  window.dispatchEvent(new Event('blur'));

  await expect.poll(() => focusPane.mock.calls.length).toBeGreaterThan(0);
});

function received(): { type: string; event?: { type: string; data: unknown } }[] {
  const frame = document.querySelector<HTMLIFrameElement>(
    `[data-testid="${TESTID.pluginSurface}"] iframe`,
  )!;
  return (frame.contentWindow as unknown as { received: { type: string }[] }).received;
}

/** postMessage delivery to one window is ordered, so a theme push sent after an
 *  event push settles the question of absence: once the theme lands, anything
 *  posted before it has landed too. */
async function afterATheme(): Promise<void> {
  theme.set('latte');
  await expect.poll(() => received().filter((m) => m.type === 'tsugite:theme').length).toBe(1);
}

const eventsSeen = () => received().filter((m) => m.type === 'tsugite:event');

test('a surface receives the daemon events its descriptor declared', async () => {
  seed(pluginPage(RECORD_AND_READY), { events: ['onlyoffice_document_update'] });
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {} });
  await handshake();

  pluginsMeta.applyPluginEvent({
    type: 'onlyoffice_document_update',
    seq: 4,
    data: { path: 'q4.docx' },
  });

  await expect.poll(() => eventsSeen().length).toBe(1);
  expect(eventsSeen()[0]?.event).toEqual({
    type: 'onlyoffice_document_update',
    data: { path: 'q4.docx' },
  });
});

test('an event type the surface did not declare never reaches it', async () => {
  seed(pluginPage(RECORD_AND_READY), { events: ['onlyoffice_document_update'] });
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {} });
  await handshake();

  // Deliberately a type with no shell route of its own, so the descriptor is the
  // only thing that could have stopped it.
  pluginsMeta.applyPluginEvent({ type: 'other_plugin_update', data: { id: 'x1' } });
  await afterATheme();

  expect(eventsSeen()).toEqual([]);
});

test('a surface that declared no events is not a window onto the daemon feed', async () => {
  seed(pluginPage(RECORD_AND_READY));
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {} });
  await handshake();

  pluginsMeta.applyPluginEvent({ type: 'onlyoffice_document_update', data: { path: 'q4.docx' } });
  await afterATheme();

  expect(eventsSeen()).toEqual([]);
});

test('a tab that outlived its plugin says so instead of framing nothing', async () => {
  pluginsMeta.surfaces = [];
  pluginsMeta.loaded = true;
  await render(PluginSurface, { kind: 'plugin/uninstalled/thing', params: {} });

  await expect.element(page.getByTestId(TESTID.pluginSurfaceMissing)).toBeInTheDocument();
  expect(
    document.querySelector<HTMLIFrameElement>(`[data-testid="${TESTID.pluginSurface}"] iframe`),
  ).toBeNull();
});

test('an unrecognized kind waits rather than accusing a plugin that may still load', async () => {
  pluginsMeta.surfaces = [];
  pluginsMeta.loaded = false;
  await render(PluginSurface, { kind: 'plugin/demo/board', params: {} });

  expect(document.querySelector(`[data-testid="${TESTID.pluginSurfaceMissing}"]`)).toBeNull();
  expect(
    document.querySelector<HTMLIFrameElement>(`[data-testid="${TESTID.pluginSurface}"] iframe`),
  ).toBeNull();
});
