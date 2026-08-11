/**
 * The two registries plugin UI surfaces seed: the surface map (which component
 * renders a docked tab) and the nav registry (which rows the rail shows), plus
 * the workspace/full split the shell reads off a row to pick a region.
 */
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { routeShellEvent, type ShellEventSink } from '$lib/api/events';
import type { SSEEvent } from '$lib/api/sse';
import { pluginsMeta, type PluginSurface } from '$lib/stores/pluginsMeta.svelte';
import { surfaceComponent } from './surfaces';
import { allViews, dockedSurface } from './index';
import PluginSurface_ from '$lib/components/plugins/PluginSurface.svelte';
import ChatSurface from './chats/Surface.svelte';

vi.mock('$lib/api/client', () => ({ api: { get: vi.fn() } }));
const { api } = await import('$lib/api/client');

const board: PluginSurface = {
  plugin: 'demo',
  kind: 'plugin/demo/board',
  label: 'Board',
  icon: 'grid',
  entry: '/api/plugins/demo/ui/board.html',
  nav: true,
  params: [],
  events: [],
  mode: 'full',
};

beforeEach(() => {
  pluginsMeta.surfaces = [];
  pluginsMeta.loaded = false;
  vi.mocked(api.get).mockReset();
});

describe('surfaceComponent', () => {
  test('built-in kinds are unaffected by the plugin registry', () => {
    expect(surfaceComponent('chat')).toBe(ChatSurface);
  });

  test('any plugin kind routes there, installed or not, so a tab can explain itself', () => {
    expect(surfaceComponent('plugin/demo/board')).toBe(PluginSurface_);
    expect(surfaceComponent('plugin/uninstalled/thing')).toBe(PluginSurface_);
  });

  test('a kind outside the plugin namespace stays unclaimed', () => {
    expect(surfaceComponent('nonsense')).toBeUndefined();
  });
});

describe('allViews', () => {
  // Captured with an empty registry, so the assertions below hold whether or not
  // this build includes the dev-only gallery row.
  const builtinIds = allViews().map((v) => v.id);

  test('a nav surface appends a rail row whose id is its surface kind', () => {
    pluginsMeta.surfaces = [board];
    expect(allViews().map((v) => v.id)).toEqual([...builtinIds, 'plugin/demo/board']);
  });

  test('a surface that did not ask for a rail row does not get one', () => {
    pluginsMeta.surfaces = [{ ...board, nav: false }];
    expect(allViews().map((v) => v.id)).toEqual(builtinIds);
  });

  test('the rail row carries the surface label and icon as a full view', () => {
    pluginsMeta.surfaces = [board];
    const row = allViews().at(-1)!;
    expect(row.label).toBe('Board');
    expect(row.icon).toBe('grid');
    expect(row.mode).toBe('full');
    expect(row.load).toBeDefined();
  });

  test('a declared workspace mode reaches the rail row', () => {
    pluginsMeta.surfaces = [{ ...board, mode: 'workspace' }];
    const row = allViews().at(-1)!;
    expect(row.mode).toBe('workspace');
    expect(row.load).toBeUndefined();
  });
});

describe('dockedSurface', () => {
  test('a workspace-mode surface docks rather than taking the region', () => {
    pluginsMeta.surfaces = [{ ...board, mode: 'workspace' }];
    expect(dockedSurface('plugin/demo/board')).toEqual({ ...board, mode: 'workspace' });
  });

  test('a full-mode surface keeps taking the whole region', () => {
    pluginsMeta.surfaces = [board];
    expect(dockedSurface('plugin/demo/board')).toBeNull();
  });

  test('a built-in workspace view docks no surface of its own', () => {
    pluginsMeta.surfaces = [{ ...board, mode: 'workspace' }];
    expect(dockedSurface('chats')).toBeNull();
  });
});

describe('daemon event fan-out', () => {
  // The shell owns the one /api/events stream, so this is App's dispatch seam:
  // every frame goes to the shell's router and to the open plugin surfaces.
  const shell: ShellEventSink = { onJobUpdate: vi.fn() };
  const dispatch = (event: SSEEvent) => {
    routeShellEvent(event, shell);
    pluginsMeta.applyPluginEvent(event);
  };

  const unbinds: (() => void)[] = [];
  const bind = (types: string[], sink: (e: SSEEvent) => void) =>
    unbinds.push(pluginsMeta.bindEvents(types, sink));
  afterEach(() => {
    for (const unbind of unbinds.splice(0)) unbind();
  });

  test('a frame the shell does not route reaches every bound surface', () => {
    const one: SSEEvent[] = [];
    const two: SSEEvent[] = [];
    bind(['onlyoffice_document_update'], (e) => one.push(e));
    bind(['onlyoffice_document_update'], (e) => two.push(e));

    dispatch({ type: 'onlyoffice_document_update', data: { path: 'a.docx' } });

    expect(one).toEqual([{ type: 'onlyoffice_document_update', data: { path: 'a.docx' } }]);
    expect(two).toEqual(one);
  });

  test('a shell-routed frame reaches the shell, and no surface that did not ask for it', () => {
    const seen: SSEEvent[] = [];
    bind(['onlyoffice_document_update'], (e) => seen.push(e));

    dispatch({ type: 'job_update', data: { job_id: 'j1' } });

    expect(shell.onJobUpdate).toHaveBeenCalledOnce();
    expect(seen).toEqual([]);
  });

  test('a surface hears a type the shell routes too, since neither consumes the other', () => {
    // Ten types have a shell route, so gating the plugin fan-out on that route
    // means a descriptor naming one of them is answered with silence, and adding
    // a shell route later stops delivering a type some plugin already asked for.
    const seen: SSEEvent[] = [];
    bind(['terminal_state'], (e) => seen.push(e));

    dispatch({ type: 'terminal_state', data: { terminal_id: 't1' } });

    expect(seen).toEqual([{ type: 'terminal_state', data: { terminal_id: 't1' } }]);
  });

  test('a closed surface stops hearing frames', () => {
    const seen: SSEEvent[] = [];
    pluginsMeta.bindEvents(['onlyoffice_document_update'], (e) => seen.push(e))();

    dispatch({ type: 'onlyoffice_document_update', data: {} });

    expect(seen).toEqual([]);
  });
});

describe('load', () => {
  test('reads the surfaces the plugins payload carries', async () => {
    vi.mocked(api.get).mockResolvedValue({ plugins: [], ui_surfaces: [board] });
    await pluginsMeta.load();
    expect(pluginsMeta.byKind('plugin/demo/board')).toEqual(board);
    expect(pluginsMeta.loaded).toBe(true);
  });

  test('an icon the host does not ship falls back to the generic plug', async () => {
    vi.mocked(api.get).mockResolvedValue({
      plugins: [],
      ui_surfaces: [{ ...board, icon: 'not-an-icon' }],
    });
    await pluginsMeta.load();
    expect(pluginsMeta.byKind('plugin/demo/board')?.icon).toBe('plug');
  });

  test('a daemon that serves no surfaces key is not an error', async () => {
    vi.mocked(api.get).mockResolvedValue({ plugins: [] });
    await pluginsMeta.load();
    expect(pluginsMeta.surfaces).toEqual([]);
    expect(pluginsMeta.error).toBeNull();
  });

  test('a failed load still settles, so tabs stop waiting on it', async () => {
    vi.mocked(api.get).mockRejectedValue(new Error('daemon down'));
    await pluginsMeta.load();
    expect(pluginsMeta.loaded).toBe(true);
  });
});
