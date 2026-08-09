/**
 * The two registries plugin UI surfaces seed: the surface map (which component
 * renders a docked tab) and the nav registry (which rows the rail shows).
 */
import { beforeEach, describe, expect, test, vi } from 'vitest';
import { pluginsMeta, type PluginSurface } from '$lib/stores/pluginsMeta.svelte';
import { surfaceComponent } from './surfaces';
import { allViews } from './index';
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
