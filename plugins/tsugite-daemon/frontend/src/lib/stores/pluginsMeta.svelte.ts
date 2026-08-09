/**
 * Plugins-metadata store: the extension-point registry (GET /api/plugins ->
 * {plugins:[{name,group,enabled,loaded,error}], ui_surfaces:[...]}, discovered
 * across every entry-point group). A 404 from this read-only endpoint degrades to
 * an `available:false` empty state (a clean "not exposed by this daemon" seam)
 * rather than an error. Exported as a class instance.
 *
 * The same payload carries the UI surfaces adapter plugins declare, which feed
 * the surface registry (views/surfaces.ts) and the nav rail (views/index.ts). The
 * shell loads this at boot so a persisted plugin tab resolves on the first frame.
 */
import { api, type ApiError } from '$lib/api/client';
import { ICONS, type IconName } from '$lib/components/icon/icons';

export interface PluginInfo {
  name: string;
  group: string;
  enabled: boolean;
  loaded: boolean;
  error: string | null;
}

/** A plugin-contributed UI surface, as the daemon normalizes it: `kind` is the
 *  `plugin/<name>/<kind>` identifier used as the mux tab kind, the nav view id,
 *  and the hash route; `entry` is a path under the plugin's own mount. */
export interface PluginSurface {
  plugin: string;
  kind: string;
  label: string;
  icon: IconName;
  entry: string;
  nav: boolean;
  params: string[];
}

interface RawSurface extends Omit<PluginSurface, 'icon'> {
  icon: string;
}

/** Plugins name an icon by string; one the host doesn't ship falls back to the
 *  generic plug rather than rendering an empty glyph. */
function toSurface(raw: RawSurface): PluginSurface {
  return { ...raw, icon: raw.icon in ICONS ? (raw.icon as IconName) : 'plug' };
}

export class PluginsMetaStore {
  plugins = $state<PluginInfo[]>([]);
  surfaces = $state<PluginSurface[]>([]);
  /** False when the daemon doesn't expose GET /api/plugins (a clean empty-state
   *  seam, distinct from a transient fetch error). */
  available = $state(true);
  loading = $state(false);
  error = $state<string | null>(null);
  /** True once a load has settled. Until then an unrecognized surface kind is a
   *  tab whose plugin may still arrive, so it renders nothing instead of
   *  flashing the "plugin unavailable" placeholder. */
  loaded = $state(false);

  byKind(kind: string): PluginSurface | undefined {
    return this.surfaces.find((s) => s.kind === kind);
  }

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await api.get<{ plugins: PluginInfo[]; ui_surfaces?: RawSurface[] }>(
        '/api/plugins',
      );
      this.plugins = res.plugins;
      this.surfaces = (res.ui_surfaces ?? []).map(toSurface);
      this.available = true;
    } catch (err) {
      if ((err as ApiError).status === 404) {
        this.available = false;
        this.plugins = [];
        this.surfaces = [];
      } else {
        this.error = err instanceof Error ? err.message : String(err);
      }
    } finally {
      this.loading = false;
      this.loaded = true;
    }
  }
}

export const pluginsMeta = new PluginsMetaStore();
