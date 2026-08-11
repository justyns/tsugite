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
 *
 * It is also where the shell hands off the broadcast frames it reads, so an open
 * surface hears a daemon event over its bridge instead of opening a second
 * /api/events stream to the same origin.
 */
import { api, type ApiError } from '$lib/api/client';
import type { SSEEvent } from '$lib/api/sse';
import { ICONS, type IconName } from '$lib/components/icon/icons';
import type { ViewMode } from '../../views';

export interface PluginInfo {
  name: string;
  group: string;
  enabled: boolean;
  loaded: boolean;
  error: string | null;
}

/** A plugin-contributed UI surface, as the daemon normalizes it. */
export interface PluginSurface {
  plugin: string;
  kind: string;
  label: string;
  icon: IconName;
  entry: string;
  nav: boolean;
  params: string[];
  /** Broadcast types this surface asked to be forwarded into its frame. The
   *  browser holds one /api/events stream for the whole origin, so a surface
   *  names what it wants instead of opening a second one. */
  events: string[];
  /** What its nav-rail row does: replace the workspace region, or dock beside
   *  whatever is already open there. */
  mode: ViewMode;
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
   *  tab whose plugin may still arrive, so it shows a loading state instead of
   *  flashing the "plugin isn't installed" placeholder. */
  loaded = $state(false);

  // Open plugin surfaces subscribed to the broadcast (many surfaces can watch,
  // and two can be the same kind).
  private eventSinks = new Set<{ types: readonly string[]; sink: (event: SSEEvent) => void }>();

  byKind(kind: string): PluginSurface | undefined {
    return this.surfaces.find((s) => s.kind === kind);
  }

  /** Offer one broadcast frame to the open plugin surfaces that asked for its
   *  type. A surface hears everything its own descriptor declared and nothing
   *  else, including types the shell acts on itself. */
  applyPluginEvent(event: SSEEvent): void {
    for (const { types, sink } of this.eventSinks) if (types.includes(event.type)) sink(event);
  }

  /** Subscribe an open plugin surface to the frame types it declared. Returns an
   *  unbind fn. */
  bindEvents(types: readonly string[], sink: (event: SSEEvent) => void): () => void {
    const entry = { types, sink };
    this.eventSinks.add(entry);
    return () => this.eventSinks.delete(entry);
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
