/**
 * Plugins-metadata store: the extension-point registry (GET /api/plugins ->
 * {plugins:[{name,group,enabled,loaded,error}]}, discovered across every
 * entry-point group). A 404 from this read-only endpoint degrades to an
 * `available:false` empty state (a clean "not exposed by this daemon" seam)
 * rather than an error. Exported as a class instance.
 */
import { api, type ApiError } from '$lib/api/client';

export interface PluginInfo {
  name: string;
  group: string;
  enabled: boolean;
  loaded: boolean;
  error: string | null;
}

export class PluginsMetaStore {
  plugins = $state<PluginInfo[]>([]);
  /** False when the daemon doesn't expose GET /api/plugins (a clean empty-state
   *  seam, distinct from a transient fetch error). */
  available = $state(true);
  loading = $state(false);
  error = $state<string | null>(null);

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await api.get<{ plugins: PluginInfo[] }>('/api/plugins');
      this.plugins = res.plugins;
      this.available = true;
    } catch (err) {
      if ((err as ApiError).status === 404) {
        this.available = false;
        this.plugins = [];
      } else {
        this.error = err instanceof Error ? err.message : String(err);
      }
    } finally {
      this.loading = false;
    }
  }
}

export const pluginsMeta = new PluginsMetaStore();
