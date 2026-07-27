/**
 * Tools store: the tool registry (GET /api/tools -> {tools:[{name, category,
 * description, source}]}). A 404 from this read-only endpoint degrades to an
 * `available:false` empty state (a clean "not exposed" seam) instead of
 * surfacing an error. Exported as a class instance.
 */
import { api, type ApiError } from '$lib/api/client';

export interface ToolInfo {
  name: string;
  category?: string;
  description?: string;
  /** 'builtin' for core tsugite tools, 'plugin' for package-registered ones. */
  source?: string;
  [key: string]: unknown;
}

export class ToolsStore {
  tools = $state<ToolInfo[]>([]);
  /** False when the daemon doesn't expose GET /api/tools. */
  available = $state(true);
  loading = $state(false);
  error = $state<string | null>(null);

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await api.get<{ tools: ToolInfo[] }>('/api/tools');
      this.tools = res.tools;
      this.available = true;
    } catch (err) {
      if ((err as ApiError).status === 404) {
        this.available = false;
        this.tools = [];
      } else {
        this.error = err instanceof Error ? err.message : String(err);
      }
    } finally {
      this.loading = false;
    }
  }
}

export const tools = new ToolsStore();
