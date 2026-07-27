/**
 * Secrets store: write-only secret management. The server returns names only
 * (never values) - GET /api/secrets lists names, POST upserts (add AND rotate
 * share one endpoint), DELETE removes. The list can be legitimately empty even
 * when secrets resolve, because the default `env` backend persists nothing and
 * always lists []; a Secrets UI should message backend capability rather than
 * assume names are always present. Exported as a class instance.
 */
import { api } from '$lib/api/client';

export class SecretsStore {
  names = $state<string[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await api.get<{ secrets: string[] }>('/api/secrets/');
      this.names = res.secrets;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }

  /** Add or rotate: same upsert endpoint. Value is write-only - never stored
   *  client-side, only the name is tracked. */
  async set(name: string, value: string): Promise<void> {
    await api.post(`/api/secrets/${encodeURIComponent(name)}`, { value });
    if (!this.names.includes(name)) this.names = [...this.names, name].sort();
  }

  async remove(name: string): Promise<void> {
    await api.del(`/api/secrets/${encodeURIComponent(name)}`);
    this.names = this.names.filter((n) => n !== name);
  }
}

export const secrets = new SecretsStore();
