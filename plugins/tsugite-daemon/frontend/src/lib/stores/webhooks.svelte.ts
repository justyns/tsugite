/**
 * Webhooks store: inbound webhook tokens (GET /api/webhooks), create + delete.
 * The public delivery endpoint (/webhook/{token}) is token-in-path and not part
 * of this authed surface. No SSE broadcast exists for webhooks, so the list is
 * only mutated by these calls. Exported as a class instance.
 */
import { api } from '$lib/api/client';

export interface Webhook {
  token: string;
  source: string;
  created_at: string;
}

export class WebhooksStore {
  list = $state<Webhook[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await api.get<{ webhooks: Webhook[] }>('/api/webhooks/');
      this.list = res.webhooks;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }

  async create(opts: { source: string; token?: string }): Promise<Webhook> {
    const webhook = await api.post<Webhook>('/api/webhooks/', opts);
    this.list = [webhook, ...this.list];
    return webhook;
  }

  async remove(token: string): Promise<void> {
    await api.del(`/api/webhooks/${encodeURIComponent(token)}`);
    this.list = this.list.filter((w) => w.token !== token);
  }
}

export const webhooks = new WebhooksStore();
