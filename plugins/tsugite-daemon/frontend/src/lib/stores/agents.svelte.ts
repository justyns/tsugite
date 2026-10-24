/**
 * Agents store: the daemon's agent identities (GET /api/agents), which own
 * sessions via `SessionRow.agent_id`. Exported as a class instance.
 */
import { api } from '$lib/api/client';

export interface Agent {
  id: string;
  name: string;
  workspace: string;
  template: string;
  model: string | null;
  reasoning_effort: string | null;
  color: string | null;
  created_at: string;
  is_default: boolean;
}

export const AGENT_COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'teal', 'pink'];

const AGENT_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/;

/** Mirrors the daemon's validate_agent_id; null when the id is acceptable. */
export function validateAgentId(id: string): string | null {
  return AGENT_ID_PATTERN.test(id)
    ? null
    : "Use lowercase letters, digits, '-' and '_' (max 64 characters), starting with a letter or digit";
}

export interface AgentCreate {
  id: string;
  name?: string;
  template?: string;
  model?: string | null;
  reasoning_effort?: string | null;
  color?: string | null;
  identity?: string;
}

export type AgentPatch = Partial<
  Pick<Agent, 'name' | 'template' | 'model' | 'reasoning_effort' | 'color'>
>;

export class AgentsStore {
  list = $state<Agent[]>([]);
  error = $state<string | null>(null);

  private inflight: Promise<void> | null = null;
  private missed = new Set<string>();

  async load(): Promise<void> {
    this.inflight ??= this.fetchAgents();
    await this.inflight;
  }

  private async fetchAgents(): Promise<void> {
    this.error = null;
    try {
      const res = await api.get<{ agents: Agent[] }>('/api/agents');
      this.list = res.agents;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.inflight = null;
    }
  }

  async create(body: AgentCreate): Promise<Agent> {
    const agent = await api.post<Agent>('/api/agents', body);
    await this.refresh();
    return agent;
  }

  async update(id: string, patch: AgentPatch): Promise<Agent> {
    const agent = await api.patch<Agent>(`/api/agents/${encodeURIComponent(id)}`, patch);
    await this.refresh();
    return agent;
  }

  private async refresh(): Promise<void> {
    // A load already in flight may predate the change.
    await this.inflight;
    await this.load();
  }

  /** The agent's main chat session id, created on first open. */
  async openMain(id: string): Promise<string> {
    const res = await api.post<{ id: string }>(`/api/agents/${encodeURIComponent(id)}/main`);
    return res.id;
  }

  /** Refetches when a session names an agent the list lacks, until one fetch
   *  succeeds without it. */
  async resolve(ids: Iterable<string>): Promise<void> {
    if (this.unknown(ids).length === 0) return;
    // A load already in flight may predate the agent.
    await this.inflight;
    const unknown = this.unknown(ids);
    if (unknown.length === 0) return;
    await this.load();
    if (this.error !== null) return;
    for (const id of this.unknown(unknown)) this.missed.add(id);
  }

  private unknown(ids: Iterable<string>): string[] {
    const known = new Set(this.list.map((a) => a.id));
    return [...new Set(ids)].filter((id) => !known.has(id) && !this.missed.has(id));
  }
}

export const agents = new AgentsStore();
