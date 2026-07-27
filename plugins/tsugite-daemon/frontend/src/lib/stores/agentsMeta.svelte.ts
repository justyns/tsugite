/**
 * Agents-metadata store: the agent roster (GET /api/agents) plus the editable
 * agent-file and skill-file browsers (GET/PUT /api/agent-files, /api/skill-files)
 * and the skill-load issues list. Files carry a `readonly` flag (builtins) the
 * editor must honour. Exported as a class instance.
 */
import { api } from '$lib/api/client';

export interface AgentMeta {
  name: string;
  agent_file: string;
  workspace_dir: string;
  running_tasks: number;
}

export interface MdFile {
  path: string;
  name: string;
  source: string;
  readonly: boolean;
  description: string;
}

export interface MdFileContent {
  path: string;
  content: string;
  readonly: boolean;
}

export interface SkillIssue {
  [key: string]: unknown;
}

export class AgentsMetaStore {
  agents = $state<AgentMeta[]>([]);
  agentFiles = $state<MdFile[]>([]);
  skillFiles = $state<MdFile[]>([]);
  skillIssues = $state<SkillIssue[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await api.get<{ agents: AgentMeta[] }>('/api/agents');
      this.agents = res.agents;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }

  async loadAgentFiles(): Promise<void> {
    const res = await api.get<{ files: MdFile[] }>('/api/agent-files');
    this.agentFiles = res.files;
  }

  async readAgentFile(path: string): Promise<MdFileContent> {
    const qs = new URLSearchParams({ path }).toString();
    return api.get<MdFileContent>(`/api/agent-files/content?${qs}`);
  }

  async saveAgentFile(path: string, content: string): Promise<void> {
    await api.put('/api/agent-files/content', { path, content });
  }

  async loadSkillFiles(): Promise<void> {
    const res = await api.get<{ files: MdFile[] }>('/api/skill-files');
    this.skillFiles = res.files;
  }

  async readSkillFile(path: string): Promise<MdFileContent> {
    const qs = new URLSearchParams({ path }).toString();
    return api.get<MdFileContent>(`/api/skill-files/content?${qs}`);
  }

  async saveSkillFile(path: string, content: string): Promise<void> {
    await api.put('/api/skill-files/content', { path, content });
  }

  async loadSkillIssues(): Promise<void> {
    const res = await api.get<{ issues: SkillIssue[] }>('/api/skills/issues');
    this.skillIssues = res.issues;
  }
}

export const agentsMeta = new AgentsMetaStore();
