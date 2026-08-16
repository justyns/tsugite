/**
 * Files store: an agent's workspace tree (GET /api/agents/{agent}/workspace),
 * text read/write, and attach-into-chat. The listing is gitignore-filtered and
 * text-only server-side (binaries never appear); a binary read comes back
 * {content:null, is_text:false}. Exported as a class instance.
 */
import { api } from '$lib/api/client';
import { writtenPath } from './fileWrites';

export interface WorkspaceEntry {
  path: string;
  name: string;
  is_dir: boolean;
  /** Files only. */
  size?: number;
  modified?: string;
}

export interface WorkspaceFile {
  path: string;
  content: string | null;
  is_text: boolean;
  size?: number;
}

export interface AttachedFile {
  name: string;
  content_type: string;
  mime_type: string;
  size: number;
  context_attach: boolean;
}

function base(agent: string): string {
  return `/api/agents/${encodeURIComponent(agent)}/workspace`;
}

export class FilesStore {
  agent = $state('');
  subdir = $state('');
  workspaceDir = $state('');
  entries = $state<WorkspaceEntry[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);
  /** Revision counter, so a second write to the same file is still a fresh signal. */
  lastWrite = $state<{ path: string; rev: number } | null>(null);

  applySessionEvent(data: Record<string, unknown>): void {
    const path = writtenPath(data);
    if (path) this.lastWrite = { path, rev: (this.lastWrite?.rev ?? 0) + 1 };
  }

  async list(agent: string, subdir = ''): Promise<void> {
    this.agent = agent;
    this.loading = true;
    this.error = null;
    try {
      const qs = subdir ? `?${new URLSearchParams({ subdir }).toString()}` : '';
      const res = await api.get<{
        entries: WorkspaceEntry[];
        subdir: string;
        workspace_dir: string;
      }>(`${base(agent)}${qs}`);
      this.entries = res.entries;
      this.subdir = res.subdir;
      this.workspaceDir = res.workspace_dir;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }

  async read(agent: string, path: string): Promise<WorkspaceFile> {
    const qs = new URLSearchParams({ path }).toString();
    return api.get<WorkspaceFile>(`${base(agent)}/content?${qs}`);
  }

  async write(agent: string, path: string, content: string): Promise<void> {
    await api.put(`${base(agent)}/content`, { path, content });
  }

  /** Copy a workspace file into uploads/ so it can be attached to a chat turn. */
  async attach(agent: string, path: string): Promise<AttachedFile[]> {
    const qs = new URLSearchParams({ path }).toString();
    const res = await api.post<{ files: AttachedFile[] }>(`${base(agent)}/attach?${qs}`);
    return res.files;
  }
}

export const files = new FilesStore();
